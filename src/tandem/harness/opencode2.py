"""OpenCode 2.0.21 session projection storage.

Native import creates sessions. Idle shadow appends reserve aggregate sequence
numbers atomically, like SessionTransfer.import, without fabricating journal
records. A fresh standalone CLI reads these projections on the next handoff.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from ..events import AssistantMessage, NormalizedEvent, SessionContext, SystemEvent, Thinking, ToolCall, ToolResult, UserMessage
from ..opencode_binding import validate_pair_prefix, validate_prefix
from . import opencode as v1
from .base import ShadowBusy

_run = subprocess.run
mint_id = v1.mint_id
db_path = v1.db_path

connect = v1.connect


@contextmanager
def database(db):
    with closing(connect(db)) as conn, conn:
        yield conn


_ZERO_TOKENS = {"input": 0, "output": 0, "reasoning": 0, "cache": {"read": 0, "write": 0}}
_REQUIRED = {
    "session_v2": {"id", "model", "time_suspended", "time_compacting", "time_idle", "time_updated"},
    "session_message": {"id", "session_id", "type", "seq", "time_created", "time_updated", "data"},
    "session_pending": {"session_id"},
    "session_inbox": {"session_id", "type", "delivery"},
    "event_sequence": {"aggregate_id", "seq"},
}


class LegacySessionUnsupported(ValueError):
    """Retained v1 identities have no verified v2 cursor/history contract."""


def _table_exists(conn, table):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def require_fresh_session(conn, sid):
    if not conn.in_transaction:
        conn.execute("BEGIN")
    if not _table_exists(conn, "session"):
        if _table_exists(conn, "session_v2") and conn.execute("SELECT 1 FROM session_v2 WHERE id=?", (sid,)).fetchone():
            validate_prefix(conn, sid)
        return
    if conn.execute("SELECT 1 FROM session WHERE id=?", (sid,)).fetchone() is None:
        if _table_exists(conn, "session_v2") and conn.execute("SELECT 1 FROM session_v2 WHERE id=?", (sid,)).fetchone():
            validate_prefix(conn, sid)
        return
    phase = "missing"
    if _table_exists(conn, "kv"):
        row = conn.execute("SELECT value FROM kv WHERE key='migration.v1-v2'").fetchone()
        if row is not None:
            try:
                state = json.loads(row[0])
            except (TypeError, json.JSONDecodeError):
                state = None
            phase = state.get("phase", "malformed") if isinstance(state, dict) else "malformed"
    raise LegacySessionUnsupported(
        f"Retained OpenCode 1 session {sid} is unsupported in this build "
        f"(native migration phase: {phase}). Verified legacy history and cursor "
        "reconciliation is not implemented. Start a fresh pair; retained history "
        "and pair state were left unchanged."
    )


def cursor_sequence(cursor):
    pos = cursor.pending.get("source_pos")
    if pos is None or pos == {}:
        if cursor.line_index == 0 and getattr(cursor, "byte_offset", 0) == 0:
            return -1
    elif isinstance(pos, dict) and set(pos) in ({"seq"}, {"seq", "binding"}):
        seq = pos["seq"]
        if isinstance(seq, int) and not isinstance(seq, bool) and seq >= -1:
            return seq
    raise LegacySessionUnsupported(
        "OpenCode 2 sync cursor is legacy or unrecognized; verified reconciliation "
        "is not implemented. Cursor and pair state were left unchanged."
    )


def _messages(conn, sid, after=-1):
    return [{**json.loads(row["data"]), "id": row["id"], "type": row["type"], "_seq": row["seq"]}
            for row in conn.execute("SELECT * FROM session_message WHERE session_id=? AND seq>? ORDER BY seq", (sid, after))]


def _settled(message):
    if message["type"] == "assistant":
        return message.get("time", {}).get("completed") is not None and all(
            part.get("state", {}).get("status") in {"completed", "error"}
            for part in message.get("content", []) if part.get("type") == "tool")
    if message["type"] in {"shell", "compaction"}:
        return message.get("status") != "running"
    return True


def _busy(conn, sid):
    require_fresh_session(conn, sid)
    proof = validate_prefix(conn, sid)
    row = conn.execute("SELECT time_suspended, time_compacting FROM session_v2 WHERE id=?", (sid,)).fetchone()
    if row is None:
        raise ValueError(f"session {sid} is not in session_v2; retained OpenCode 1 sessions are unsupported in this build")
    if row["time_suspended"] is not None or row["time_compacting"] is not None:
        return True
    if conn.execute("SELECT 1 FROM session_pending WHERE session_id=? LIMIT 1", (sid,)).fetchone():
        return True
    # Agent selection may park a synthetic steer reminder for the next prompt.
    # It does not start execution; claims and admitted pending work above do.
    if conn.execute(
        "SELECT 1 FROM session_inbox WHERE session_id=?"
        " AND NOT (type='synthetic' AND delivery='steer') LIMIT 1", (sid,)
    ).fetchone():
        return True
    last = conn.execute("SELECT type FROM session_message WHERE session_id=? AND type NOT IN ('agent-switched','model-switched','location-switched') ORDER BY seq DESC LIMIT 1", (sid,)).fetchone()
    maximum = conn.execute("SELECT max(seq) FROM session_message WHERE session_id=?", (sid,)).fetchone()[0]
    if proof and maximum <= proof["prefix_end"]:
        return False
    return last is not None and last["type"] != "idle"


class _Opencode2UsageMeter(v1._OpencodeUsageMeter):
    def feed(self, raw):
        if not isinstance(raw, dict):
            return
        super().feed({"assistants": [
            {"message": m} for m in raw.get("messages", [])
            if m.get("type") == "assistant" and not (m.get("metadata") or {}).get("tandem")
        ]})


class Opencode2TurnReader:
    def __init__(self, session_id, db, cursor, session=None):
        self.session_id, self.db, self.cursor = session_id, db, cursor
        self.session = session

    def poll(self):
        from ..tailer import TailedLine

        after = cursor_sequence(self.cursor)
        out, turn = [], []
        with database(self.db) as conn:
            conn.execute("BEGIN")
            require_fresh_session(conn, self.session_id)
            proof = (validate_pair_prefix(conn, self.session) if self.session is not None
                     else validate_prefix(conn, self.session_id))
            remaining = _messages(conn, self.session_id, after)
            offset = 0
            if proof:
                maximum = conn.execute("SELECT max(seq) FROM session_message WHERE session_id=?", (self.session_id,)).fetchone()[0]
                if maximum is None or after > maximum:
                    raise ValueError("OpenCode cursor exceeds native history")
                if after <= proof["prefix_end"] and after not in {0, *(unit["seq"] for unit in proof["units"])}:
                    raise ValueError("OpenCode historical cursor does not match a unit boundary")
                if self.cursor.pending.get("source_pos", {}).get("binding") != proof["binding_id"]:
                    raise ValueError("OpenCode historical cursor is not bound to this reconciliation")
                for unit in proof["units"]:
                    if unit["seq"] <= after:
                        continue
                    start = offset
                    while offset < len(remaining) and remaining[offset]["_seq"] <= unit["seq"]:
                        offset += 1
                    messages = remaining[start:offset]
                    out.append(TailedLine(line_index=self.cursor.line_index + len(out), end_offset=0,
                                          raw={"messages": messages, "_reconciliation_echo": unit["echo"]},
                                          text=f"opencode historical turn {self.cursor.line_index + len(out)}",
                                          pos={"seq": unit["seq"], "binding": proof["binding_id"]}))
                    after = unit["seq"]
            for message in remaining[offset:]:
                turn.append(message)
                if message["type"] != "idle":
                    continue
                if not all(_settled(m) for m in turn):
                    break
                index = self.cursor.line_index + len(out)
                out.append(TailedLine(line_index=index, end_offset=0,
                                      raw={"messages": turn}, text=f"opencode turn {index}",
                                      pos={"seq": message["_seq"], **({"binding": proof["binding_id"]} if proof else {})}))
                turn = []
        return out


class Opencode2Adapter(v1.OpencodeAdapter):
    def runtime_ready(self):
        version = self.detect_version()
        if not version or not self.version_supported(version):
            return False, f"unsupported OpenCode version: {version!r}"
        db = db_path()
        if db is None:
            return False, "OpenCode database not discoverable"
        try:
            with database(db) as conn:
                for table, columns in _REQUIRED.items():
                    actual = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                    if columns - actual:
                        return False, f"OpenCode 2 table {table} missing columns: {sorted(columns - actual)}"
                if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                    return False, "OpenCode database requires WAL mode"
                conn.execute("BEGIN IMMEDIATE")
                conn.rollback()
        except sqlite3.Error as exc:
            return False, f"OpenCode database unavailable: {exc}"
        return True, ""

    def preflight_session(self, session, cursors):
        """Check containment before an operation can seed another participant."""
        for cursor in cursors:
            cursor_sequence(cursor)
        db, sid = db_path(), session.native_id(self.id)
        if sid:
            if db is None:
                raise ValueError("OpenCode database unavailable; cannot verify the paired native session")
            with database(db) as conn:
                require_fresh_session(conn, sid)
                proof = validate_pair_prefix(conn, session)
                for cursor in cursors:
                    pos = cursor.pending.get("source_pos") or {}
                    if proof and pos.get("binding") != proof["binding_id"]:
                        raise ValueError("OpenCode source cursor is not bound to the committed reconciliation")
                    if not proof and "binding" in pos:
                        raise ValueError("OpenCode source cursor proof is missing")

    def transcript_path(self, cwd, session_id):
        db = db_path()
        if db is None or not session_id:
            return None
        with database(db) as conn:
            require_fresh_session(conn, session_id)
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='session_v2'"
            ).fetchone()
            if table is None:
                return None
            row = conn.execute("SELECT 1 FROM session_v2 WHERE id=?", (session_id,)).fetchone()
        return db if row else None

    def validate_transcript(self, path, session_id):
        try:
            with database(path) as conn:
                busy = _busy(conn, session_id)
        except (sqlite3.Error, ValueError) as exc:
            return [str(exc)]
        return ["OpenCode session has an active or unfinished turn"] if busy else []

    def make_usage_meter(self):
        return _Opencode2UsageMeter()

    def make_source_reader(self, session, cursor, transcript):
        cursor_sequence(cursor)
        sid = session.native_id(self.id)
        with database(transcript) as conn:
            require_fresh_session(conn, sid)
            validate_pair_prefix(conn, session)
        return Opencode2TurnReader(sid, transcript, cursor, session)

    def fast_forward_cursor(self, session, cursor):
        cursor_sequence(cursor)
        db, sid = db_path(), session.native_id(self.id)
        if db is None or not sid:
            return
        with database(db) as conn:
            require_fresh_session(conn, sid)
            seq = conn.execute("SELECT max(seq) FROM session_message WHERE session_id=?", (sid,)).fetchone()[0]
            proof = validate_pair_prefix(conn, session)
        cursor.pending["source_pos"] = {"seq": seq if seq is not None else -1,
                                       **({"binding": proof["binding_id"]} if proof else {})}

    def pending_units(self, session, cursor):
        after = cursor_sequence(cursor)
        db, sid = db_path(), session.native_id(self.id)
        if db is None or not sid:
            return 0
        with database(db) as conn:
            require_fresh_session(conn, sid)
            validate_pair_prefix(conn, session)
            return conn.execute("SELECT count(*) FROM session_message WHERE session_id=? AND seq>?", (sid, after)).fetchone()[0]

    def session_status(self, session_id):
        db = db_path()
        if db is None:
            return None
        try:
            with database(db) as conn:
                return "busy" if _busy(conn, session_id) else "waiting"
        except LegacySessionUnsupported:
            raise
        except (sqlite3.Error, ValueError):
            return None

    def parse_entry(self, raw: dict[str, Any], ctx: SessionContext):
        if raw.get("_reconciliation_echo"):
            return [SystemEvent(source=self.id, subtype="tandem_echo")]
        messages = raw.get("messages")
        if not isinstance(messages, list):
            return [SystemEvent(source=self.id, subtype="opencode:unknown")]
        if any((m.get("metadata") or {}).get("tandem") for m in messages):
            return [SystemEvent(source=self.id, subtype="tandem_echo")]
        ctx.turn_index += 1
        events: list[NormalizedEvent] = []
        for message in messages:
            kind = message.get("type")
            args: dict[str, Any] = {"source": self.id, "turn_index": ctx.turn_index}
            if kind == "user":
                text = "\n".join([
                    *(skill["text"] for skill in message.get("skills", []) if skill.get("text")),
                    message.get("text", ""),
                ]).strip()
                events.append(UserMessage(source="user", turn_index=ctx.turn_index, text=text))
            elif kind == "assistant":
                model = message.get("model") or {}
                for part in message.get("content", []):
                    if part.get("type") == "text":
                        events.append(AssistantMessage(**args, text=part.get("text", ""), model=model.get("id")))
                    elif part.get("type") == "reasoning":
                        events.append(Thinking(**args))
                    elif part.get("type") == "tool":
                        state = part.get("state") or {}
                        events.append(ToolCall(**args, call_id=part.get("id", ""), tool=part.get("name", ""), arguments=state.get("input", {})))
                        error = state.get("error") or {}
                        output = "\n".join(item.get("text", "") for item in state.get("content", []) if item.get("type") == "text")
                        events.append(ToolResult(**args, call_id=part.get("id", ""), output=error.get("message", output), is_error=state.get("status") == "error"))
                if message.get("error"):
                    events.append(SystemEvent(**args, subtype="assistant:error", text=message["error"].get("message", "")))
            elif kind in {"synthetic", "skill", "system"}:
                # These are model-facing context, not disposable telemetry.
                text = message.get("text", "")
                if kind == "system":
                    text = f"[OpenCode system context] {text}"
                events.append(UserMessage(**args, text=text))
            elif kind == "shell":
                if (message.get("metadata") or {}).get("background") is True:
                    continue  # Native background completion enters via synthetic input.
                output = (message.get("output") or {}).get("output", "")
                events.append(UserMessage(**args, text=(
                    "The following shell command was executed by the user:\n\n"
                    f"Command:\n{message.get('command', '')}\n\nOutput:\n{output}")))
            elif kind == "location-switched":
                directory = (message.get("location") or {}).get("directory", "")
                events.append(UserMessage(**args, text=f"The working directory has been changed to {directory}."))
            elif kind == "compaction" and message.get("status") == "completed":
                events.append(SystemEvent(**args, subtype="compaction"))
                text = "\n".join([
                    "<conversation-checkpoint>",
                    "Historical context from the earlier conversation:",
                    f"<summary>\n{message.get('summary', '')}\n</summary>",
                    f"<recent-context>\n{message.get('recent', '')}\n</recent-context>",
                    "</conversation-checkpoint>",
                ])
                events.append(UserMessage(**args, text=text))
            elif kind != "idle":
                events.append(SystemEvent(**args, subtype=f"opencode:{kind}", text=message.get("text", "")))
        return events

    def prepare_shadow(self, ref, ctx):
        with database(ref) as conn:
            require_fresh_session(conn, ctx.target_session_id)
        # Whole-turn rows have no parent pointers or mutable renderer chain.
        ctx.state_for(self.id).clear()

    def _entry(self, kind, ctx, **data):
        now = v1._now_ms()
        return {"session_id": ctx.target_session_id, "message": {
            "id": mint_id("msg"), "type": kind, "time": {"created": now},
            "metadata": {"tandem": {"source": ctx.source_id, "session_id": ctx.source_session_id}}, **data}}

    def render_events(self, events, ctx):
        out, assistant = [], None
        calls = {}

        def ensure_assistant():
            nonlocal assistant
            if assistant is None:
                assistant = self._entry("assistant", ctx, agent="build", model={"providerID": "tandem", "id": "<synced>"}, content=[], finish="stop", tokens=_ZERO_TOKENS, cost=0)
                assistant["message"]["time"]["completed"] = v1._now_ms()
                out.append(assistant)
            return assistant["message"]

        def tool_result(call_id, output, is_error):
            if call_id is None and len(calls) == 1:
                call_id = next(iter(calls))
            call = calls.pop(call_id, None)
            if call is None:
                return
            now = v1._now_ms()
            state = {"status": "error" if is_error else "completed", "input": call.arguments if isinstance(call.arguments, dict) else {"input": call.arguments}}
            if is_error:
                state["error"] = {"type": "TandemToolError", "message": output}
            else:
                state["content"] = [{"type": "text", "text": output}]
            ensure_assistant()["content"].append({"type": "tool", "id": call.call_id, "name": call.tool, "state": state, "time": {"created": now, "completed": now}})

        def flush_calls():
            for call_id in list(calls):
                tool_result(call_id, "(tool result not recorded)", True)

        for event in events:
            if event.kind == "user_message":
                flush_calls()
                assistant = None
                out.append(self._entry("user", ctx, text=event.text))
            elif event.kind == "assistant_message":
                message = ensure_assistant()
                message["content"].append({"type": "text", "text": event.text})
                message["metadata"]["tandem"]["source_model"] = event.model
            elif event.kind == "tool_call":
                calls[event.call_id] = event
            elif event.kind == "tool_result":
                tool_result(event.call_id, event.output, event.is_error)
        flush_calls()
        if out:
            if assistant is None:
                ensure_assistant()
            out.append(self._entry("idle", ctx, outcome="succeeded"))
        return out

    def render_placeholder(self, text, ctx):
        return self.render_events([UserMessage(source="tandem", text=text), AssistantMessage(source="tandem", text="[tandem] Turn recorded as a placeholder.")], ctx)

    def shadow_append(self, ref, entries):
        if not entries:
            return
        sid = entries[0]["session_id"]
        if any(e["session_id"] != sid for e in entries) or entries[-1]["message"]["type"] != "idle" or not all(_settled(e["message"]) for e in entries):
            raise ValueError("OpenCode 2 append requires one whole settled session turn")
        conn = connect(ref)
        try:
            conn.execute("BEGIN IMMEDIATE")
            require_fresh_session(conn, sid)
            fresh = []
            for entry in entries:
                row = conn.execute("SELECT session_id, type, data FROM session_message WHERE id=?", (entry["message"]["id"],)).fetchone()
                if row is None:
                    fresh.append(entry)
                else:
                    data = {k: v for k, v in entry["message"].items() if k not in {"id", "type"}}
                    if row["session_id"] != sid or row["type"] != entry["message"]["type"] or json.loads(row["data"]) != data:
                        raise ValueError("OpenCode message ID collision")
            if fresh:
                if len(fresh) != len(entries):
                    raise ValueError("partially landed OpenCode turn")
                if _busy(conn, sid):
                    raise ShadowBusy("OpenCode session has active or pending execution")
                row = conn.execute("SELECT seq FROM event_sequence WHERE aggregate_id=?", (sid,)).fetchone()
                seq = max(row[0] if row else 0, conn.execute("SELECT coalesce(max(seq),0) FROM session_message WHERE session_id=?", (sid,)).fetchone()[0])
                for entry in fresh:
                    seq += 1
                    message = entry["message"]
                    data = {k: v for k, v in message.items() if k not in {"id", "type"}}
                    now = message["time"]["created"]
                    conn.execute("INSERT INTO session_message(id,session_id,type,seq,time_created,time_updated,data) VALUES (?,?,?,?,?,?,?)", (message["id"], sid, message["type"], seq, now, now, json.dumps(data)))
                conn.execute("INSERT INTO event_sequence(aggregate_id,seq) VALUES (?,?) ON CONFLICT(aggregate_id) DO UPDATE SET seq=max(seq,excluded.seq)", (sid, seq))
                now = v1._now_ms()
                conn.execute("UPDATE session_v2 SET time_updated=?,time_idle=?,idle_outcome='succeeded' WHERE id=?", (now, now, sid))
            conn.commit()
        except sqlite3.OperationalError as exc:
            conn.rollback()
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                raise ShadowBusy(str(exc)) from exc
            raise
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def shadow_intent(self, ref, entries):
        with database(ref) as conn:
            for sid in {e["session_id"] for e in entries}:
                require_fresh_session(conn, sid)
        return {"ids": [e["message"]["id"] for e in entries]}

    def intent_landed(self, ref, intent):
        ids = intent.get("ids") or []
        if not ids:
            return False
        with database(ref) as conn:
            return all(conn.execute("SELECT 1 FROM session_message WHERE id=?", (id,)).fetchone() for id in ids)

    def interactive_argv(self, session_id, fresh):
        assert session_id, "OpenCode sessions are created at pair time"
        return [self.binary, "--standalone", "-s", session_id]

    def oneoff_argv(self, session_id, prompt):
        return [self.binary, "run", "--standalone", "-s", session_id, prompt]

    def create_shadow_transcript(self, cwd, session_id, ctx, note):
        db = db_path()
        if db is not None:
            with database(db) as conn:
                require_fresh_session(conn, session_id)
        now = v1._now_ms()
        entries = self.render_events([UserMessage(source="tandem", text=note), AssistantMessage(source="tandem", text="[tandem] Session created; context syncs from the paired session.")], ctx)
        payload = {"info": {"id": session_id, "projectID": "tandem-import", "agent": "build", "location": {"directory": cwd}, "title": "tandem paired session", "time": {"created": now, "updated": now, "idle": now}, "outcome": "succeeded", "cost": 0, "tokens": _ZERO_TOKENS}, "messages": [e["message"] for e in entries]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="tandem-oc2-seed-", delete=False) as file:
            json.dump(payload, file)
            seed = Path(file.name)
        try:
            result = _run([self.binary, "session", "import", "--standalone", str(seed)], cwd=cwd, capture_output=True, text=True, timeout=120)
            if result.returncode or "Session already exists" in result.stderr:
                raise RuntimeError(f"OpenCode session import failed: {result.stderr[-300:]}")
        finally:
            seed.unlink(missing_ok=True)
        db = self.transcript_path(cwd, session_id)
        if db is None:
            raise RuntimeError(f"OpenCode import did not create session {session_id}")
        if not self.intent_landed(db, self.shadow_intent(db, entries)):
            raise RuntimeError("OpenCode import did not persist seed messages")
        return db
