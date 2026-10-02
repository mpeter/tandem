"""Explicit retained OpenCode migration through a private native snapshot.

The native import and the pair commit are separate durable operations. A failed
commit leaves the exact prepared identity available for a verified retry.
"""

from __future__ import annotations

import copy
import fcntl
import os
import json
import shutil
import sqlite3
import subprocess
import tempfile
import uuid
from dataclasses import asdict
from contextlib import contextmanager
from pathlib import Path

from .harness import opencode as legacy
from .harness.opencode2 import database
from .opencode_binding import (
    MARKER_KEY,
    PROOF_KEY,
    digest,
    prefix_rows,
    validate_pair_prefix,
)
from .opencode_native_snapshot import (
    NATIVE_VERSION,
    capture_source,
    import_payload,
    imported_info,
    native_oracle,
    normalized_import_info,
)
from .state import StateStore, SyncCursor


_LEASES: set[int] = set()


def _close_forked_leases() -> None:
    for fd in tuple(_LEASES):
        try:
            os.close(fd)
        except OSError:
            pass
    _LEASES.clear()


os.register_at_fork(after_in_child=_close_forked_leases)


def _exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def _source(conn: sqlite3.Connection, sid: str) -> dict:
    tables = ("session", "message", "part")
    snapshot: dict = {
        table: [
            dict(row)
            for row in conn.execute(
                f"SELECT * FROM {table} WHERE {'id' if table == 'session' else 'session_id'}=? ORDER BY id",
                (sid,),
            )
        ]
        for table in tables
    }
    if len(snapshot["session"]) != 1:
        raise ValueError("Selected identity has no retained legacy session")
    snapshot["ddl"] = {
        table: conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()[0]
        for table in tables
    }
    snapshot["project"] = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM project WHERE id=?", (snapshot["session"][0]["project_id"],)
        )
    ]
    return snapshot


def _native_state(conn: sqlite3.Connection, sid: str) -> dict:
    """Snapshot old executable/context state; projection rows are checked separately."""
    state: dict = {}
    if _exists(conn, "session_v2"):
        row = conn.execute("SELECT * FROM session_v2 WHERE id=?", (sid,)).fetchone()
        if row:
            info = dict(row)
            if any(
                info.get(key) is not None
                for key in (
                    "parent_id",
                    "revert",
                    "fork_session_id",
                    "time_suspended",
                    "time_compacting",
                )
            ):
                raise ValueError(
                    "Retained session has parent, revert, fork or active native state"
                )
            state["info"] = info
    for table in (
        "session_pending",
        "session_inbox",
        "instruction_entry",
        "instruction_state",
    ):
        if _exists(conn, table):
            rows = [
                dict(row)
                for row in conn.execute(
                    f"SELECT * FROM {table} WHERE session_id=?", (sid,)
                )
            ]
            if rows:
                raise ValueError(
                    "Retained session has pending work or extra instruction state"
                )
    state["projection"] = (
        [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM session_message WHERE session_id=? ORDER BY seq", (sid,)
            )
        ]
        if _exists(conn, "session_message")
        else []
    )
    return state


def _source_digest(snapshot: dict) -> str:
    # Project access timestamps are rewritten by supported import/location
    # resolution; frozen legacy session/message/part rows remain the authority.
    source = {key: snapshot[key] for key in ("session", "message", "part", "ddl")}
    source["project"] = [
        {
            key: value
            for key, value in row.items()
            if key not in {"time_updated", "time_active"}
        }
        for row in snapshot["project"]
    ]
    return digest(source)


def _projection(rows: list[dict]) -> list[dict]:
    fields = ("id", "type", "seq", "time_created", "data")
    return [
        {key: json.loads(row[key]) if key == "data" else row[key] for key in fields}
        for row in rows
    ]


def _verify_old_native(conn: sqlite3.Connection, record: dict) -> None:
    current = _native_state(conn, record["old_id"])
    original = record["original_native"]
    if original.get("info") is not None:
        if digest(current) != digest(original):
            raise ValueError(
                "Retained native projection changed; prepared import preserved unbound"
            )
    else:
        if current.get("info") is not None and digest(
            imported_info(conn, record["old_id"])
        ) != digest(record["oracle_info"]):
            raise ValueError(
                "Newly migrated retained session Info differs from native oracle; preserved unbound"
            )
        if current["projection"] and digest(
            _projection(current["projection"])
        ) != digest(record["oracle_rows"]):
            raise ValueError(
                "Retained native projection has changed content or a suffix; prepared import preserved unbound"
            )


def _check_source(snapshot: dict) -> None:
    session = snapshot["session"][0]
    if any(
        session.get(key) is not None
        for key in ("parent_id", "revert", "time_compacting", "workspace_id")
    ):
        raise ValueError(
            "Initial migration supports closed root sessions without revert"
        )
    messages = {row["id"]: json.loads(row["data"]) for row in snapshot["message"]}
    if not messages:
        raise ValueError("Empty retained history requires a fresh pair")
    for message in messages.values():
        if message.get("role") not in {"user", "assistant"}:
            raise ValueError("Unsupported legacy message shape")
        if message["role"] == "assistant" and not legacy._is_closed(message):
            raise ValueError(
                "Every retained assistant must be completed with a terminal finish"
            )
    for row in snapshot["part"]:
        part = json.loads(row["data"])
        if row["message_id"] not in messages:
            raise ValueError("Orphan legacy part is unsupported")
        if part.get("type") == "text" and part.get("ignored") and part.get("text"):
            raise ValueError(
                "Unsupported ignored legacy text requires explicit context-loss recovery"
            )
        if part.get("type") in {"file", "agent", "subtask"}:
            raise ValueError(
                "Attachments, agent mentions and subtask wrappers require additional recovery"
            )
        state = part.get("state", {})
        cleared = (
            state.get("status") == "completed"
            and state.get("time", {}).get("compacted") is not None
        )
        if part.get("type") == "tool" and state.get("attachments") and not cleared:
            raise ValueError(
                "Noncompacted tool attachments require additional peer-context recovery"
            )
        if part.get("type") == "tool" and part.get("state", {}).get("status") not in {
            "completed",
            "error",
        }:
            raise ValueError("Unfinished retained tool is unsupported")


def _cursors(store: StateStore, session) -> list[dict]:
    directions = {
        (source, target)
        for source in session.participants
        for target in session.participants
        if source != target and "opencode" in (source, target)
    }
    directions.update(
        (row[0], row[1])
        for row in store._conn.execute(
            "SELECT source,target FROM sync_cursors WHERE tandem_id=? AND (source='opencode' OR target='opencode')",
            (session.tandem_id,),
        )
    )
    cursors = [
        asdict(store.get_cursor(session.tandem_id, source, target))
        for source, target in sorted(directions)
    ]
    for cursor in cursors:
        if "intent" in cursor["pending"]:
            raise ValueError(
                "Unresolved sync intent requires recovery before retained migration"
            )
        if cursor["source"] == "opencode":
            pos = cursor["pending"].get("source_pos")
            if not pos and (cursor["line_index"] or cursor["byte_offset"]):
                raise ValueError("Nonzero legacy cursor has no source position")
            if pos and (
                not isinstance(pos, dict)
                or set(pos) != {"time", "id"}
                or type(pos["time"]) is not int
                or not isinstance(pos["id"], str)
            ):
                raise ValueError("Unrecognized legacy OpenCode cursor")
    return cursors


def _cut(source_order: list[dict], origins: list[set[str]], anchor: dict | None) -> int:
    if not anchor:
        return 0
    matches = [
        row
        for row in source_order
        if row["id"] == anchor["id"] and row["time_created"] == anchor["time"]
    ]
    if len(matches) != 1:
        raise ValueError("Legacy cursor anchor is missing or changed")
    consumed = {
        row["id"]
        for row in source_order
        if (row["time_created"], row["id"]) <= (anchor["time"], anchor["id"])
    }
    flags = []
    for owned in origins:
        if owned & consumed and not owned <= consumed:
            raise ValueError("Legacy cut splits a native folded message")
        flags.append(owned <= consumed)
    count = sum(flags)
    if flags != [True] * count + [False] * (len(flags) - count):
        raise ValueError("Native ordering crosses the legacy cursor cut")
    return count


def _mapping(
    snapshot: dict, oracle: list[dict], old_units: list
) -> tuple[list[dict], list[set[str]]]:
    source = {row["id"]: json.loads(row["data"]) for row in snapshot["message"]}
    origins: list[set[str]] = []
    for index, row in enumerate(oracle):
        if row["seq"] != index:
            raise ValueError("Native converted prefix has gaps")
        if row["id"] in source:
            owned = {row["id"]}
            if row["type"] == "compaction":
                summaries = [
                    sid
                    for sid, message in source.items()
                    if message.get("parentID") == row["id"]
                    and message.get("summary") is True
                ]
                if (
                    len(summaries) != 1
                    or json.loads(row["data"]).get("status") != "completed"
                ):
                    raise ValueError("Compaction has no unique completed summary")
                owned.update(summaries)
        elif origins and row["type"] in {"synthetic", "system"}:
            # These rows were constructed by the pinned native oracle, never by
            # a Python ID converter. Native companion/notice owns its predecessor.
            owned = origins[-1].copy()
        else:
            raise ValueError("Unclassified native output origin")
        origins.append(owned)
    if not origins or set().union(*origins) != set(source):
        raise ValueError("Native conversion dropped unsupported source history")
    order = sorted(
        snapshot["message"], key=lambda row: (row["time_created"], row["id"])
    )
    units, previous = [], 0
    consumed_messages: set[str] = set()
    for index, unit in enumerate(old_units):
        end = _cut(order, origins, unit.pos)
        if end <= previous:
            raise ValueError("Historical turn has no distinct native boundary")
        raw = unit.raw
        messages = [
            raw["user"]["message"],
            *(item["message"] for item in raw["assistants"]),
        ]
        consumed_messages.update(message["id"] for message in messages)
        echo = (
            raw["user"]["message"].get("model", {}).get("providerID")
            == legacy.SENTINEL_PROVIDER
        )
        units.append({"index": index, "seq": end, "echo": echo})
        previous = end
    if previous != len(oracle) or consumed_messages != set(source):
        raise ValueError("Old reader cannot certify the complete retained history")
    return units, origins


def _expected_rows(payload: dict) -> list[dict]:
    return [
        {
            "id": message["id"],
            "type": message["type"],
            "seq": index + 1,
            "time_created": message["time"]["created"],
            "data": {
                key: value
                for key, value in message.items()
                if key not in {"id", "type"}
            },
        }
        for index, message in enumerate(payload["messages"])
    ]


def _verify_target(conn: sqlite3.Connection, record: dict) -> bool:
    sid = record["fresh_id"]
    row = conn.execute("SELECT metadata FROM session_v2 WHERE id=?", (sid,)).fetchone()
    if row is None:
        return False
    if digest(imported_info(conn, sid)) != digest(record["expected_info"]):
        raise ValueError(
            "Prepared native identity conflicts with supported session Info; preserved unbound"
        )
    if digest(json.loads(row["metadata"]) or {}) != digest(
        record["payload"]["info"].get("metadata") or {}
    ):
        raise ValueError(
            "Prepared native identity conflicts with existing session metadata; preserved unbound"
        )
    rows = prefix_rows(conn, sid, record["proof"]["prefix_end"])
    maximum = conn.execute(
        "SELECT max(seq) FROM session_message WHERE session_id=?", (sid,)
    ).fetchone()[0]
    if (
        digest(rows) != digest(record["expected_rows"])
        or maximum != record["proof"]["prefix_end"]
    ):
        raise ValueError(
            "Prepared native identity conflicts with existing content; preserved unbound"
        )
    if conn.execute("SELECT 1 FROM session WHERE id=?", (sid,)).fetchone():
        raise ValueError("Prepared identity unexpectedly belongs to the legacy table")
    _native_state(conn, sid)
    watermark = conn.execute(
        "SELECT seq FROM event_sequence WHERE aggregate_id=?", (sid,)
    ).fetchone()
    if watermark is None or type(watermark[0]) is not int or watermark[0] < maximum:
        raise ValueError(
            "Native import sequence reservation is missing or behind history"
        )
    return True


@contextmanager
def migration_lease(store: StateStore, tandem_id: str):
    directory = store.db_path.parent / "opencode-migration-locks"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(
        directory / (digest(tandem_id) + ".lock"),
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    _LEASES.add(fd)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "This pair has an active retained migration; wait for its native helper cleanup"
            ) from exc
        yield fd
    finally:
        _LEASES.discard(fd)
        os.close(fd)


def migrate_opencode(
    store: StateStore,
    tandem_id: str,
    db: Path,
    *,
    dry_run: bool = False,
    binary: str = "opencode",
    scratch: Path | None = None,
) -> dict:
    if dry_run:
        return _migrate_opencode(
            store, tandem_id, db, dry_run=True, binary=binary, scratch=scratch
        )
    with migration_lease(store, tandem_id) as lease_fd:
        return _migrate_opencode(
            store, tandem_id, db, binary=binary, scratch=scratch, lease_fd=lease_fd
        )


def _migrate_opencode(
    store: StateStore,
    tandem_id: str,
    db: Path,
    *,
    dry_run: bool = False,
    binary: str = "opencode",
    scratch: Path | None = None,
    lease_fd: int = -1,
) -> dict:
    """Prepare, import, verify and atomically rebind one explicit paired session."""
    db = db.resolve()
    session = store.get_session(tandem_id)
    if (
        session is None
        or "opencode" not in session.participants
        or not session.native_id("opencode")
    ):
        raise ValueError("Select a pair with a retained OpenCode identity")
    record = store.get_opencode_reconciliation(tandem_id)
    if record and Path(record["native_db"]) != db:
        raise ValueError("Prepared migration belongs to a different native database")
    if record and record["phase"] == "committed":
        with database(db) as conn:
            validate_pair_prefix(conn, session)
        return {
            "phase": "committed",
            "old_id": record["old_id"],
            "fresh_id": record["fresh_id"],
        }
    sid = record["old_id"] if record else session.native_id("opencode")
    if not isinstance(sid, str):
        raise ValueError("Missing retained identity")
    snapshot = capture_source(db, sid)
    _check_source(snapshot)
    with database(db) as conn:
        native = _native_state(conn, sid)
    with store._lock, store._conn:
        store._conn.execute("BEGIN")
        cursors = _cursors(store, session)
        stored_directions = [
            list(row)
            for row in store._conn.execute(
                "SELECT source,target FROM sync_cursors WHERE tandem_id=? "
                "AND (source='opencode' OR target='opencode') ORDER BY source,target",
                (tandem_id,),
            )
        ]
    units = legacy.OpencodeTurnReader(
        sid, db, SyncCursor(tandem_id, "opencode", "migration")
    ).poll()
    if not units:
        raise ValueError("Retained history is not closed under the old reader")
    anchors = [unit.pos for unit in units]
    for cursor in cursors:
        if cursor["source"] == "opencode":
            pos = cursor["pending"].get("source_pos")
            if pos and pos not in anchors:
                raise ValueError(
                    "Legacy cursor is not at a verified old reader boundary"
                )
    if dry_run:
        return {
            "phase": "prepared" if record else "preflight",
            "native_verified": False,
            "note": "Source preflight only; native conversion, filtering and cursor cuts are checked during migration",
            "old_id": sid,
            "fresh_id": record["fresh_id"] if record else None,
            "historical_turns": len(units),
        }
    executable = shutil.which(binary)
    if executable is None:
        raise ValueError("OpenCode binary is unavailable")
    version = subprocess.run(
        [executable, "--version"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if version.returncode or version.stdout.strip() not in {
        NATIVE_VERSION,
        f"opencode v{NATIVE_VERSION}",
    }:
        raise ValueError(
            f"Retained migration requires native OpenCode {NATIVE_VERSION}"
        )
    if scratch is None:
        scratch = store.db_path.parent / "opencode-migration-scratch"
    if scratch:
        scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(
        prefix="tandem-retained-", dir=scratch
    ) as directory:
        root = Path(directory)
        if record is None:
            payload, oracle, old_info = native_oracle(
                executable, snapshot, sid, root, lease_fd
            )
            historical, origins = _mapping(snapshot, oracle, units)
            if native["projection"]:
                if digest(_projection(native["projection"])) != digest(
                    _projection(oracle)
                ):
                    raise ValueError(
                        "Old native projection has changed content or a suffix; recovery required"
                    )
            payload = copy.deepcopy(payload)
            binding = uuid.uuid4().hex
            payload["messages"][0].setdefault("metadata", {})[MARKER_KEY] = binding
            expected = _expected_rows(payload)
            proof = {
                "version": 1,
                "native_version": NATIVE_VERSION,
                "binding_id": binding,
                "prefix_end": len(expected),
                "prefix_digest": digest(expected),
                "units": historical,
                "source_digest": _source_digest(snapshot),
            }
            payload["info"].setdefault("metadata", {})[PROOF_KEY] = proof
            translated = copy.deepcopy(cursors)
            order = sorted(
                snapshot["message"], key=lambda row: (row["time_created"], row["id"])
            )
            checkpoint = max(
                (
                    index + 1
                    for index, row in enumerate(oracle)
                    if row["type"] == "compaction"
                    and json.loads(row["data"]).get("status") == "completed"
                ),
                default=0,
            )
            for cursor in translated:
                if cursor["source"] == "opencode":
                    cut = _cut(order, origins, cursor["pending"].get("source_pos"))
                    if checkpoint and cut < checkpoint - 1:
                        raise ValueError(
                            "Outgoing cursor has pending archival context before the latest checkpoint; explicit peer-context recovery required"
                        )
                    cursor["pending"]["source_pos"] = {
                        "seq": _cut(
                            order, origins, cursor["pending"].get("source_pos")
                        ),
                        "binding": binding,
                    }
            record = {
                "old_id": sid,
                "fresh_id": payload["info"]["id"],
                "native_db": str(db),
                "source_digest": _source_digest(snapshot),
                "original_native": native,
                "oracle_rows": _projection(oracle),
                "oracle_info": old_info,
                "original_ids": session.native_session_ids,
                "original_cursors": cursors,
                "original_pair": {
                    "participants": session.participants,
                    "cwd": session.cwd,
                    "active": session.active,
                },
                "cursor_directions": stored_directions,
                "translated_cursors": translated,
                "payload": payload,
                "expected_rows": expected,
                "expected_info": normalized_import_info(
                    executable, payload, root, lease_fd
                ),
                "proof": proof,
                "origins": [sorted(owned) for owned in origins],
            }
            store.prepare_opencode_reconciliation(tandem_id, record)
        with database(db) as conn:
            present = _verify_target(conn, record)
        if not present:
            import_payload(executable, db, record["payload"], root, lease_fd)
        # Native lock is acquired only after native oracle/import have stopped.
        # It freezes retained writers through the StateStore CAS, in this order.
        with database(db) as conn:
            conn.execute("BEGIN IMMEDIATE")
            if _source_digest(_source(conn, sid)) != record["source_digest"]:
                raise ValueError(
                    "Retained source changed; prepared native identity preserved unbound"
                )
            _verify_old_native(conn, record)
            if not _verify_target(conn, record):
                raise ValueError("Native import did not persist the prepared identity")
            store.commit_opencode_reconciliation(tandem_id)
    return {"phase": "committed", "old_id": sid, "fresh_id": record["fresh_id"]}
