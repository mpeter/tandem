"""Projection contracts from OpenCode tag v2.0.21 session/sql.ts."""
import json
from contextlib import closing
from types import SimpleNamespace

import pytest

from tandem.events import AssistantMessage, SessionContext, ToolCall, ToolResult, UserMessage
from tandem.harness import opencode2 as oc
from tandem.harness.base import ShadowBusy


@pytest.fixture
def store(tmp_path, monkeypatch):
    db = tmp_path / "opencode.db"
    with oc.database(db) as conn:
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE session_v2 (
                id TEXT PRIMARY KEY, model TEXT, time_suspended INTEGER,
                time_compacting INTEGER, time_idle INTEGER,
                time_updated INTEGER NOT NULL, idle_outcome TEXT
            );
            CREATE TABLE session_message (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES session_v2(id),
                type TEXT NOT NULL, seq INTEGER NOT NULL,
                time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
                data TEXT NOT NULL, UNIQUE(session_id,seq)
            );
            CREATE TABLE session_pending (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES session_v2(id),
                type TEXT NOT NULL, data TEXT NOT NULL, delivery TEXT,
                admitted_seq INTEGER NOT NULL, time_created INTEGER NOT NULL
            );
            CREATE TABLE session_inbox (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES session_v2(id),
                type TEXT NOT NULL, payload TEXT NOT NULL, delivery TEXT NOT NULL,
                enqueued_seq INTEGER NOT NULL, time_created INTEGER NOT NULL
            );
            CREATE TABLE event_sequence (aggregate_id TEXT PRIMARY KEY, seq INTEGER NOT NULL, owner_id TEXT);
            INSERT INTO session_v2(id,time_updated) VALUES ('ses_test',1);
            INSERT INTO event_sequence VALUES ('ses_test',20,NULL);
        """)
    monkeypatch.setattr(oc, "db_path", lambda: db)
    return db


@pytest.fixture
def ctx():
    return SessionContext(tandem_id="pair", cwd="/tmp", direction="codex->opencode", target_session_id="ses_test")


def turn(adapter, ctx):
    return adapter.render_events([UserMessage(source="user", text="hi"), AssistantMessage(source="codex", text="hello", model="source-model")], ctx)


def test_append_reserves_sequence_and_idempotent(store, ctx):
    adapter = oc.Opencode2Adapter()
    entries = turn(adapter, ctx)
    intent = adapter.shadow_intent(store, entries)
    assert not adapter.intent_landed(store, intent)
    adapter.shadow_append(store, entries)
    adapter.shadow_append(store, entries)
    assert adapter.intent_landed(store, intent)
    with oc.database(store) as conn:
        rows = conn.execute("SELECT seq,type,data FROM session_message ORDER BY seq").fetchall()
        assert [r["seq"] for r in rows] == [21, 22, 23]
        assert conn.execute("SELECT seq FROM event_sequence").fetchone()[0] == 23
        assistant = json.loads(rows[1]["data"])
        assert "id" not in assistant and "type" not in assistant
        assert assistant["model"] == {"providerID": "tandem", "id": "<synced>"}
        assert assistant["metadata"]["tandem"]["source_model"] == "source-model"
    assert adapter.session_status("ses_test") == "waiting"


@pytest.mark.parametrize("busy", ["claim", "compacting", "pending", "inbox", "user"])
def test_active_and_pending_append_retry(store, ctx, busy):
    adapter = oc.Opencode2Adapter()
    entries = turn(adapter, ctx)
    with oc.database(store) as conn:
        if busy == "claim":
            conn.execute("UPDATE session_v2 SET time_suspended=1")
        elif busy == "compacting":
            conn.execute("UPDATE session_v2 SET time_compacting=1")
        elif busy == "pending":
            conn.execute("INSERT INTO session_pending VALUES ('msg_p','ses_test','user','{}',NULL,1,1)")
        elif busy == "inbox":
            conn.execute("INSERT INTO session_inbox VALUES ('msg_p','ses_test','user','{}','next',1,1)")
        else:
            conn.execute("INSERT INTO session_message VALUES ('msg_p','ses_test','user',1,1,1,'{}')")
    assert adapter.session_status("ses_test") == "busy"
    with pytest.raises(ShadowBusy):
        adapter.shadow_append(store, entries)
    assert not adapter.intent_landed(store, adapter.shadow_intent(store, entries))
    with oc.database(store) as conn:
        assert conn.execute("SELECT seq FROM event_sequence").fetchone()[0] == 20
        conn.execute("UPDATE session_v2 SET time_suspended=NULL,time_compacting=NULL")
        for table in ("session_pending", "session_inbox", "session_message"):
            conn.execute(f"DELETE FROM {table}")
    adapter.shadow_append(store, entries)
    assert adapter.intent_landed(store, adapter.shadow_intent(store, entries))


def test_reader_whole_idle_turn_cursor_and_echo(store, ctx):
    adapter = oc.Opencode2Adapter()
    entries = turn(adapter, ctx)
    adapter.shadow_append(store, entries)
    cursor = SimpleNamespace(pending={"source_pos": {"seq": 20}}, line_index=7)
    reader = oc.Opencode2TurnReader("ses_test", store, cursor)
    units = reader.poll()
    assert len(units) == 1
    assert units[0].line_index == 7
    assert units[0].pos == {"seq": 23}
    assert adapter.parse_entry(units[0].raw, ctx)[0].subtype == "tandem_echo"
    cursor.pending["source_pos"] = units[0].pos
    assert reader.poll() == []
    with oc.database(store) as conn:
        conn.execute("INSERT INTO session_message VALUES ('msg_u','ses_test','user',24,1,1,?)", (json.dumps({"text": "steer", "time": {"created": 1}}),))
    assert reader.poll() == []


def test_reader_does_not_advance_incomplete_tools_even_with_idle(store, ctx):
    adapter = oc.Opencode2Adapter()
    entries = turn(adapter, ctx)
    entries[1]["message"]["content"] = [{"type": "tool", "state": {"status": "running"}}]
    with pytest.raises(ValueError, match="settled"):
        adapter.shadow_append(store, entries)
    assert not adapter.intent_landed(store, adapter.shadow_intent(store, entries))


def test_tool_text_reasoning_and_error_conversion(store, ctx):
    raw = {"messages": [{"type": "user", "text": "run"}, {"type": "assistant", "model": {"providerID": "openai", "id": "native"}, "error": {"type": "ProviderError", "message": "oops"}, "content": [
        {"type": "reasoning", "text": "private"}, {"type": "text", "text": "done"},
        {"type": "tool", "id": "call", "name": "bash", "state": {"status": "completed", "input": {"command": "pwd"}, "content": [{"type": "text", "text": "/tmp"}]}},
        {"type": "tool", "id": "fail", "name": "read", "state": {"status": "error", "input": {}, "error": {"type": "NotFound", "message": "missing"}}},
    ]}, {"type": "idle", "outcome": "failed"}]}
    adapter = oc.Opencode2Adapter()
    events = adapter.parse_entry(raw, ctx)
    assert [e.kind for e in events] == ["user_message", "thinking", "assistant_message", "tool_call", "tool_result", "tool_call", "tool_result", "system"]
    assert events[2].model == "native"
    assert events[4].output == "/tmp"
    assert events[6].is_error and events[6].output == "missing"
    entries = adapter.render_events(events, ctx)
    tool = entries[1]["message"]["content"][1]
    assert tool["name"] == "bash" and tool["state"]["content"][0]["text"] == "/tmp"
    assert entries[1]["message"]["content"][2]["state"]["error"]["message"] == "missing"


def test_busy_database_rolls_back_and_retries(store, ctx, monkeypatch):
    original = oc.connect
    def short_connect(db):
        conn = original(db)
        conn.execute("PRAGMA busy_timeout=1")
        return conn
    monkeypatch.setattr(oc, "connect", short_connect)
    adapter = oc.Opencode2Adapter()
    entries = turn(adapter, ctx)
    with closing(original(store)) as locked, locked:
        locked.execute("BEGIN IMMEDIATE")
        with pytest.raises(ShadowBusy):
            adapter.shadow_append(store, entries)
    adapter.shadow_append(store, entries)
    assert adapter.intent_landed(store, adapter.shadow_intent(store, entries))


def test_native_launch_is_standalone():
    adapter = oc.Opencode2Adapter()
    assert "--standalone" in adapter.interactive_argv("ses_test", False)
    assert "--standalone" in adapter.oneoff_argv("ses_test", "hello")


def test_usage_ignores_echoes():
    meter = oc.Opencode2Adapter().make_usage_meter()
    tokens = {"input": 100, "output": 20, "reasoning": 5, "cache": {"read": 10, "write": 3}}
    meter.feed({"messages": [{"type": "assistant", "tokens": tokens}, {"type": "assistant", "tokens": tokens, "metadata": {"tandem": {"source": "codex"}}}]})
    snapshot = meter.snapshot()
    assert (snapshot.input_tokens, snapshot.output_tokens, snapshot.ctx_tokens) == (113, 25, 110)


def test_birth_uses_native_import_and_verifies_seed(store, ctx, monkeypatch):
    adapter = oc.Opencode2Adapter()
    captured = {}
    def native_import(argv, **kwargs):
        captured["argv"] = argv
        captured["cwd"] = kwargs["cwd"]
        payload = json.loads(oc.Path(argv[-1]).read_text())
        captured["payload"] = payload
        adapter.shadow_append(store, [{"session_id": payload["info"]["id"], "message": m} for m in payload["messages"]])
        return SimpleNamespace(returncode=0, stdout="Imported session", stderr="")
    monkeypatch.setattr(oc, "_run", native_import)
    assert adapter.create_shadow_transcript("/tmp", "ses_test", ctx, "seed") == store
    assert captured["argv"][:4] == ["opencode", "session", "import", "--standalone"]
    assert captured["payload"]["info"]["location"] == {"directory": "/tmp"}
    assert captured["payload"]["info"]["agent"] == "build"
    assert "model" not in captured["payload"]["info"]
    assert [m["type"] for m in captured["payload"]["messages"]] == ["user", "assistant", "idle"]
    assert not oc.Path(captured["argv"][-1]).exists()


def test_birth_success_without_seed_is_failure(store, ctx, monkeypatch):
    monkeypatch.setattr(oc, "_run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""))
    with pytest.raises(RuntimeError, match="persist seed"):
        oc.Opencode2Adapter().create_shadow_transcript("/tmp", "ses_test", ctx, "seed")


def test_runtime_checks_columns(store, monkeypatch):
    adapter = oc.Opencode2Adapter()
    monkeypatch.setattr(adapter, "detect_version", lambda: "2.0.21")
    monkeypatch.setattr(adapter, "version_supported", lambda version: True)
    assert adapter.runtime_ready() == (True, "")
    with oc.database(store) as conn:
        conn.execute("ALTER TABLE session_v2 RENAME COLUMN time_suspended TO old_claim")
    assert "time_suspended" in adapter.runtime_ready()[1]


def test_parallel_tool_results_are_matched_by_call_id(ctx):
    entries = oc.Opencode2Adapter().render_events([
        UserMessage(source="user", text="parallel"),
        ToolCall(source="codex", call_id="a", tool="read", arguments={"path": "a"}),
        ToolCall(source="codex", call_id="b", tool="read", arguments={"path": "b"}),
        ToolResult(source="codex", call_id="b", output="B"),
        ToolResult(source="codex", call_id="a", output="A"),
    ], ctx)
    content = entries[1]["message"]["content"]
    assert {p["id"]: p["state"]["content"][0]["text"] for p in content} == {"a": "A", "b": "B"}
    assert all(p["state"]["status"] == "completed" for p in content)


@pytest.mark.parametrize("kind", ["agent-switched", "model-switched", "location-switched"])
def test_control_switch_after_idle_does_not_block_handoff(store, ctx, kind):
    adapter = oc.Opencode2Adapter()
    adapter.shadow_append(store, turn(adapter, ctx))
    with oc.database(store) as conn:
        conn.execute("INSERT INTO session_message VALUES ('msg_switch','ses_test',?,24,1,1,'{}')", (kind,))
    assert adapter.session_status("ses_test") == "waiting"
    adapter.shadow_append(store, turn(adapter, ctx))
    with oc.database(store) as conn:
        assert conn.execute("SELECT max(seq) FROM session_message").fetchone()[0] == 27


def test_native_context_survives_converter_policy():
    from tandem.converter import ReferenceConverter

    ctx = SessionContext(tandem_id="pair", cwd="/tmp", direction="opencode->codex")
    messages = [
        {"type": "shell", "shellID": "sh_test", "command": "pwd", "status": "exited", "exit": 0,
         "output": {"output": "/tmp", "cursor": 4, "size": 4, "truncated": False}, "time": {"created": 1, "completed": 2}},
        {"type": "synthetic", "text": "background completed", "time": {"created": 3}},
        {"type": "skill", "skill": "sk_test", "name": "test", "text": "skill context", "time": {"created": 4}},
        {"type": "system", "text": "native system update", "time": {"created": 5}},
        {"type": "user", "text": "prompt", "skills": [{"name": "inline", "text": "inline skill context"}]},
        {"type": "location-switched", "location": {"directory": "/new"}},
        {"type": "compaction", "status": "completed", "reason": "manual", "summary": "earlier facts", "recent": "recent actions", "time": {"created": 6}},
        {"type": "idle", "outcome": "succeeded", "time": {"created": 7}},
    ]
    events = oc.Opencode2Adapter().parse_entry({"messages": messages}, ctx)
    translated = ReferenceConverter()._apply_policy(events, "opencode", ctx)
    text = "\n".join(e.text for e in translated)
    assert "Command:\npwd\n\nOutput:\n/tmp" in text
    for expected in ("background completed", "skill context", "native system update", "inline skill context", "prompt", "/new", "earlier facts", "recent actions", "compacted"):
        assert expected in text
    assert any(e.kind == "system" and e.subtype == "compaction" for e in events)


def test_background_shell_and_failed_compaction_remain_telemetry():
    ctx = SessionContext(tandem_id="pair", cwd="/tmp", direction="opencode->codex")
    events = oc.Opencode2Adapter().parse_entry({"messages": [
        {"type": "shell", "command": "pwd", "status": "exited", "metadata": {"background": True}},
        {"type": "compaction", "status": "failed", "error": {"type": "Failure", "message": "no summary"}},
        {"type": "idle", "outcome": "failed"},
    ]}, ctx)
    assert [e.kind for e in events] == ["system"]


def test_transcript_discovery_before_native_v2_initialization(tmp_path, monkeypatch):
    db = tmp_path / "retained-v1.db"
    with oc.database(db) as conn:
        conn.execute("CREATE TABLE session(id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO session VALUES ('ses_legacy')")
    monkeypatch.setattr(oc, "db_path", lambda: db)
    with pytest.raises(oc.LegacySessionUnsupported, match="unsupported"):
        oc.Opencode2Adapter().transcript_path("/tmp", "ses_legacy")
    with oc.database(db) as conn:
        assert conn.execute("SELECT id FROM session").fetchone()[0] == "ses_legacy"
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='session_v2'").fetchone() is None


def test_transcript_discovery_preserves_database_errors(tmp_path, monkeypatch):
    import sqlite3

    db = tmp_path / "corrupt.db"
    db.write_text("not a sqlite database")
    monkeypatch.setattr(oc, "db_path", lambda: db)
    with pytest.raises(sqlite3.DatabaseError):
        oc.Opencode2Adapter().transcript_path("/tmp", "ses_test")


def test_idle_plan_restore_keeps_passive_synthetic_steer(store, ctx):
    adapter = oc.Opencode2Adapter()
    adapter.shadow_append(store, turn(adapter, ctx))
    reminder = json.dumps({"text": "You are no longer in Plan mode.", "description": "Plan mode exited"})
    with oc.database(store) as conn:
        conn.execute("INSERT INTO session_message VALUES ('msg_switch','ses_test','agent-switched',24,1,1,?)", (json.dumps({"agent": "build", "previous": "plan", "time": {"created": 1}}),))
        conn.execute("INSERT INTO session_inbox VALUES ('msg_reminder','ses_test','synthetic',?,'steer',25,1)", (reminder,))
        conn.execute("UPDATE event_sequence SET seq=25")
    assert adapter.session_status("ses_test") == "waiting"
    adapter.shadow_append(store, turn(adapter, ctx))
    with oc.database(store) as conn:
        row = conn.execute("SELECT payload,delivery FROM session_inbox WHERE id='msg_reminder'").fetchone()
        assert row["payload"] == reminder and row["delivery"] == "steer"
        assert conn.execute("SELECT seq FROM event_sequence").fetchone()[0] == 28
        conn.execute("UPDATE session_v2 SET time_suspended=2")
    assert adapter.session_status("ses_test") == "busy"
    with pytest.raises(ShadowBusy):
        adapter.shadow_append(store, turn(adapter, ctx))


@pytest.mark.parametrize("kind,delivery", [("synthetic", "queue"), ("user", "steer"), ("compaction", "steer"), ("move", "steer")])
def test_executable_inbox_still_blocks_idle_append(store, ctx, kind, delivery):
    adapter = oc.Opencode2Adapter()
    adapter.shadow_append(store, turn(adapter, ctx))
    with oc.database(store) as conn:
        conn.execute("INSERT INTO session_inbox VALUES ('msg_work','ses_test',?,'{}',?,24,1)", (kind, delivery))
    assert adapter.session_status("ses_test") == "busy"
    with pytest.raises(ShadowBusy):
        adapter.shadow_append(store, turn(adapter, ctx))


@pytest.mark.parametrize("state", [None, "not-json", "[]", "null", '{"phase":"sessions"}', '{"phase":"sessions","cursor":"ses_z"}', '{"phase":"completed"}', '{"phase":"unknown"}'])
def test_retained_session_blocked_across_storage_entrypoints(store, ctx, state, monkeypatch):
    adapter = oc.Opencode2Adapter()
    entries = turn(adapter, ctx)
    with oc.database(store) as conn:
        conn.executescript("CREATE TABLE session(id TEXT PRIMARY KEY);"
                          "INSERT INTO session VALUES ('ses_test');"
                          "CREATE TABLE kv(key TEXT PRIMARY KEY,value TEXT);")
        if state is not None:
            conn.execute("INSERT INTO kv VALUES ('migration.v1-v2',?)", (state,))
        before = list(conn.iterdump())
    session = SimpleNamespace(native_id=lambda hid: "ses_test")
    cursor = SimpleNamespace(pending={"source_pos": {"seq": 0}}, line_index=0)
    monkeypatch.setattr(oc, "_run", lambda *a, **kw: pytest.fail("must not spawn native CLI"))
    operations = [
        lambda: adapter.transcript_path("/tmp", "ses_test"),
        lambda: adapter.make_source_reader(session, cursor, store),
        lambda: oc.Opencode2TurnReader("ses_test", store, cursor).poll(),
        lambda: adapter.pending_units(session, cursor),
        lambda: adapter.fast_forward_cursor(session, cursor),
        lambda: adapter.session_status("ses_test"),
        lambda: adapter.prepare_shadow(store, ctx),
        lambda: adapter.shadow_intent(store, entries),
        lambda: adapter.shadow_append(store, entries),
        lambda: adapter.create_shadow_transcript("/tmp", "ses_test", ctx, "seed"),
    ]
    for operation in operations:
        with pytest.raises(oc.LegacySessionUnsupported, match="unsupported"):
            operation()
    assert "unsupported" in adapter.validate_transcript(store, "ses_test")[0]
    assert cursor.pending == {"source_pos": {"seq": 0}}
    with oc.database(store) as conn:
        assert list(conn.iterdump()) == before


@pytest.mark.parametrize("position,line_index", [({"time": 1, "id": "old"}, 0), ({"seq": "1"}, 0), ({"seq": True}, 0), ({"seq": -2}, 0), ({"unknown": 1}, 0), (None, 1), ({}, 1)])
def test_unsafe_cursor_blocked_before_reads_or_fast_forward(store, position, line_index):
    adapter = oc.Opencode2Adapter()
    cursor = SimpleNamespace(pending={"source_pos": position}, line_index=line_index)
    session = SimpleNamespace(native_id=lambda hid: "ses_test")
    original = json.loads(json.dumps(cursor.pending))
    for operation in (
        lambda: oc.Opencode2TurnReader("ses_test", store, cursor).poll(),
        lambda: adapter.make_source_reader(session, cursor, store),
        lambda: adapter.pending_units(session, cursor),
        lambda: adapter.fast_forward_cursor(session, cursor),
    ):
        with pytest.raises(oc.LegacySessionUnsupported, match="cursor"):
            operation()
    assert cursor.pending == original and cursor.line_index == line_index


def test_virgin_cursor_includes_native_sequence_zero(store):
    with oc.database(store) as conn:
        conn.execute("INSERT INTO session_message VALUES ('msg_zero','ses_test','user',0,1,1,?)", (json.dumps({"text": "first", "time": {"created": 1}}),))
        conn.execute("INSERT INTO session_message VALUES ('msg_idle','ses_test','idle',1,1,1,?)", (json.dumps({"outcome": "succeeded", "time": {"created": 1}}),))
    cursor = SimpleNamespace(pending={}, line_index=0)
    units = oc.Opencode2TurnReader("ses_test", store, cursor).poll()
    assert len(units) == 1 and units[0].raw["messages"][0]["id"] == "msg_zero"
    assert units[0].pos == {"seq": 1}
    session = SimpleNamespace(native_id=lambda hid: "ses_test")
    assert oc.Opencode2Adapter().pending_units(session, cursor) == 2
    cursor.pending["source_pos"] = {"seq": 1}
    cursor.line_index = 1
    assert oc.Opencode2TurnReader("ses_test", store, cursor).poll() == []


def test_fresh_identity_usable_beside_pending_legacy_migration(store, ctx):
    with oc.database(store) as conn:
        conn.executescript("CREATE TABLE session(id TEXT PRIMARY KEY); INSERT INTO session VALUES ('ses_other');")
    adapter = oc.Opencode2Adapter()
    adapter.shadow_append(store, turn(adapter, ctx))
    assert adapter.transcript_path("/tmp", "ses_test") == store
    cursor = SimpleNamespace(pending={}, line_index=0)
    assert len(oc.Opencode2TurnReader("ses_test", store, cursor).poll()) == 1


def test_empty_fast_forward_position_is_before_sequence_zero(store):
    cursor = SimpleNamespace(pending={}, line_index=0)
    session = SimpleNamespace(native_id=lambda hid: "ses_test")
    oc.Opencode2Adapter().fast_forward_cursor(session, cursor)
    assert cursor.pending["source_pos"] == {"seq": -1}


def test_legacy_lifecycle_rechecked_inside_append_transaction(store, ctx, monkeypatch):
    with oc.database(store) as conn:
        conn.executescript("CREATE TABLE session(id TEXT PRIMARY KEY); INSERT INTO session VALUES ('ses_test');"
                          "CREATE TABLE kv(key TEXT PRIMARY KEY,value TEXT);"
                          "INSERT INTO kv VALUES ('migration.v1-v2','{\"phase\":\"sessions\"}');")
    statements = []
    original_connect = oc.connect
    def traced_connect(db):
        conn = original_connect(db)
        conn.set_trace_callback(statements.append)
        return conn
    monkeypatch.setattr(oc, "connect", traced_connect)
    adapter = oc.Opencode2Adapter()
    with pytest.raises(oc.LegacySessionUnsupported):
        adapter.shadow_append(store, turn(adapter, ctx))
    begin = statements.index("BEGIN IMMEDIATE")
    state_read = next(i for i, sql in enumerate(statements) if "SELECT value FROM kv" in sql)
    assert begin < state_read
    assert "ROLLBACK" in statements
    assert not any(sql.startswith("INSERT INTO session_message") for sql in statements)


def test_preflight_without_native_identity_allows_uninitialized_database(monkeypatch):
    monkeypatch.setattr(oc, "db_path", lambda: None)
    session = SimpleNamespace(native_id=lambda harness: None)
    cursor = SimpleNamespace(pending={}, line_index=0, byte_offset=0)
    oc.Opencode2Adapter().preflight_session(session, [cursor])
    assert cursor.pending == {}
