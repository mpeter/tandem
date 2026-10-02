"""Round-trips through the REAL opencode binary (skip-if-missing).

Isolated via OPENCODE_DB -> a temp database that opencode migrates into
existence on first run; the operator's real DB is never touched.
"""

import json
import os
import shutil
import subprocess

import pytest

from tandem import compat
from tandem.harness import get_adapter, opencode

_version = compat.detect_cli_version("opencode")
pytestmark = pytest.mark.skipif(
    shutil.which("opencode") is None
    or _version is None
    or not compat.version_supported("opencode", _version),
    reason="no supported opencode binary on PATH",
)


@pytest.fixture
def oracle_env(tmp_path, monkeypatch):
    db = tmp_path / "oracle.db"
    isolated = {
        "HOME": str(tmp_path), "OPENCODE_TEST_HOME": str(tmp_path),
        "OPENCODE_DB": str(db), "OPENCODE_CONFIG_CONTENT": "{}",
        "OPENCODE_CONFIG_DIR": str(tmp_path / "config"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg-config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "OPENCODE_DISABLE_FILEWATCHER": "true",
        "OPENCODE_DISABLE_MODELS_FETCH": "true",
        "OPENCODE_CONFIG_PROJECT_DISABLE": "1",
    }
    for key, value in isolated.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("OPENCODE_CLI_CONFIG_CONTENT", raising=False)
    opencode._reset_db_cache()
    cwd = tmp_path / "proj"
    cwd.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=cwd, check=True)
    env = dict(os.environ, OPENCODE_DB=str(db))
    return db, str(cwd), env


def _export(sid, cwd, env):
    args = (["opencode", "session", "export", sid, "--standalone"]
            if compat.opencode_major() == 2 else ["opencode", "export", sid])
    out = subprocess.run(args, cwd=cwd, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _texts(data):
    if compat.opencode_major() == 2:
        return [m.get("text", "") for m in data["messages"]] + [
            p.get("text", "") for m in data["messages"] for p in m.get("content", [])]
    return [p.get("text", "") for m in data["messages"] for p in m["parts"]]


def test_import_birth_roundtrips_through_export(oracle_env):
    db, cwd, env = oracle_env
    adapter = get_adapter("opencode")
    from tandem.events import SessionContext
    sid = adapter.mint_session_id()
    ctx = SessionContext(tandem_id="t", cwd=cwd,
                         direction="claude->opencode",
                         target_session_id=sid)
    path = adapter.create_shadow_transcript(cwd, sid, ctx, "[tandem] oracle seed")
    assert path == db
    data = _export(sid, cwd, env)
    assert data["info"]["id"] == sid
    texts = _texts(data)
    assert any("oracle seed" in t for t in texts)


def test_synced_turn_survives_export(oracle_env):
    db, cwd, env = oracle_env
    adapter = get_adapter("opencode")
    from tandem.events import (AssistantMessage, SessionContext, ToolCall,
                               ToolResult, UserMessage)
    sid = adapter.mint_session_id()
    ctx = SessionContext(tandem_id="t", cwd=cwd,
                         direction="claude->opencode",
                         target_session_id=sid)
    adapter.create_shadow_transcript(cwd, sid, ctx, "[tandem] oracle seed")
    events = [
        UserMessage(source="user", turn_index=1, text="[via claude-code] hello"),
        ToolCall(source="claude", turn_index=1, call_id="c1", tool="Bash",
                 arguments={"command": "true"}),
        ToolResult(source="claude", turn_index=1, call_id="c1", output="ok"),
        AssistantMessage(source="claude", turn_index=1,
                         text="[via claude-code] done", model="claude-fable-5"),
    ]
    adapter.shadow_append(db, adapter.render_events(events, ctx))
    data = _export(sid, cwd, env)
    texts = _texts(data)
    assert any("hello" in t for t in texts)
    assert any("done" in t for t in texts)
    roles = ([m["type"] for m in data["messages"] if m["type"] in ("user", "assistant")]
             if compat.opencode_major() == 2 else [m["info"]["role"] for m in data["messages"]])
    assert roles[-1] == "assistant"


def test_session_listed(oracle_env):
    db, cwd, env = oracle_env
    adapter = get_adapter("opencode")
    from tandem.events import SessionContext
    sid = adapter.mint_session_id()
    ctx = SessionContext(tandem_id="t", cwd=cwd,
                         direction="claude->opencode",
                         target_session_id=sid)
    adapter.create_shadow_transcript(cwd, sid, ctx, "[tandem] oracle seed")
    args = ["opencode", "session", "list"]
    if compat.opencode_major() == 2:
        args.append("--standalone")
    out = subprocess.run(args, cwd=cwd, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert sid in out.stdout


def test_retained_v1_history_refused_without_native_or_state_writes(oracle_env, monkeypatch):
    if compat.opencode_major() != 2:
        pytest.skip("OpenCode 2 containment oracle")
    from tandem.events import SessionContext
    from tandem.harness import opencode2
    from tandem.state import StateStore
    from test_opencode import MINI_SCHEMA

    db, cwd, env = oracle_env
    adapter = get_adapter("opencode")
    bootstrap = adapter.mint_session_id()
    ctx = SessionContext(tandem_id="t", cwd=cwd, direction="claude->opencode",
                         target_session_id=bootstrap)
    adapter.create_shadow_transcript(cwd, bootstrap, ctx, "schema bootstrap")
    sid = adapter.mint_session_id()
    user_id, assistant_id = opencode.mint_id("msg"), opencode.mint_id("msg")
    with opencode2.database(db) as conn:
        conn.executescript(MINI_SCHEMA.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS "))
        project = conn.execute("SELECT project_id FROM session_v2 WHERE id=?", (bootstrap,)).fetchone()[0]
        conn.execute("INSERT INTO session(id,project_id,slug,directory,title,version,time_created,time_updated) "
                     "VALUES (?,?,?,?,?,'1.18.20',1,3)", (sid, project, "old-pair", cwd, "legacy task"))
        conn.executemany("INSERT INTO message VALUES (?, ?, ?, ?, ?)", [
            (user_id, sid, 1, 1, json.dumps({"role": "user", "time": {"created": 1}})),
            (assistant_id, sid, 2, 3, json.dumps({"role": "assistant", "time": {"created": 2, "completed": 3},
                                                "finish": "stop", "modelID": "old-model", "providerID": "tandem"})),
        ])
        conn.executemany("INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)", [
            (opencode.mint_id("prt"), user_id, sid, 1, 1, json.dumps({"type": "text", "text": "retained prompt"})),
            (opencode.mint_id("prt"), assistant_id, sid, 2, 3, json.dumps({"type": "text", "text": "retained answer"})),
        ])
    monkeypatch.setenv("TANDEM_HOME", str(db.parent / "tandem"))
    with StateStore() as store:
        pair = store.create_session(cwd, "opencode", ["codex", "opencode"], {"codex": "x", "opencode": sid})
        cursor = store.get_cursor(pair.tandem_id, "opencode", "codex")
        cursor.line_index = 1
        cursor.pending["source_pos"] = {"time": 2, "id": assistant_id}
        store.save_cursor(cursor)
        original_cursor = store.get_cursor(pair.tandem_id, "opencode", "codex")
    before = db.read_bytes()
    with opencode2.database(db) as conn:
        original_rows = list(conn.iterdump())
    monkeypatch.setattr(compat, "detect_cli_version", lambda binary: "2.0.21")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("retained operations must not invoke native CLI"))
    monkeypatch.setattr(opencode, "_run", subprocess.run)
    monkeypatch.setattr(opencode2, "_run", subprocess.run)
    with pytest.raises(opencode2.LegacySessionUnsupported):
        adapter.transcript_path(cwd, sid)
    assert db.read_bytes() == before
    with opencode2.database(db) as conn:
        assert list(conn.iterdump()) == original_rows
    with StateStore() as store:
        assert store.get_cursor(pair.tandem_id, "opencode", "codex") == original_cursor
        assert store.get_session(pair.tandem_id).native_session_ids == pair.native_session_ids
