"""Retained-history cut safety and real native fixed-ID migration recovery."""

import json
import os
import signal
import subprocess
from pathlib import Path

import pytest

from tandem.harness import opencode2 as oc
from tandem.opencode_migration import (
    _check_source,
    _cut,
    _source_digest,
    migrate_opencode,
)
from tandem.opencode_native_snapshot import capture_source, import_payload
from tandem.state import StateStore


def test_cut_refuses_mixed_fold_and_locale_reorder():
    source = [{"id": "A", "time_created": 1}, {"id": "a", "time_created": 1}]
    with pytest.raises(ValueError, match="folded"):
        _cut(source, [{"A", "a"}], {"id": "A", "time": 1})
    with pytest.raises(ValueError, match="ordering"):
        _cut(source, [{"a"}, {"A"}], {"id": "A", "time": 1})
    assert _cut(source, [{"a"}, {"A"}], {"id": "a", "time": 1}) == 2
    with pytest.raises(ValueError, match="missing"):
        _cut(source, [{"a"}], {"id": "a", "time": 2})


@pytest.mark.parametrize("change", ["attachment", "unfinished", "parent", "orphan"])
def test_source_refuses_unsupported_history(change):
    snapshot = {
        "session": [{}],
        "message": [{"id": "u", "data": '{"role":"user"}'}],
        "part": [],
    }
    if change == "attachment":
        snapshot["part"] = [{"message_id": "u", "data": '{"type":"file"}'}]
    elif change == "unfinished":
        snapshot["message"].append({"id": "a", "data": '{"role":"assistant"}'})
    elif change == "parent":
        snapshot["session"][0]["parent_id"] = "p"
    else:
        snapshot["part"] = [{"message_id": "missing", "data": '{"type":"text"}'}]
    with pytest.raises(ValueError):
        _check_source(snapshot)


@pytest.fixture
def native_pair(tmp_path):
    binary = os.environ.get("TANDEM_RETAINED_NATIVE")
    if not binary:
        pytest.skip(
            "set TANDEM_RETAINED_NATIVE to pinned OpenCode 2.0.21 for disposable native tests"
        )
    db = tmp_path / "native.db"
    payload = {
        "info": {
            "id": "ses_bootstrap",
            "projectID": "bootstrap",
            "agent": "build",
            "location": {"directory": str(tmp_path)},
            "cost": 0,
            "tokens": {
                "input": 0,
                "output": 0,
                "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            },
            "time": {"created": 1, "updated": 1},
        },
        "messages": [],
    }
    import_payload(binary, db, payload, tmp_path)
    with oc.database(db) as conn:
        project = conn.execute(
            "SELECT project_id FROM session_v2 WHERE id='ses_bootstrap'"
        ).fetchone()[0]
        conn.executescript("""
            CREATE TABLE session (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, workspace_id TEXT, parent_id TEXT,
                slug TEXT NOT NULL, directory TEXT NOT NULL, path TEXT, title TEXT NOT NULL,
                version TEXT NOT NULL, share_url TEXT, summary_additions INTEGER,
                summary_deletions INTEGER, summary_files INTEGER, summary_diffs TEXT, metadata TEXT,
                cost REAL DEFAULT 0 NOT NULL, tokens_input INTEGER DEFAULT 0 NOT NULL,
                tokens_output INTEGER DEFAULT 0 NOT NULL, tokens_reasoning INTEGER DEFAULT 0 NOT NULL,
                tokens_cache_read INTEGER DEFAULT 0 NOT NULL, tokens_cache_write INTEGER DEFAULT 0 NOT NULL,
                revert TEXT, permission TEXT, agent TEXT, model TEXT, time_created INTEGER NOT NULL,
                time_updated INTEGER NOT NULL, time_compacting INTEGER, time_archived INTEGER);
            CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                time_created INTEGER NOT NULL,time_updated INTEGER NOT NULL,data TEXT NOT NULL);
            CREATE TABLE part (id TEXT PRIMARY KEY,message_id TEXT NOT NULL,session_id TEXT NOT NULL,
                time_created INTEGER NOT NULL,time_updated INTEGER NOT NULL,data TEXT NOT NULL);
        """)
        conn.execute(
            "INSERT INTO session (id,project_id,slug,directory,title,version,time_created,time_updated) VALUES (?,?,?,?,?,?,?,?)",
            ("ses_old", project, "old", str(tmp_path), "retained", "1.18.15", 1, 6),
        )
        for index, text in enumerate(
            ("FIRST_USER", "FIRST_ANSWER", "SECOND_USER", "SECOND_ANSWER"), 1
        ):
            data = {
                "role": "user",
                "time": {"created": index},
                "agent": "build",
                "model": {"providerID": "openai", "modelID": "gpt-6.1-sol"},
            }
            if index % 2 == 0:
                data = {
                    "role": "assistant",
                    "time": {"created": index, "completed": index + 1},
                    "parentID": f"msg_{index - 1}",
                    "providerID": "openai",
                    "modelID": "gpt-6.1-sol",
                    "mode": "build",
                    "agent": "build",
                    "path": {"cwd": str(tmp_path), "root": str(tmp_path)},
                    "cost": 0,
                    "tokens": payload["info"]["tokens"],
                    "finish": "stop",
                }
            conn.execute(
                "INSERT INTO message VALUES (?,?,?,?,?)",
                (f"msg_{index}", "ses_old", index, index + 1, json.dumps(data)),
            )
            conn.execute(
                "INSERT INTO part VALUES (?,?,?,?,?,?)",
                (
                    f"prt_{index}",
                    f"msg_{index}",
                    "ses_old",
                    index,
                    index + 1,
                    json.dumps({"type": "text", "text": text}),
                ),
            )
    db.chmod(0o600)
    with StateStore(tmp_path / "state.db") as store:
        pair = store.create_session(
            str(tmp_path),
            "opencode",
            ["opencode", "codex", "claude"],
            {"opencode": "ses_old", "codex": "codex", "claude": "claude"},
        )
        cursor = store.get_cursor(pair.tandem_id, "opencode", "codex")
        cursor.pending = {"source_pos": {"time": 2, "id": "msg_2"}}
        cursor.line_index = 1
        cursor.failed_turns = 3
        store.save_cursor(cursor)
        yield binary, db, store, pair


def _migrate(native_pair):
    binary, db, store, pair = native_pair
    return migrate_opencode(store, pair.tandem_id, db, binary=binary, scratch=db.parent)


def test_native_migration_unconsumed_history_and_exact_retry(native_pair):
    binary, db, store, pair = native_pair
    original = _source_digest(capture_source(db, "ses_old"))
    result = _migrate(native_pair)
    assert result["fresh_id"] != "ses_old"
    current = store.get_session(pair.tandem_id)
    cursor = store.get_cursor(pair.tandem_id, "opencode", "codex")
    reader = oc.Opencode2TurnReader(result["fresh_id"], db, cursor, current)
    lines = reader.poll()
    assert len(lines) == 1 and lines[0].line_index == 1
    assert [
        m.get("text") or m.get("content", [{}])[0].get("text")
        for m in lines[0].raw["messages"]
    ] == ["SECOND_USER", "SECOND_ANSWER"]
    assert cursor.failed_turns == 3
    cursor.line_index += 1
    cursor.pending["source_pos"] = lines[0].pos
    store.save_cursor(cursor)
    assert reader.poll() == []
    assert _migrate(native_pair) == result
    assert _source_digest(capture_source(db, "ses_old")) == original
    with oc.database(db) as conn:
        info = conn.execute(
            "SELECT directory FROM session_v2 WHERE id=?", (result["fresh_id"],)
        ).fetchone()
        assert info[0] == str(db.parent)
        rows = conn.execute(
            "SELECT type,seq FROM session_message WHERE session_id=? ORDER BY seq",
            (result["fresh_id"],),
        ).fetchall()
        assert [row["seq"] for row in rows] == [1, 2, 3, 4]
        assert all(row["type"] != "idle" for row in rows)
        assert (
            conn.execute(
                "SELECT seq FROM event_sequence WHERE aggregate_id=?",
                (result["fresh_id"],),
            ).fetchone()[0]
            >= 4
        )


@pytest.mark.parametrize(
    "phase", ["prepared", "imported", "before_commit", "committed"]
)
def test_native_sigkill_fixed_identity_recovery(native_pair, phase):
    binary, db, store, pair = native_pair
    child = """
import os,signal,sys
from pathlib import Path
from tandem.state import StateStore
import tandem.opencode_migration as m
phase,sdb,pair,db,binary=sys.argv[1:]
def kill(): os.kill(os.getpid(),signal.SIGKILL)
if phase=='prepared':
 original=StateStore.prepare_opencode_reconciliation
 def prepare(self,*a): original(self,*a); kill()
 StateStore.prepare_opencode_reconciliation=prepare
elif phase=='imported':
 original=m.import_payload
 def imported(*a): original(*a); kill()
 m.import_payload=imported
else:
 original=StateStore.commit_opencode_reconciliation
 def commit(self,*a):
  if phase=='before_commit': kill()
  original(self,*a); kill()
 StateStore.commit_opencode_reconciliation=commit
with StateStore(Path(sdb)) as s: m.migrate_opencode(s,pair,Path(db),binary=binary,scratch=Path(db).parent)
"""
    proc = subprocess.run(
        [
            str(Path(os.sys.executable)),
            "-c",
            child,
            phase,
            str(store.db_path),
            pair.tandem_id,
            str(db),
            binary,
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == -signal.SIGKILL, proc.stderr
    prepared = store.get_opencode_reconciliation(pair.tandem_id)
    assert prepared is not None
    fresh = prepared["fresh_id"]
    result = _migrate(native_pair)
    assert result["fresh_id"] == fresh
    assert store.get_opencode_reconciliation(pair.tandem_id)["phase"] == "committed"
    assert store.get_session(pair.tandem_id).native_id("opencode") == fresh


def test_native_changed_cursor_preserves_fixed_unbound_import(native_pair, monkeypatch):
    binary, db, store, pair = native_pair
    original = store.commit_opencode_reconciliation

    def changed(tid):
        cursor = store.get_cursor(tid, "opencode", "claude")
        cursor.failed_turns += 1
        store.save_cursor(cursor)
        original(tid)

    monkeypatch.setattr(store, "commit_opencode_reconciliation", changed)
    with pytest.raises(ValueError, match="changed"):
        _migrate(native_pair)
    record = store.get_opencode_reconciliation(pair.tandem_id)
    assert record["phase"] == "prepared"
    assert store.get_session(pair.tandem_id).native_id("opencode") == "ses_old"
    with oc.database(db) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM session_message WHERE session_id=?",
                (record["fresh_id"],),
            ).fetchone()[0]
            == 4
        )
        conn.execute(
            "UPDATE session_v2 SET agent='plan' WHERE id=?", (record["fresh_id"],)
        )
    with pytest.raises(ValueError, match="Info"):
        _migrate(native_pair)


@pytest.mark.parametrize("change", ["source", "prefix", "suffix"])
def test_native_content_changes_preserve_unbound_history(
    native_pair, monkeypatch, change
):
    binary, db, store, pair = native_pair
    from tandem import opencode_migration as migration

    original_import = migration.import_payload

    def changed(*args):
        original_import(*args)
        fresh = store.get_opencode_reconciliation(pair.tandem_id)["fresh_id"]
        with oc.database(db) as conn:
            if change == "source":
                conn.execute(
                    "UPDATE part SET data=? WHERE id='prt_1'",
                    ('{"type":"text","text":"changed"}',),
                )
            elif change == "prefix":
                conn.execute(
                    "UPDATE session_message SET data=? WHERE session_id=? AND seq=2",
                    ('{"time":{"created":2,"completed":3},"content":[]}', fresh),
                )
            else:
                conn.execute(
                    "INSERT INTO session_message VALUES (?,?,?,?,?,?,?)",
                    (
                        "msg_suffix",
                        "ses_old",
                        "user",
                        4,
                        8,
                        8,
                        '{"time":{"created":8},"text":"new retained work"}',
                    ),
                )

    monkeypatch.setattr(migration, "import_payload", changed)
    with pytest.raises(ValueError, match="changed|conflicts|suffix"):
        _migrate(native_pair)
    record = store.get_opencode_reconciliation(pair.tandem_id)
    assert record["phase"] == "prepared"
    assert store.get_session(pair.tandem_id).native_id("opencode") == "ses_old"
    with oc.database(db) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM session_message WHERE session_id=?",
                (record["fresh_id"],),
            ).fetchone()[0]
            == 4
        )


def _wait_ready(path, proc):
    import time

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        assert proc.poll() is None, proc.communicate()
        time.sleep(0.05)
    raise AssertionError("Owned native helper did not reach observed phase")


def _running(pid):
    status = Path(f"/proc/{pid}/stat")
    return (
        status.exists()
        and status.read_text().split(")", 1)[1].strip().split()[0] != "Z"
    )


@pytest.mark.parametrize(
    "phase,signum",
    [
        ("oracle", signal.SIGKILL),
        ("oracle", signal.SIGTERM),
        ("import", signal.SIGKILL),
    ],
)
def test_native_mid_helper_parent_death_and_lease_recovery(native_pair, phase, signum):
    import time
    from tandem.opencode_migration import migration_lease

    binary, db, store, pair = native_pair
    source = _source_digest(capture_source(db, "ses_old"))
    ready = db.parent / "helper-ready.json"
    child = """
import json,sys,time
from pathlib import Path
from tandem.state import StateStore
import tandem.opencode_migration as m
import tandem.opencode_native_snapshot as n
phase,sdb,pair,db,binary,ready=sys.argv[1:]
if phase=='oracle':
 original=n.NativeServer.request
 def request(self,method,route,body=None):
  if method=='POST' and route.endswith('/fork'):
   Path(ready).write_text(json.dumps({'pid':self.native_pid,'port':self.port}))
   while True:time.sleep(0.05)
  return original(self,method,route,body)
 n.NativeServer.request=request
else:
 original=n.start_native
 def start(args,root,env,*a,**kw):
  result=original(args,root,env,*a,**kw)
  if env.get('OPENCODE_DB')==db:
   Path(ready).write_text(json.dumps({'pid':result[2]}))
  return result
 n.start_native=start
with StateStore(Path(sdb)) as s:m.migrate_opencode(s,pair,Path(db),binary=binary,scratch=Path(db).parent)
"""
    lock = oc.connect(db)
    if phase == "import":
        lock.execute("BEGIN IMMEDIATE")
    proc = subprocess.Popen(
        [
            os.sys.executable,
            "-c",
            child,
            phase,
            str(store.db_path),
            pair.tandem_id,
            str(db),
            binary,
            str(ready),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        observed = _wait_ready(ready, proc)
        assert _running(observed["pid"])
        with pytest.raises(ValueError, match="active retained migration"):
            with migration_lease(store, pair.tandem_id):
                pass
        if phase == "import":
            time.sleep(0.5)
            assert _running(observed["pid"]) and proc.poll() is None
            record = store.get_opencode_reconciliation(pair.tandem_id)
            assert record["phase"] == "prepared"
            assert (
                lock.execute(
                    "SELECT count(*) FROM session_v2 WHERE id=?", (record["fresh_id"],)
                ).fetchone()[0]
                == 0
            )
        proc.send_signal(signum)
        proc.wait(timeout=5)
        assert proc.returncode == -signum
        deadline = time.monotonic() + 10
        while _running(observed["pid"]) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _running(observed["pid"])
        if phase == "oracle":
            import socket

            with socket.socket() as sock:
                assert sock.connect_ex(("127.0.0.1", observed["port"])) != 0
        lock.rollback()
        deadline = time.monotonic() + 5
        while True:
            try:
                with migration_lease(store, pair.tandem_id):
                    pass
                break
            except ValueError:
                assert time.monotonic() < deadline
                time.sleep(0.05)
        result = _migrate(native_pair)
        if phase == "import":
            assert result["fresh_id"] == record["fresh_id"]
        assert (
            store.get_session(pair.tandem_id).native_id("opencode")
            == result["fresh_id"]
        )
        assert _source_digest(capture_source(db, "ses_old")) == source
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        lock.close()
        for stream in (proc.stdout, proc.stderr):
            stream.close()


@pytest.mark.parametrize("caught_up", [False, True])
def test_native_checkpoint_does_not_reinject_archival_context(native_pair, caught_up):
    from tandem.events import SessionContext

    binary, db, store, pair = native_pair
    secret = "ARCHIVAL-CONTEXT-NOT-IN-CHECKPOINT"
    with oc.database(db) as conn:
        conn.execute(
            "UPDATE part SET data=? WHERE id='prt_1'",
            (json.dumps({"type": "text", "text": secret}),),
        )
        conn.execute(
            "UPDATE part SET data=? WHERE id='prt_3'",
            (json.dumps({"type": "compaction", "auto": True}),),
        )
        summary = json.loads(
            conn.execute("SELECT data FROM message WHERE id='msg_4'").fetchone()[0]
        )
        summary["summary"] = True
        conn.execute(
            "UPDATE message SET data=? WHERE id='msg_4'", (json.dumps(summary),)
        )
        conn.execute(
            "UPDATE part SET data=? WHERE id='prt_4'",
            (json.dumps({"type": "text", "text": "ACTIVE-CHECKPOINT-SUMMARY"}),),
        )
    if not caught_up:
        with pytest.raises(ValueError, match="archival context"):
            _migrate(native_pair)
        assert store.get_opencode_reconciliation(pair.tandem_id) is None
        return
    cursor = store.get_cursor(pair.tandem_id, "opencode", "claude")
    cursor.pending = {"source_pos": {"time": 2, "id": "msg_2"}}
    store.save_cursor(cursor)
    result = _migrate(native_pair)
    current = store.get_session(pair.tandem_id)
    reader = oc.Opencode2TurnReader(
        result["fresh_id"],
        db,
        cursor := store.get_cursor(pair.tandem_id, "opencode", "claude"),
        current,
    )
    lines = reader.poll()
    assert len(lines) == 1 and lines[0].line_index == 0
    events = oc.Opencode2Adapter().parse_entry(
        lines[0].raw,
        SessionContext(
            tandem_id=pair.tandem_id, cwd=str(db.parent), direction="opencode->claude"
        ),
    )
    texts = "\n".join(getattr(event, "text", "") for event in events)
    assert "ACTIVE-CHECKPOINT-SUMMARY" in texts and secret not in texts
    with oc.database(db) as conn:
        assert (
            secret
            in conn.execute("SELECT data FROM part WHERE id='prt_1'").fetchone()[0]
        )
        assert (
            secret
            in conn.execute(
                "SELECT data FROM session_message WHERE session_id=? AND seq=1",
                (result["fresh_id"],),
            ).fetchone()[0]
        )
