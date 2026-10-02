import copy
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tandem import ops
from tandem.harness import opencode2
from tandem.harness.opencode import OpencodeAdapter
from tandem.events import SessionContext, SystemEvent
from tandem.opencode_binding import MARKER_KEY, PROOF_KEY, digest, prefix_rows, validate_pair_prefix, validate_prefix
from tandem.opencode_migration import _check_source, _mapping
from tandem.state import StateStore, SyncCursor


@pytest.fixture
def reconciled(tmp_path):
    native = tmp_path / "native.db"
    with closing(sqlite3.connect(native)) as conn, conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            "CREATE TABLE session_v2(id TEXT PRIMARY KEY, metadata TEXT);"
            "CREATE TABLE session_message(id TEXT PRIMARY KEY, session_id TEXT, type TEXT,"
            "seq INTEGER, time_created INTEGER, data TEXT);"
        )
        conn.execute("INSERT INTO session_v2 VALUES (?,?)", ("fresh", "{}"))
        messages = [
            ("m1", "user", {"text": "original", "metadata": {MARKER_KEY: "binding"}}),
            ("m2", "assistant", {"content": [{"type": "text", "text": "answer"}], "time": {"completed": 2}}),
        ]
        for seq, (mid, kind, data) in enumerate(messages, 1):
            conn.execute("INSERT INTO session_message VALUES (?,?,?,?,?,?)",
                         (mid, "fresh", kind, seq, seq, json.dumps(data)))
        proof = {"version": 1, "native_version": "2.0.21", "binding_id": "binding",
                 "prefix_end": 2, "prefix_digest": digest(prefix_rows(conn, "fresh", 2)),
                 "units": [{"index": 0, "seq": 2, "echo": False}]}
        conn.execute("UPDATE session_v2 SET metadata=?", (json.dumps({PROOF_KEY: proof}),))
    with StateStore(tmp_path / "state.db") as store:
        pair = store.create_session(str(tmp_path), "claude", ["claude", "opencode"],
                                    {"claude": "peer", "opencode": "fresh"})
        record = {"fresh_id": "fresh", "proof": proof}
        with store._tx() as conn:
            conn.execute("INSERT INTO opencode_reconciliations VALUES (?, 'committed', ?)",
                         (pair.tandem_id, json.dumps(record)))
        yield native, store, store.get_session(pair.tandem_id), proof


def test_pair_admission_detects_stripped_annotations_with_stale_empty_record(reconciled):
    native, store, session, proof = reconciled
    session.opencode_reconciliation = None
    with opencode2.database(native) as conn:
        conn.execute("UPDATE session_v2 SET metadata='{}'")
        conn.execute("UPDATE session_message SET data=? WHERE id='m1'",
                     (json.dumps({"text": "original"}),))
    with opencode2.database(native) as conn:
        with pytest.raises(ValueError, match="reconciliation changed"):
            validate_pair_prefix(conn, session)


def test_pair_admission_detects_removed_authoritative_record(reconciled):
    native, store, session, proof = reconciled
    with store._tx() as conn:
        conn.execute("DELETE FROM opencode_reconciliations")
    with opencode2.database(native) as conn:
        with pytest.raises(ValueError, match="reconciliation changed"):
            validate_pair_prefix(conn, session)


@pytest.mark.parametrize("field,value", [("version", True), ("version", 1.0),
                                         ("prefix_end", True), ("prefix_end", 2.0)])
def test_native_proof_rejects_boolean_and_float_integer_fields(reconciled, field, value):
    native, store, session, proof = reconciled
    corrupt = {**proof, field: value}
    with opencode2.database(native) as conn:
        conn.execute("UPDATE session_v2 SET metadata=?", (json.dumps({PROOF_KEY: corrupt}),))
    with opencode2.database(native) as conn:
        with pytest.raises(ValueError, match="retained-history admission failed"):
            validate_prefix(conn, "fresh")


@pytest.mark.parametrize("field,value", [("index", False), ("index", 0.0),
                                         ("seq", True), ("seq", 2.0), ("echo", 0)])
def test_native_proof_requires_strict_unit_field_types(reconciled, field, value):
    native, store, session, proof = reconciled
    corrupt = copy.deepcopy(proof)
    corrupt["units"][0][field] = value
    with opencode2.database(native) as conn:
        conn.execute("UPDATE session_v2 SET metadata=?", (json.dumps({PROOF_KEY: corrupt}),))
    with opencode2.database(native) as conn:
        with pytest.raises(ValueError, match="retained-history admission failed"):
            validate_prefix(conn, "fresh")


def test_reader_rechecks_authority_after_construction(reconciled):
    native, store, session, proof = reconciled
    cursor = SyncCursor(session.tandem_id, "opencode", "claude",
                        pending={"source_pos": {"seq": 0, "binding": "binding"}})
    reader = opencode2.Opencode2Adapter().make_source_reader(session, cursor, native)
    store.set_native_session_id(session.tandem_id, "opencode", "replacement")
    with pytest.raises(ValueError, match="pair binding changed"):
        reader.poll()


def test_reader_uses_validated_snapshot_during_concurrent_prefix_mutation(reconciled, monkeypatch):
    native, store, session, proof = reconciled
    cursor = SyncCursor(session.tandem_id, "opencode", "claude",
                        pending={"source_pos": {"seq": 0, "binding": "binding"}})
    reader = opencode2.Opencode2Adapter().make_source_reader(session, cursor, native)
    original_messages = opencode2._messages
    changed = False

    def mutate_after_validation(conn, sid, after=-1):
        nonlocal changed
        if not changed:
            with closing(sqlite3.connect(native)) as writer, writer:
                writer.execute("UPDATE session_message SET data=? WHERE id='m1'",
                               (json.dumps({"text": "mutated", "metadata": {MARKER_KEY: "binding"}}),))
            changed = True
        return original_messages(conn, sid, after)

    monkeypatch.setattr(opencode2, "_messages", mutate_after_validation)
    units = reader.poll()
    assert units[0].raw["messages"][0]["text"] == "original"
    with pytest.raises(ValueError, match="prefix changed"):
        reader.poll()


def test_reader_preserves_local_index_without_skipping_unconsumed_history(reconciled):
    native, store, session, proof = reconciled
    cursor = SyncCursor(session.tandem_id, "opencode", "claude", line_index=7,
                        pending={"source_pos": {"seq": 0, "binding": "binding"}})
    reader = opencode2.Opencode2Adapter().make_source_reader(session, cursor, native)
    units = reader.poll()
    assert len(units) == 1
    assert units[0].line_index == 7
    assert units[0].raw["messages"][0]["text"] == "original"


@pytest.mark.parametrize("line_index", [0, 7])
def test_reader_does_not_replay_fast_forwarded_history(reconciled, line_index):
    native, store, session, proof = reconciled
    cursor = SyncCursor(session.tandem_id, "opencode", "claude", line_index=line_index,
                        pending={"source_pos": {"seq": 2, "binding": "binding"}})
    reader = opencode2.Opencode2Adapter().make_source_reader(session, cursor, native)
    assert reader.poll() == []


def test_fast_forward_preserves_local_line_index(reconciled, monkeypatch):
    native, store, session, proof = reconciled
    monkeypatch.setattr(opencode2, "db_path", lambda: native)
    cursor = SyncCursor(session.tandem_id, "opencode", "claude",
                        pending={"source_pos": {"seq": 0, "binding": "binding"}})
    opencode2.Opencode2Adapter().fast_forward_cursor(session, cursor)
    assert cursor.line_index == 0
    assert cursor.pending["source_pos"] == {"seq": 2, "binding": "binding"}


def test_reader_refuses_cursor_inside_historical_turn(reconciled):
    native, store, session, proof = reconciled
    cursor = SyncCursor(session.tandem_id, "opencode", "claude",
                        pending={"source_pos": {"seq": 1, "binding": "binding"}})
    reader = opencode2.Opencode2Adapter().make_source_reader(session, cursor, native)
    with pytest.raises(ValueError, match="cursor"):
        reader.poll()


def test_reader_refuses_cursor_beyond_native_history(reconciled):
    native, store, session, proof = reconciled
    cursor = SyncCursor(session.tandem_id, "opencode", "claude",
                        pending={"source_pos": {"seq": 100, "binding": "binding"}})
    reader = opencode2.Opencode2Adapter().make_source_reader(session, cursor, native)
    with pytest.raises(ValueError, match="cursor"):
        reader.poll()


def test_reader_emits_first_live_suffix_after_fast_forward_at_local_index(reconciled):
    native, store, session, proof = reconciled
    with opencode2.database(native) as conn:
        for seq, kind, data in [(3, "user", {"text": "live user"}),
                                (4, "assistant", {"content": [], "time": {"completed": 4}}),
                                (5, "idle", {"outcome": "succeeded"})]:
            conn.execute("INSERT INTO session_message VALUES (?,?,?,?,?,?)",
                         (f"m{seq}", "fresh", kind, seq, seq, json.dumps(data)))
    cursor = SyncCursor(session.tandem_id, "opencode", "claude",
                        pending={"source_pos": {"seq": 2, "binding": "binding"}})
    reader = opencode2.Opencode2Adapter().make_source_reader(session, cursor, native)
    units = reader.poll()
    assert len(units) == 1
    assert units[0].line_index == 0
    assert units[0].pos == {"seq": 5, "binding": "binding"}
    assert units[0].raw["messages"][0]["text"] == "live user"
    cursor.line_index = 1
    cursor.pending["source_pos"] = units[0].pos
    assert reader.poll() == []


@pytest.fixture
def prepared(tmp_path):
    with StateStore(tmp_path / "state.db") as store:
        session = store.create_session(str(tmp_path), "claude", ["claude", "opencode"],
                                       {"claude": "peer", "opencode": "old"})
        for source, target in [("claude", "opencode"), ("opencode", "claude")]:
            pending = {"source_pos": {"time": 1, "id": "boundary"}} if source == "opencode" else {}
            store.save_cursor(SyncCursor(session.tandem_id, source, target, pending=pending))
        original = [asdict(store.get_cursor(session.tandem_id, source, target))
                    for source, target in [("claude", "opencode"), ("opencode", "claude")]]
        translated = copy.deepcopy(original)
        translated[1].update(byte_offset=17, line_index=3, turn_index=7, failed_turns=2,
                             pending={"source_pos": {"seq": 6, "binding": "binding"}})
        record = {"original_ids": session.native_session_ids, "fresh_id": "fresh",
                  "original_pair": {"participants": session.participants, "cwd": session.cwd,
                                    "active": session.active},
                  "cursor_directions": [["claude", "opencode"], ["opencode", "claude"]],
                  "original_cursors": original, "translated_cursors": translated}
        store.prepare_opencode_reconciliation(session.tandem_id, record)
        yield store, session, record


@pytest.mark.parametrize("change", ["membership", "new_direction", "deleted_direction", "cursor"])
def test_pair_commit_refuses_changed_membership_or_direction_state(prepared, change):
    store, session, record = prepared
    if change == "membership":
        store.set_participants(session.tandem_id, ["claude", "opencode", "codex"])
    elif change == "new_direction":
        store.save_cursor(SyncCursor(session.tandem_id, "opencode", "codex"))
    elif change == "deleted_direction":
        with store._tx() as conn:
            conn.execute("DELETE FROM sync_cursors WHERE source='claude'")
    else:
        cursor = store.get_cursor(session.tandem_id, "opencode", "claude")
        cursor.turn_index = 99
        store.save_cursor(cursor)
    with pytest.raises(ValueError, match="changed"):
        store.commit_opencode_reconciliation(session.tandem_id)
    assert store.get_session(session.tandem_id).native_id("opencode") == "old"
    assert store.get_opencode_reconciliation(session.tandem_id)["phase"] == "prepared"


def test_pair_commit_persists_every_translated_cursor_field(prepared):
    store, session, record = prepared
    store.commit_opencode_reconciliation(session.tandem_id)
    assert store.get_session(session.tandem_id).native_id("opencode") == "fresh"
    actual = asdict(store.get_cursor(session.tandem_id, "opencode", "claude"))
    expected = record["translated_cursors"][1]
    assert {key: value for key, value in actual.items() if key != "updated_at"} == {
        key: value for key, value in expected.items() if key != "updated_at"
    }
    assert store.get_opencode_reconciliation(session.tandem_id)["phase"] == "committed"


@pytest.mark.parametrize("changed_time", [True, 1.0])
def test_pair_commit_detects_pending_json_numeric_type_change(prepared, changed_time):
    store, session, record = prepared
    with store._tx() as conn:
        conn.execute("UPDATE sync_cursors SET pending=? WHERE source='opencode'",
                     (json.dumps({"source_pos": {"time": changed_time, "id": "boundary"}}),))
    with pytest.raises(ValueError, match="cursor changed"):
        store.commit_opencode_reconciliation(session.tandem_id)
    assert store.get_session(session.tandem_id).native_id("opencode") == "old"
    assert store.get_opencode_reconciliation(session.tandem_id)["phase"] == "prepared"


@pytest.mark.parametrize("user_provider,assistant_provider", [("native", "tandem"),
                                                              ("tandem", "native")])
def test_historical_echo_matches_old_parser_whole_turn_classification(user_provider, assistant_provider):
    user = {"id": "user", "role": "user", "model": {"providerID": user_provider}}
    assistant = {"id": "assistant", "role": "assistant", "providerID": assistant_provider}
    raw = {"user": {"message": user, "parts": [{"type": "text", "text": "genuine user"}]},
           "assistants": [{"message": assistant, "parts": [{"type": "text", "text": "answer"}]}]}
    snapshot = {"message": [{"id": "user", "time_created": 1, "data": json.dumps(user)},
                            {"id": "assistant", "time_created": 2, "data": json.dumps(assistant)}]}
    oracle = [{"id": "user", "type": "user", "seq": 0, "data": "{}"},
              {"id": "assistant", "type": "assistant", "seq": 1, "data": "{}"}]
    unit = SimpleNamespace(pos={"id": "assistant", "time": 2}, raw=raw)
    context = SessionContext(tandem_id="pair", cwd="/fixture", direction="opencode->claude")
    old_events = OpencodeAdapter().parse_entry(raw, context)
    old_echo = any(isinstance(event, SystemEvent) and event.subtype == "tandem_echo"
                   for event in old_events)
    historical, origins = _mapping(snapshot, oracle, [unit])
    assert historical[0]["echo"] is old_echo


def test_migration_refuses_nonempty_ignored_text_in_retained_context():
    snapshot = {"session": [{"id": "old"}],
                "message": [{"id": "user", "data": json.dumps({"role": "user"})}],
                "part": [{"message_id": "user", "data": json.dumps(
                    {"type": "text", "text": "pending peer context", "ignored": True})}]}
    with pytest.raises(ValueError, match="ignored"):
        _check_source(snapshot)


@pytest.mark.parametrize("status,compacted", [("completed", None), ("error", 3)])
def test_migration_refuses_tool_attachments_without_native_clear(status, compacted):
    state = {"status": status, "attachments": [{"url": "data:text/plain,context"}],
             "time": {"start": 1, "end": 2}}
    if compacted is not None:
        state["time"]["compacted"] = compacted
    snapshot = {"session": [{"id": "old"}],
                "message": [{"id": "assistant", "data": json.dumps(
                    {"role": "assistant", "time": {"completed": 2}, "finish": "stop"})}],
                "part": [{"message_id": "assistant", "data": json.dumps(
                    {"type": "tool", "state": state})}]}
    with pytest.raises(ValueError, match="attachments"):
        _check_source(snapshot)


def test_migration_allows_completed_tool_native_clear_provenance():
    snapshot = {"session": [{"id": "old"}],
                "message": [{"id": "assistant", "data": json.dumps(
                    {"role": "assistant", "time": {"completed": 2}, "finish": "stop"})}],
                "part": [{"message_id": "assistant", "data": json.dumps(
                    {"type": "tool", "state": {"status": "completed", "output": "old raw output",
                      "attachments": [{"url": "data:text/plain,context"}],
                      "time": {"start": 1, "end": 2, "compacted": 3}}})}]}
    _check_source(snapshot)


def test_ops_adoption_cannot_detach_committed_reconciliation(reconciled, monkeypatch):
    native, store, session, proof = reconciled
    cursor = store.get_cursor(session.tandem_id, "opencode", "claude")
    cursor.pending = {"source_pos": {"seq": 2, "binding": "binding"}}
    store.save_cursor(cursor)
    before_cursor = asdict(store.get_cursor(session.tandem_id, "opencode", "claude"))
    before_record = store.get_opencode_reconciliation(session.tandem_id)
    with opencode2.database(native) as conn:
        conn.execute("INSERT INTO session_v2 VALUES ('virgin','{}')")
    monkeypatch.setattr(opencode2, "db_path", lambda: native)
    monkeypatch.setattr(ops, "get_adapter", lambda harness: opencode2.Opencode2Adapter())
    with pytest.raises(ValueError, match="authoritative StateStore"):
        ops.adopt_native_id(store, session, "opencode", "virgin")
    assert store.get_session(session.tandem_id).native_id("opencode") == "fresh"
    assert store.get_opencode_reconciliation(session.tandem_id) == before_record
    assert asdict(store.get_cursor(session.tandem_id, "opencode", "claude")) == before_cursor


def test_ops_adoption_refuses_imported_identity_without_its_pair_binding(reconciled, monkeypatch):
    native, store, session, proof = reconciled
    with opencode2.database(native) as conn:
        conn.execute("INSERT INTO session_v2 VALUES ('virgin','{}')")
    pair = store.create_session(str(native.parent / "ordinary"), "claude", ["claude", "opencode"],
                                {"claude": "peer", "opencode": "virgin"})
    pair = store.get_session(pair.tandem_id)
    before_cursor = asdict(store.get_cursor(pair.tandem_id, "opencode", "claude"))
    monkeypatch.setattr(opencode2, "db_path", lambda: native)
    monkeypatch.setattr(ops, "get_adapter", lambda harness: opencode2.Opencode2Adapter())
    with pytest.raises(ValueError, match="no committed pair binding"):
        ops.adopt_native_id(store, pair, "opencode", "fresh")
    assert store.get_session(pair.tandem_id).native_id("opencode") == "virgin"
    assert store.get_opencode_reconciliation(pair.tandem_id) is None
    assert asdict(store.get_cursor(pair.tandem_id, "opencode", "claude")) == before_cursor
