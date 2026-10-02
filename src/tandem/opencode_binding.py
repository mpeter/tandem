"""Content validation for explicitly imported retained-history snapshots.

Native metadata proves closure/content only. Committed StateStore records prove
that a paired session owns the replacement identity.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

PROOF_KEY = "tandem_reconciliation"
MARKER_KEY = "tandem_reconciliation_binding"


def digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def prefix_rows(conn: sqlite3.Connection, sid: str, end: int) -> list[dict]:
    return [
        {
            "id": r["id"],
            "type": r["type"],
            "seq": r["seq"],
            "time_created": r["time_created"],
            "data": json.loads(r["data"]),
        }
        for r in conn.execute(
            "SELECT * FROM session_message WHERE session_id=? AND seq<=? ORDER BY seq",
            (sid, end),
        )
    ]


def validate_prefix(
    conn: sqlite3.Connection, sid: str, expected: dict | None = None
) -> dict | None:
    if not conn.in_transaction:
        conn.execute("BEGIN")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(session_v2)")}
    if "metadata" not in columns:
        if expected is not None:
            raise ValueError("OpenCode committed prefix metadata is missing")
        return None
    row = conn.execute("SELECT metadata FROM session_v2 WHERE id=?", (sid,)).fetchone()
    if row is None:
        raise ValueError(f"OpenCode session {sid} is missing")
    try:
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        proof = metadata.get(PROOF_KEY)
        first = conn.execute(
            "SELECT data FROM session_message WHERE session_id=? ORDER BY seq LIMIT 1",
            (sid,),
        ).fetchone()
        marker = (
            (json.loads(first["data"]).get("metadata") or {}).get(MARKER_KEY)
            if first
            else None
        )
        if proof is None:
            if marker or expected:
                raise ValueError("retained-history proof is missing")
            return None
        if (
            not isinstance(proof, dict)
            or type(proof.get("version")) is not int
            or proof.get("version") != 1
            or proof.get("native_version") != "2.0.21"
        ):
            raise ValueError("unsupported retained-history proof")
        end = proof["prefix_end"]
        if not isinstance(end, int) or isinstance(end, bool) or end < 1:
            raise ValueError("invalid retained-history prefix")
        if marker != proof["binding_id"]:
            raise ValueError("retained-history binding marker differs")
        units = proof["units"]
        if not isinstance(units, list) or not units or units[-1]["seq"] != end:
            raise ValueError("invalid historical turn boundaries")
        previous = 0
        for index, unit in enumerate(units):
            if (
                type(unit["index"]) is not int
                or unit["index"] != index
                or type(unit["seq"]) is not int
                or not previous < unit["seq"] <= end
            ):
                raise ValueError("invalid historical turn order")
            if not isinstance(unit["echo"], bool):
                raise ValueError("invalid historical echo provenance")
            previous = unit["seq"]
        if digest(prefix_rows(conn, sid, end)) != proof["prefix_digest"]:
            raise ValueError("retained-history prefix changed")
        if expected is not None and (
            expected["fresh_id"] != sid or digest(expected["proof"]) != digest(proof)
        ):
            raise ValueError(
                "native retained-history proof differs from the committed pair binding"
            )
        return proof
    except (
        KeyError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"OpenCode retained-history admission failed: {exc}. Native compaction, revert or manual "
            "history edits may require reconciliation recovery; all history was preserved."
        ) from exc


def validate_pair_prefix(conn: sqlite3.Connection, session: Any) -> dict | None:
    sid = session.native_id("opencode")
    expected = getattr(session, "opencode_reconciliation", None)
    state_db = getattr(session, "state_db", None)
    if expected is not None and state_db is None:
        raise ValueError("OpenCode committed binding has no authoritative StateStore")
    if state_db is not None:
        with closing(
            sqlite3.connect(f"{Path(state_db).resolve().as_uri()}?mode=ro", uri=True)
        ) as state:
            state.execute("BEGIN")
            row = state.execute(
                "SELECT native_session_ids FROM sessions WHERE tandem_id=?",
                (session.tandem_id,),
            ).fetchone()
            record = state.execute(
                "SELECT phase,record FROM opencode_reconciliations WHERE tandem_id=?",
                (session.tandem_id,),
            ).fetchone()
        if row is None or json.loads(row[0]).get("opencode") != sid:
            raise ValueError("OpenCode pair binding changed; reload the paired session")
        current = (
            {**json.loads(record[1]), "phase": record[0]}
            if record is not None and record[0] == "committed"
            else None
        )
        if digest(current) != digest(expected):
            raise ValueError(
                "OpenCode reconciliation changed; reload the paired session"
            )
    proof = validate_prefix(conn, sid, expected)
    if proof and (expected is None or expected.get("phase") != "committed"):
        raise ValueError(
            "OpenCode imported history has no committed pair binding; resume its migration"
        )
    return proof
