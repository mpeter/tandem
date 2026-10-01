"""Use the pinned native migrator, fork and import for retained snapshots."""

from __future__ import annotations

import base64
import http.client
import json
import os
import secrets
import select
import sys
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .harness.opencode2 import database

NATIVE_VERSION = "2.0.21"


_LIFELINES: set[int] = set()


def _close_forked_lifelines() -> None:
    for fd in tuple(_LIFELINES):
        try:
            os.close(fd)
        except OSError:
            pass
    _LIFELINES.clear()


os.register_at_fork(after_in_child=_close_forked_lifelines)


def isolated_env(root: Path, db: Path) -> dict[str, str]:
    env = {
        "PATH": "/usr/bin:/bin",
        "OPENCODE_DB": str(db),
        "OPENCODE_TEST_HOME": str(root / "home"),
        "HOME": str(root / "home"),
        "XDG_DATA_HOME": str(root / "data"),
        "XDG_CONFIG_HOME": str(root / "config"),
        "XDG_CACHE_HOME": str(root / "cache"),
        "XDG_STATE_HOME": str(root / "state"),
        "TMPDIR": str(root / "tmp"),
        "OPENCODE_CONFIG_DIR": str(root / "config/opencode"),
        "OPENCODE_CONFIG_PROJECT_DISABLE": "1",
        "OPENCODE_DISABLE_MODELS_FETCH": "true",
        "OPENCODE_DISABLE_FILEWATCHER": "true",
        "OPENCODE_CONFIG_CONTENT": json.dumps(
            {
                "plugin": [],
                "disabled_providers": [
                    "openai",
                    "anthropic",
                    "google",
                    "google-vertex",
                ],
                "permission": "deny",
            }
        ),
    }
    for key in (
        "HOME",
        "XDG_DATA_HOME",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_STATE_HOME",
        "TMPDIR",
        "OPENCODE_CONFIG_DIR",
    ):
        Path(env[key]).mkdir(parents=True, exist_ok=True, mode=0o700)
    return env


def import_payload(
    binary: str, db: Path, payload: dict, root: Path, lease_fd: int = -1
) -> None:
    path = root / "import.json"
    path.write_text(json.dumps(payload))
    path.chmod(0o600)
    args = [
        binary,
        "session",
        "import",
        "--standalone",
        "--directory",
        payload["info"]["location"]["directory"],
        str(path),
    ]
    proc, lifeline, _ = start_native(
        args,
        root,
        isolated_env(root, db),
        lease_fd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        proc.communicate(timeout=120)
        if proc.returncode:
            raise ValueError(
                f"Native OpenCode snapshot import failed (exit {proc.returncode})"
            )
    finally:
        try:
            close_native(proc, lifeline)
        finally:
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()


def start_native(
    args: list[str], root: Path, env: dict[str, str], lease_fd: int = -1, **stdio
) -> tuple[subprocess.Popen, int, int]:
    """The guardian owns native lifetime even when this Python process dies."""
    read_fd, write_fd = os.pipe()
    _LIFELINES.add(write_fd)
    ready_read, ready_write = os.pipe()
    guardian = Path(__file__).with_name("opencode_native_worker.py")
    inherited = (read_fd, ready_write, *((lease_fd,) if lease_fd >= 0 else ()))
    proc = None
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(guardian),
                str(read_fd),
                str(lease_fd),
                str(ready_write),
                *args,
            ],
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
            pass_fds=inherited,
            **stdio,
        )
        os.close(read_fd)
        read_fd = -1
        os.close(ready_write)
        ready_write = -1
        if not select.select([ready_read], [], [], 10)[0]:
            raise ValueError("Native guardian did not report readiness")
        raw = os.read(ready_read, 128).decode().strip()
        if not raw.isdecimal():
            raise ValueError("Native guardian stopped before startup")
        return proc, write_fd, int(raw)
    except BaseException:
        if proc is not None:
            try:
                close_native(proc, write_fd)
            finally:
                for stream in (proc.stdout, proc.stderr):
                    if stream is not None:
                        stream.close()
        else:
            _LIFELINES.discard(write_fd)
            os.close(write_fd)
        raise
    finally:
        for descriptor in (read_fd, ready_read, ready_write):
            if descriptor >= 0:
                os.close(descriptor)


def close_native(proc: subprocess.Popen, lifeline: int) -> None:
    _LIFELINES.discard(lifeline)
    os.close(lifeline)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            raise RuntimeError("Native guardian failed to finish owned process cleanup")


class NativeServer:
    def __init__(self, binary: str, db: Path, root: Path, lease_fd: int = -1):
        self.binary, self.db, self.root = binary, db, root
        self.lease_fd = lease_fd
        self.native_pid: int | None = None
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.password = secrets.token_urlsafe(32)

    def request(self, method: str, route: str, body: dict | None = None) -> dict:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            auth = base64.b64encode(f"opencode:{self.password}".encode()).decode()
            conn.request(
                method,
                route,
                json.dumps(body) if body is not None else None,
                {"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
            )
            response = conn.getresponse()
            data = response.read()
            if response.status == 503:
                raise OSError("Native migration is initializing")
            if response.status != 200:
                raise ValueError(
                    f"Native OpenCode snapshot operation failed (HTTP {response.status})"
                )
            return json.loads(data)
        finally:
            conn.close()

    @contextmanager
    def running(self) -> Iterator["NativeServer"]:
        log_path = self.root / "native.log"
        with log_path.open("w") as log:
            args = [
                self.binary,
                "serve",
                "--hostname",
                "127.0.0.1",
                "--port",
                str(self.port),
            ]
            proc, lifeline, self.native_pid = start_native(
                args,
                self.root,
                {
                    **isolated_env(self.root, self.db),
                    "OPENCODE_PASSWORD": self.password,
                },
                self.lease_fd,
                stdout=log,
                stderr=log,
            )
            try:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise ValueError(
                            "Native OpenCode snapshot server stopped before migration completed"
                        )
                    try:
                        state = self.request("GET", "/api/experimental/migration/v1")
                        if state.get("status") == "error":
                            raise ValueError(
                                "Native OpenCode snapshot migration failed"
                            )
                        if state.get("status") == "completed":
                            with database(self.db) as conn:
                                row = conn.execute(
                                    "SELECT value FROM kv WHERE key='migration.v1-v2'"
                                ).fetchone()
                                if row and json.loads(row[0]) == {"phase": "completed"}:
                                    break
                    except (OSError, http.client.HTTPException):
                        pass
                    time.sleep(0.05)
                else:
                    raise ValueError("Native OpenCode snapshot migration timed out")
                log.flush()
                if "Skipped V1 migration row" in log_path.read_text():
                    raise ValueError(
                        "Native migration skipped an unsupported legacy row; source preserved"
                    )
                yield self
            finally:
                close_native(proc, lifeline)


def capture_source(db: Path, sid: str) -> dict:
    with database(db) as conn:
        conn.execute("BEGIN")
        snapshot: dict = {
            table: [
                dict(row)
                for row in conn.execute(
                    f"SELECT * FROM {table} WHERE {'id' if table == 'session' else 'session_id'}=? ORDER BY id",
                    (sid,),
                )
            ]
            for table in ("session", "message", "part")
        }
        if len(snapshot["session"]) != 1:
            raise ValueError("Selected OpenCode identity has no retained legacy source")
        snapshot["ddl"] = {
            table: conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()[0]
            for table in ("session", "message", "part")
        }
        project = snapshot["session"][0]["project_id"]
        snapshot["project"] = [
            dict(row)
            for row in conn.execute("SELECT * FROM project WHERE id=?", (project,))
        ]
        return snapshot


def native_oracle(
    binary: str, snapshot: dict, sid: str, root: Path, lease_fd: int = -1
) -> tuple[dict, list[dict], dict]:
    """Copy raw source tables; native code alone constructs the transcript."""
    db = root / "oracle.db"
    bootstrap = {
        "info": {
            "id": "ses_tandem_snapshot_bootstrap",
            "projectID": "tandem-snapshot",
            "agent": "build",
            "location": {"directory": str(root)},
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
    import_payload(binary, db, bootstrap, root, lease_fd)
    with database(db) as conn:
        for row in snapshot["project"]:
            columns = list(row)
            conn.execute(
                f"INSERT OR IGNORE INTO project ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                [row[column] for column in columns],
            )
        for table in ("session", "message", "part"):
            conn.execute(snapshot["ddl"][table])
            for row in snapshot[table]:
                columns = list(row)
                conn.execute(
                    f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                    [row[column] for column in columns],
                )
    server = NativeServer(binary, db, root, lease_fd)
    with server.running():
        with database(db) as conn:
            old_info = imported_info(conn, sid)
            oracle = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM session_message WHERE session_id=? ORDER BY seq",
                    (sid,),
                )
            ]
        fresh = server.request("POST", f"/api/session/{sid}/fork", {})["data"]["id"]
        payload = server.request("GET", f"/api/experimental/session/{fresh}/export")[
            "data"
        ]
        with database(db) as conn:
            copied = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM session_message WHERE session_id=? ORDER BY seq",
                    (fresh,),
                )
            ]
        fields = ("type", "seq", "time_created", "data")
        if [{key: row[key] for key in fields} for row in copied] != [
            {key: row[key] for key in fields} for row in oracle
        ]:
            raise ValueError(
                "Native fork filtered legacy history; this history requires additional recovery"
            )
        if len(payload["messages"]) != len(oracle):
            raise ValueError("Native export filtered legacy history")
        return payload, oracle, old_info


IMPORT_INFO_FIELDS = (
    "id",
    "project_id",
    "directory",
    "path",
    "agent",
    "model",
    "permission",
    "metadata",
    "title",
    "version",
    "time_created",
    "time_idle",
    "idle_outcome",
    "cost",
    "tokens_input",
    "tokens_output",
    "tokens_reasoning",
    "tokens_cache_read",
    "tokens_cache_write",
    "parent_id",
    "fork_session_id",
    "fork_boundary",
    "revert",
    "workspace_id",
)


def imported_info(conn, sid: str) -> dict:
    row = conn.execute("SELECT * FROM session_v2 WHERE id=?", (sid,)).fetchone()
    if row is None:
        raise ValueError("Native import did not persist session Info")
    info = {key: row[key] for key in IMPORT_INFO_FIELDS}
    for key in ("model", "permission", "metadata", "fork_boundary", "revert"):
        if info[key] is not None:
            info[key] = json.loads(info[key])
    return info


def normalized_import_info(
    binary: str, payload: dict, root: Path, lease_fd: int = -1
) -> dict:
    """Ask the pinned native importer for supported transfer normalization."""
    db = root / "normalized.db"
    import_payload(binary, db, payload, root, lease_fd)
    with database(db) as conn:
        return imported_info(conn, payload["info"]["id"])
