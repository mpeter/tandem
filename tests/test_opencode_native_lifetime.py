"""Parent lifetime descriptors must not leak through unrelated fork children."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from tandem.opencode_migration import migration_lease
from tandem.state import StateStore


def _running(pid):
    path = Path(f"/proc/{pid}/stat")
    try:
        return path.read_text().split(")", 1)[1].strip().split()[0] != "Z"
    except FileNotFoundError:
        return False


def test_unrelated_fork_cannot_keep_native_lifeline_or_pair_lease(tmp_path):
    ready = tmp_path / "ready.json"
    script = """
import json,os,sys,time
from pathlib import Path
from tandem.state import StateStore
from tandem.opencode_migration import migration_lease
from tandem.opencode_native_snapshot import start_native
root=Path(sys.argv[1])
with StateStore(root/'state.db') as store, migration_lease(store,'pair') as lease:
 proc,line,native=start_native([sys.executable,'-c','import time; time.sleep(60)'],root,os.environ.copy(),lease)
 unrelated=os.fork()
 if unrelated==0:
  null=os.open(os.devnull,os.O_RDWR)
  os.dup2(null,1);os.dup2(null,2);os.close(null)
  while True:time.sleep(0.1)
 (root/'ready.json').write_text(json.dumps({'native':native,'unrelated':unrelated}))
 while True:time.sleep(0.1)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    observed = None
    try:
        deadline = time.monotonic() + 10
        while not ready.exists():
            assert proc.poll() is None, proc.communicate()
            assert time.monotonic() < deadline
            time.sleep(0.05)
        observed = json.loads(ready.read_text())
        assert _running(observed["native"]) and _running(observed["unrelated"])
        proc.kill()
        proc.wait(timeout=5)
        assert proc.returncode == -signal.SIGKILL
        deadline = time.monotonic() + 5
        while _running(observed["native"]) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _running(observed["native"])
        assert _running(observed["unrelated"])
        with StateStore(tmp_path / "state.db") as store:
            while True:
                try:
                    with migration_lease(store, "pair"):
                        break
                except ValueError:
                    assert time.monotonic() < deadline
                    time.sleep(0.05)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if observed and _running(observed["unrelated"]):
            os.kill(observed["unrelated"], signal.SIGTERM)
        for stream in (proc.stdout, proc.stderr):
            stream.close()
