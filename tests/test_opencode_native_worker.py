import os
import selectors
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from tandem import opencode_native_worker


WORKER = Path(opencode_native_worker.__file__)


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("timed out waiting for process state")


def running(pid):
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True, text=True, check=False,
    )
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")


@contextmanager
def launch(code, *, parent_closed=False, lease=True):
    lifeline_read, lifeline_write = os.pipe()
    lease_read, lease_write = os.pipe()
    ready_read, ready_write = os.pipe()
    if parent_closed:
        os.close(lifeline_write)
        lifeline_write = -1
    inherited = (lifeline_read, ready_write, lease_write) if lease else (lifeline_read, ready_write)
    worker = subprocess.Popen(
        [sys.executable, str(WORKER), str(lifeline_read),
         str(lease_write if lease else -1), str(ready_write), sys.executable, "-c", code],
        pass_fds=inherited, start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    for fd in (lifeline_read, lease_write, ready_write):
        os.close(fd)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(ready_read, selectors.EVENT_READ)
            assert selector.select(5), "worker did not send readiness or close pipe"
        announced = os.read(ready_read, 128)
        native_pid = int(announced) if announced else None
        yield worker, lifeline_write, lease_read, native_pid
    finally:
        if lifeline_write >= 0:
            os.close(lifeline_write)
        if worker.poll() is None:
            worker.terminate()
        worker.communicate(timeout=5)
        os.close(ready_read)
        os.close(lease_read)


def test_closed_lifeline_never_starts_native(tmp_path):
    marker = tmp_path / "started"
    code = f"from pathlib import Path; Path({str(marker)!r}).touch()"
    with launch(code, parent_closed=True) as (worker, _, lease, native_pid):
        stdout, stderr = worker.communicate(timeout=5)
        assert worker.returncode == 125
        assert native_pid is None
        assert not marker.exists()
        assert (stdout, stderr) == (b"", b"")
        assert os.read(lease, 1) == b""


@pytest.mark.parametrize("lease", [True, False])
def test_output_and_native_exit_code_are_forwarded(lease):
    code = "import sys; print('native output'); print('native error', file=sys.stderr); sys.exit(7)"
    with launch(code, lease=lease) as (worker, _, _, native_pid):
        stdout, stderr = worker.communicate(timeout=5)
        assert native_pid is not None
        assert worker.returncode == 7
        assert stdout == b"native output\n"
        assert stderr == b"native error\n"


def native_family_code(marker, *, ignore_term=False, early_exit=False):
    child_code = (
        "import signal,time; from pathlib import Path; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
        + f"Path({str(marker)!r}).write_text(str(__import__('os').getpid())); time.sleep(60)"
    )
    return (
        "import signal,subprocess,sys,time; from pathlib import Path\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
        + f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        + f"while not Path({str(marker)!r}).exists(): time.sleep(0.01)\n"
        + ("sys.exit(9)\n" if early_exit else "time.sleep(60)\n")
    )


@pytest.mark.parametrize("trigger", ["eof", signal.SIGTERM, signal.SIGHUP])
@pytest.mark.parametrize("ignore_term", [False, True])
def test_parent_departure_cleans_native_and_grandchild(tmp_path, trigger, ignore_term):
    marker = tmp_path / "grandchild"
    with launch(native_family_code(marker, ignore_term=ignore_term)) as (worker, lifeline, lease, pid):
        wait_for(marker.exists)
        grandchild = int(marker.read_text())
        assert running(pid) and running(grandchild)
        if trigger == "eof":
            # dup2 replaces the pipe endpoint without invalidating fixture ownership.
            with open(os.devnull, "wb") as sink:
                os.dup2(sink.fileno(), lifeline)
        else:
            os.kill(worker.pid, trigger)
        os.set_blocking(lease, False)
        with pytest.raises(BlockingIOError):
            os.read(lease, 1)
        worker.communicate(timeout=5)
        assert worker.returncode == (-signal.SIGKILL if ignore_term else -signal.SIGTERM)
        wait_for(lambda: not running(pid) and not running(grandchild))
        assert os.read(lease, 1) == b""


def test_early_leader_exit_still_cleans_lingering_child(tmp_path):
    marker = tmp_path / "grandchild"
    code = native_family_code(marker, ignore_term=True, early_exit=True)
    with launch(code) as (worker, _, lease, pid):
        wait_for(marker.exists)
        grandchild = int(marker.read_text())
        worker.communicate(timeout=5)
        assert worker.returncode == 9
        wait_for(lambda: not running(pid) and not running(grandchild))
        assert os.read(lease, 1) == b""


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGKILL])
def test_native_signal_exit_is_forwarded(signum):
    code = f"import os; os.kill(os.getpid(), {int(signum)})"
    with launch(code) as (worker, _, _, native_pid):
        stdout, stderr = worker.communicate(timeout=5)
        assert native_pid is not None
        assert worker.returncode == -signum
        assert (stdout, stderr) == (b"", b"")
