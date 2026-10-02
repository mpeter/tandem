"""Own one native process group until its caller closes the lifeline pipe."""

import os
import selectors
import signal
import subprocess
import sys
import time
from types import FrameType


POLL_INTERVAL = 0.05
TERM_GRACE = 0.3
KILL_SETTLE = 0.1


def _exited(pid: int) -> bool:
    # Keep the leader's PID reserved until the final group signal has been sent.
    return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _signal_group(pid: int, signum: int) -> None:
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def _cleanup(child: subprocess.Popen[bytes]) -> int:
    _signal_group(child.pid, signal.SIGTERM)
    deadline = time.monotonic() + TERM_GRACE
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL)
    # The leader may already be a zombie, but its children can still be running.
    _signal_group(child.pid, signal.SIGKILL)
    while not _exited(child.pid):
        time.sleep(POLL_INTERVAL)
    time.sleep(KILL_SETTLE)
    return child.wait()


def supervise(lifeline_fd: int, lease_fd: int, ready_fd: int, argv: list[str]) -> int:
    """Hold the inherited lease through cleanup; announce only the native PID."""
    stopping = False
    child: subprocess.Popen[bytes] | None = None

    def request_stop(signum: int, frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True

    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, request_stop)
    os.set_blocking(lifeline_fd, False)

    def parent_gone() -> bool:
        try:
            return os.read(lifeline_fd, 4096) == b""
        except BlockingIOError:
            return False

    try:
        with selectors.DefaultSelector() as selector:
            selector.register(lifeline_fd, selectors.EVENT_READ)
            if stopping or parent_gone():
                return 125
            child = subprocess.Popen(argv, start_new_session=True, close_fds=True)
            os.write(ready_fd, f"{child.pid}\n".encode("ascii"))
            os.close(ready_fd)
            ready_fd = -1
            while not stopping and not _exited(child.pid):
                if selector.select(POLL_INTERVAL) and parent_gone():
                    break
    finally:
        try:
            if child is not None:
                result = _cleanup(child)
        finally:
            os.close(lifeline_fd)
            if ready_fd >= 0:
                os.close(ready_fd)
            if lease_fd >= 0:
                os.close(lease_fd)
    return result


def main() -> int:
    if len(sys.argv) < 5:
        print("usage: worker lifeline_fd lease_fd ready_fd command [args ...]", file=sys.stderr)
        return 2
    lifeline_fd, lease_fd, ready_fd = (int(value) for value in sys.argv[1:4])
    returncode = supervise(lifeline_fd, lease_fd, ready_fd, sys.argv[4:])
    if returncode < 0:
        signum = -returncode
        if signum != signal.SIGKILL:
            signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
