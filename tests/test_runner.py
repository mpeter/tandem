"""InteractiveRunner: user-configured [harness] args land in the spawned argv."""

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from tandem import paths, runner
from tandem.runner import FlipMonitor, wait_until_safe


class _Sink:
    def handle(self, line, ctx, cursor): ...

    def close(self): ...


def _null_sink(store, session, source, target):
    return _Sink()


def _run_capturing_argv(env, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        runner, "run_in_pty",
        lambda argv, cwd=None, **kw: calls.update(argv=argv) or 0,
    )
    code = runner.InteractiveRunner(
        env.session, lambda store, session, source: _Sink()).run()
    assert code == 0
    return calls["argv"]


def test_claude_resume_gets_args_before_hook_extras(env_factory, monkeypatch):
    env = env_factory(active="claude")
    (paths.tandem_home() / "config.toml").write_text(
        '[claude]\nargs = ["--dangerously-skip-permissions"]\n'
    )
    argv = _run_capturing_argv(env, monkeypatch)
    i = argv.index("--resume")
    assert argv[i + 1] == env.session.native_id("claude")
    assert argv[i + 2] == "--dangerously-skip-permissions"
    assert argv[i + 3] == "--settings"  # hook extras immediately after


def test_claude_fresh_launch_gets_args(env_factory, monkeypatch):
    env = env_factory(active="claude")
    env.claude_shadow.unlink()  # no transcript -> fresh --session-id launch
    (paths.tandem_home() / "config.toml").write_text(
        '[claude]\nargs = ["--dangerously-skip-permissions"]\n'
    )
    argv = _run_capturing_argv(env, monkeypatch)
    i = argv.index("--session-id")
    assert argv[i + 2] == "--dangerously-skip-permissions"


def test_codex_gets_its_own_args(env_factory, monkeypatch):
    env = env_factory(active="codex")
    (paths.tandem_home() / "config.toml").write_text(
        '[codex]\nargs = ["--dangerously-bypass-approvals-and-sandbox"]\n\n'
        '[claude]\nargs = ["--should-not-appear"]\n'
    )
    argv = _run_capturing_argv(env, monkeypatch)
    i = argv.index("resume")
    assert argv[i + 1] == env.session.native_id("codex")
    assert argv[i + 2] == "--dangerously-bypass-approvals-and-sandbox"
    assert "--should-not-appear" not in argv


def test_codex_fresh_mint_gets_args_after_bare_binary(env_factory, monkeypatch):
    # codex minting its own session id launches as bare `codex`, so the
    # configured args are the first tokens after the binary.
    env = env_factory(active="codex")
    session = env.store.create_session(
        env.cwd, "codex", ["claude", "codex"],
        {"claude": env.session.native_id("claude"), "codex": None},
    )
    (paths.tandem_home() / "config.toml").write_text(
        '[codex]\nargs = ["--dangerously-bypass-approvals-and-sandbox"]\n'
    )
    calls = {}
    monkeypatch.setattr(
        runner, "run_in_pty",
        lambda argv, cwd=None, **kw: calls.update(argv=argv) or 0,
    )
    code = runner.InteractiveRunner(
        session, lambda store, session, source: _Sink()).run()
    assert code == 0
    argv = calls["argv"]
    assert argv[:2] == ["codex", "--dangerously-bypass-approvals-and-sandbox"]


def test_oneoff_argv_never_gains_args(tmp_path, monkeypatch):
    # `tandem run` and doctor probes build argv via oneoff_argv; a configured
    # args list must never leak in (the append lives in InteractiveRunner).
    from tandem.harness import get_adapter

    home = tmp_path / ".tandem"
    home.mkdir()
    monkeypatch.setenv("TANDEM_HOME", str(home))
    (home / "config.toml").write_text(
        '[claude]\nargs = ["--should-not-appear"]\n\n'
        '[codex]\nargs = ["--should-not-appear"]\n'
    )
    assert get_adapter("claude").oneoff_argv("sid", "task") == [
        "claude", "--resume", "sid", "-p", "task"]
    assert get_adapter("codex").oneoff_argv("sid", "task") == [
        "codex", "exec", "--skip-git-repo-check", "resume", "sid", "task"]


def test_no_config_leaves_argv_unchanged(env_factory, monkeypatch):
    env = env_factory(active="claude")
    argv = _run_capturing_argv(env, monkeypatch)
    assert argv[:3] == ["claude", "--resume", env.session.native_id("claude")]
    assert argv[3] == "--settings"


def _touch(path, mtime):
    path.write_text("x")
    os.utime(path, (mtime, mtime))


@pytest.mark.parametrize("marker_wired", [False, True])
def test_wait_idle_when_sentinel_newer_than_transcript(tmp_path, marker_wired):
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    now = time.time()
    _touch(t, now - 10)
    _touch(s, now - 5)   # marker closed the last turn
    assert wait_until_safe(t, s, cancelled=lambda: False,
                           marker_wired=marker_wired) is True


def test_wait_idle_when_no_files(tmp_path):
    assert (
        wait_until_safe(tmp_path / "none", tmp_path / "none2",
                        cancelled=lambda: False)
        is True
    )


def test_wait_quiescence_fallback(tmp_path):
    # The marker-less shape, built honestly: tandem refused to clobber a
    # user-configured codex notify, so the hook was never wired and the
    # sentinel file never appears at all (mtime 0 < transcript, forever).
    # Quiescence is the only exit, which is exactly what the fallback is for.
    t, s = tmp_path / "t.jsonl", tmp_path / "never-touched.turn"
    _touch(t, time.time() - 3)   # transcript quiet for 3s, no marker since
    assert not s.exists()
    assert (
        wait_until_safe(t, s, cancelled=lambda: False, quiesce=2.0,
                        marker_wired=False) is True
    )


def test_wait_missing_marker_wired_does_not_infer_idle(tmp_path):
    # No transcript or marker yet can also mean startup or pending approval.
    polls = 0

    def cancel():
        nonlocal polls
        polls += 1
        return polls > 2

    assert wait_until_safe(tmp_path / "none", tmp_path / "none2",
                           cancelled=cancel, marker_wired=True, poll=0) is False
    assert polls == 3


def test_wait_wired_long_tool_or_approval_wait_requires_marker(tmp_path):
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(t, time.time() - 3600)   # unfinished turn silent beyond the old valve
    _touch(s, time.time() - 7200)   # only the previous turn completed
    done = threading.Event()
    cancel = threading.Event()
    result = {}

    def waiter():
        result["ok"] = wait_until_safe(t, s, cancelled=cancel.is_set,
                                       quiesce=0.1, poll=0.05,
                                       marker_wired=True)
        done.set()

    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    try:
        assert not done.wait(timeout=0.3)  # even an explicit timeout is ignored
        _touch(s, time.time())            # completion, not silence, releases
        assert done.wait(timeout=2)
        assert result["ok"] is True
    finally:
        cancel.set()
        thread.join(timeout=2)


def test_wait_wired_default_ignores_quiescence(tmp_path):
    # The mode's whole point: the same 3s-quiet mid-turn transcript that the
    # marker-less mode calls idle must still be mid-turn when the marker is
    # wired, on default settings.
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(t, time.time() - 3)
    _touch(s, time.time() - 30)
    assert wait_until_safe(t, s, cancelled=lambda: False,
                           marker_wired=False) is True
    cancel, done = threading.Event(), threading.Event()
    result = {}

    def waiter():
        result["ok"] = wait_until_safe(t, s, cancelled=cancel.is_set,
                                       poll=0.05, marker_wired=True)
        done.set()

    threading.Thread(target=waiter, daemon=True).start()
    assert not done.wait(timeout=0.4)   # still waiting for completion
    cancel.set()                        # cancellation remains available
    assert done.wait(timeout=2)
    assert result["ok"] is False


def test_wait_wired_marker_touch_releases_promptly(tmp_path):
    # The marker path is unchanged by the mode: the touch is the trigger.
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(t, time.time())
    _touch(s, time.time() - 30)
    done = threading.Event()
    result = {}

    def waiter():
        result["ok"] = wait_until_safe(t, s, cancelled=lambda: False,
                                       poll=0.05, marker_wired=True)
        done.set()

    threading.Thread(target=waiter, daemon=True).start()
    time.sleep(0.2)
    assert not done.is_set()
    _touch(s, time.time() + 1)      # marker fires
    assert done.wait(timeout=2)
    assert result["ok"] is True


def test_wait_blocks_midturn_then_marker_releases(tmp_path):
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(t, time.time())          # a line just landed: turn in flight
    _touch(s, time.time() - 30)
    done = threading.Event()
    result = {}

    def waiter():
        result["ok"] = wait_until_safe(t, s, cancelled=lambda: False,
                                       quiesce=30.0, poll=0.05)
        done.set()

    threading.Thread(target=waiter, daemon=True).start()
    time.sleep(0.2)
    assert not done.is_set()        # still waiting
    _touch(s, time.time() + 1)      # marker fires
    assert done.wait(timeout=2)
    assert result["ok"] is True


def test_wait_unknown_transcript_wired_is_not_read_as_idle(tmp_path):
    # A previous marker cannot prove completion before rollout discovery.
    s = tmp_path / "s.turn"
    _touch(s, time.time() - 30)
    cancel, done = threading.Event(), threading.Event()
    result = {}

    def waiter():
        result["ok"] = wait_until_safe(None, s, cancelled=cancel.is_set,
                                       quiesce=0.1, poll=0.05,
                                       marker_wired=True)
        done.set()

    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    try:
        assert not done.wait(timeout=0.3)
    finally:
        cancel.set()
        thread.join(timeout=2)
    assert done.is_set()
    assert result["ok"] is False


def test_wait_provider_publishing_a_path_restores_normal_rules(tmp_path):
    # The tail thread discovers the rollout mid-wait and publishes it; from
    # that poll on the ordinary completion-marker rule applies.
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(s, time.time() - 30)
    published: list = [None]
    done = threading.Event()
    result = {}

    def waiter():
        result["ok"] = wait_until_safe(None, s, cancelled=lambda: False,
                                       quiesce=30.0, poll=0.05,
                                       marker_wired=True,
                                       provider=lambda: published[0])
        done.set()

    threading.Thread(target=waiter, daemon=True).start()
    assert not done.wait(timeout=0.3)   # unknown transcript: parked
    _touch(t, time.time())              # a turn is in flight
    published[0] = t
    assert not done.wait(timeout=0.3)   # known now, and mid-turn: still parked
    _touch(s, time.time() + 1)          # marker closes the turn
    assert done.wait(timeout=2)
    assert result["ok"] is True


def test_wait_standalone_default_provider_reads_its_argument(tmp_path):
    # Back-compat: with no provider the positional transcript is what every
    # poll reads, exactly as before the provider existed.
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(t, time.time())
    _touch(s, time.time() - 30)
    done = threading.Event()
    result = {}

    def waiter():
        result["ok"] = wait_until_safe(t, s, cancelled=lambda: False,
                                       poll=0.05, marker_wired=True)
        done.set()

    threading.Thread(target=waiter, daemon=True).start()
    assert not done.wait(timeout=0.3)
    _touch(s, time.time() + 1)
    assert done.wait(timeout=2)
    assert result["ok"] is True


def test_monitor_transcript_published_midwait_is_picked_up(tmp_path):
    # The monitor half of the same story: the runner's tail thread assigns
    # monitor.transcript once codex's rollout appears, and the live wait sees
    # it on its next poll.
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(t, time.time())
    _touch(s, time.time() - 30)
    control = _StubControl()
    m = FlipMonitor(control, [b"\x04"], transcript=None, sentinel=s,
                    marker_wired=True, quiesce=30.0, poll=0.05)
    m.start()
    m.flip_pressed()
    time.sleep(0.3)
    assert m.flip_requested is False     # unknown transcript: no instant fire
    m.transcript = t                     # tail thread discovered the rollout
    time.sleep(0.2)
    assert m.flip_requested is False     # mid-turn under the normal rules
    _touch(s, time.time() + 1)           # marker fires
    deadline = time.time() + 3
    while not m.flip_requested and time.time() < deadline:
        time.sleep(0.05)
    m.stop()
    assert m.flip_requested is True
    assert control.calls == [[b"\x04"]]


def test_runner_publishes_the_discovered_codex_rollout_to_the_monitor(
    env_factory, monkeypatch
):
    # Fresh codex (no session id yet): the tail thread finds the rollout and
    # hands it to the monitor, so an armed flip stops being judged blind.
    env = env_factory(active="codex")
    session = env.store.create_session(
        env.cwd, "codex", ["claude", "codex"],
        {"claude": env.session.native_id("claude"), "codex": None},
    )
    rollout = env.codex_shadow
    monkeypatch.setattr(
        runner, "await_codex_rollout",
        lambda cwd, after, timeout=None, **kwargs: rollout,
    )
    made = {}
    real = runner.FlipMonitor

    def capture(*a, **kw):
        made["monitor"] = real(*a, **kw)
        return made["monitor"]

    monkeypatch.setattr(runner, "FlipMonitor", capture)

    def fake_run_in_pty(argv, cwd=None, frame=None, control=None, child=None, env=None):
        deadline = time.time() + 3
        while made["monitor"].transcript is None and time.time() < deadline:
            time.sleep(0.02)
        return 0

    monkeypatch.setattr(runner, "run_in_pty", fake_run_in_pty)
    runner.InteractiveRunner(session, lambda st, se, so, tg: _Sink()).run()
    assert made["monitor"].transcript == rollout


def test_wait_cancelled(tmp_path):
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(t, time.time())
    _touch(s, time.time() - 30)
    assert (
        wait_until_safe(t, s, cancelled=lambda: True, quiesce=30.0) is False
    )


class _StubControl:
    def __init__(self):
        self.calls = []

    def terminate(self, soft, **kw):
        self.calls.append(soft)
        return "soft"


def test_monitor_arm_wait_terminate(tmp_path):
    control = _StubControl()
    m = FlipMonitor(control, [b"\x04"], transcript=None,
                    sentinel=tmp_path / "s.turn")
    m.start()
    assert m.armed() is False
    m.flip_pressed()                 # idle (no files) -> fires immediately
    deadline = time.time() + 3
    while not m.flip_requested and time.time() < deadline:
        time.sleep(0.05)
    m.stop()
    assert m.flip_requested is True
    assert m.how == "soft"
    assert control.calls == [[b"\x04"]]


class _OrderingControl:
    """Records the ladder against the fire on one list: the hook appends
    "fired", `terminate` appends "ladder"."""

    def __init__(self, order):
        self.order = order

    def terminate(self, soft, **kw):
        self.order.append("ladder")
        return "soft"


def _fire_and_join(control, hook, tmp_path):
    """Drive one decided flip through a real FlipMonitor with `hook` wired
    through the constructor, and join the thread so the ladder has finished
    before anything is asserted. `status_probe` makes the boundary wait
    return at once; the loop is the same settle `test_monitor_arm_wait_
    terminate` uses, because `stop()` landing before the thread wakes from
    its arm would cancel the flip instead of firing it."""
    m = FlipMonitor(control, [b"\x04"], transcript=None,
                    sentinel=tmp_path / "s.turn",
                    status_probe=lambda: "waiting",
                    on_flip_decided=hook)
    m.start()
    m.flip_pressed()
    deadline = time.time() + 3
    while not m.flip_requested and time.time() < deadline:
        time.sleep(0.05)
    m.stop()
    return m


def test_monitor_fires_the_flip_hook_before_the_ladder(tmp_path):
    # The hook is where the incoming harness is spawned, and the whole point
    # of firing it from here is that the boot overlaps the outgoing harness's
    # teardown. Run after `control.terminate` it would still spawn, still
    # hand over, still pass every runner test — and serialize precisely what
    # this pipelines. So the order is the contract.
    order = []
    m = _fire_and_join(_OrderingControl(order), lambda: order.append("fired"),
                       tmp_path)
    assert order == ["fired", "ladder"]
    assert m.flip_requested is True
    assert m.how == "soft"          # the ladder's answer still lands


def test_monitor_survives_a_raising_flip_hook(tmp_path):
    # A hook that raises costs a cold flip and nothing else. Without the
    # swallow the exception kills this thread mid-flip: `flip_requested` is
    # already True, so the runner still reports a flip and the tests above
    # still pass, but the ladder never runs and the harness the user just
    # pressed Ctrl-] in is never terminated. The ladder's survival is the
    # assertion, not a warning.
    order = []

    def boom():
        raise OSError("spawn failed")

    m = _fire_and_join(_OrderingControl(order), boom, tmp_path)
    assert order == ["ladder"]
    assert m.flip_requested is True
    assert m.how == "soft"


def test_monitor_toggle_cancels(tmp_path):
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(t, time.time())           # mid-turn: monitor will block
    _touch(s, time.time() - 30)
    control = _StubControl()
    m = FlipMonitor(control, [b"\x04"], transcript=t, sentinel=s,
                    marker_wired=True, poll=0.05)
    m.start()
    m.flip_pressed()
    time.sleep(0.1)
    assert m.armed() is True
    m.flip_pressed()                 # toggle: cancel
    time.sleep(0.3)
    assert m.armed() is False
    assert m.flip_requested is False
    m.stop()
    assert control.calls == []


def test_monitor_stop_unblocks_cleanly(tmp_path):
    m = FlipMonitor(_StubControl(), [b"\x04"], transcript=None,
                    sentinel=tmp_path / "s.turn")
    m.start()
    m.stop()                         # never armed: must not hang or fire
    assert m.flip_requested is False


def test_monitor_stop_cancels_armed_marker_wait(tmp_path):
    # Manual harness exit stops the monitor without terminating it again.
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(t, time.time() - 3600)
    control = _StubControl()
    m = FlipMonitor(control, [b"\x04"], transcript=t, sentinel=s,
                    marker_wired=True, poll=0.05)
    m.start()
    m.flip_pressed()
    try:
        time.sleep(0.1)
        assert m.armed() is True
        assert m.flip_requested is False
    finally:
        m.stop()
    assert not m._thread.is_alive()
    assert m.flip_requested is False
    assert control.calls == []


def test_monitor_quiesce_default_and_override(tmp_path):
    s = tmp_path / "s.turn"
    unwired = FlipMonitor(_StubControl(), [], None, s)
    wired = FlipMonitor(_StubControl(), [], None, s, marker_wired=True)
    override = FlipMonitor(_StubControl(), [], None, s, marker_wired=True,
                           quiesce=1.5)
    assert unwired.quiesce == 2.0        # marker-less fallback
    assert wired.quiesce == 2.0          # ignored when marker is wired
    assert override.quiesce == 1.5       # explicit injection wins


def test_monitor_passes_mode_through_to_wait(tmp_path, monkeypatch):
    seen = {}

    def fake_wait(transcript, sentinel, cancelled, quiesce=None, poll=0.2,
                  marker_wired=False, provider=None, status_probe=None, turn_boundary=None):
        seen.update(quiesce=quiesce, poll=poll, marker_wired=marker_wired,
                    provider_reads=provider(), turn_boundary=turn_boundary)
        return True

    monkeypatch.setattr(runner, "wait_until_safe", fake_wait)
    control = _StubControl()
    m = FlipMonitor(control, [b"\x04"], transcript=None,
                    sentinel=tmp_path / "s.turn", marker_wired=True,
                    poll=0.05)
    m.start()
    m.flip_pressed()
    deadline = time.time() + 3
    while not m.flip_requested and time.time() < deadline:
        time.sleep(0.05)
    m.stop()
    assert m.flip_requested is True
    assert seen == {"quiesce": 2.0, "poll": 0.05, "marker_wired": True,
                    "provider_reads": None, "turn_boundary": None}


# -- the runner wiring the frame ------------------------------------------


class _DeadChild:
    """Stands in for the pty child a real run_in_pty attaches; the
    termination ladder reads it as already gone ("dead") without waiting out
    the attach timeout."""

    def isalive(self):
        return False


def test_runner_passes_frame_and_control(env_factory, monkeypatch):
    env = env_factory(active="claude")
    seen = {}

    def fake_run_in_pty(argv, cwd=None, frame=None, control=None, child=None):
        seen.update(frame=frame, control=control)
        return 0

    monkeypatch.setattr(runner, "run_in_pty", fake_run_in_pty)
    r = runner.InteractiveRunner(env.session, lambda st, se, so, tg: _Sink())
    code = r.run()
    assert code == 0
    assert seen["control"] is not None
    frame = seen["frame"]
    assert frame is not None
    assert frame.flip_byte == 0x1D
    assert frame.key_label == "^]"
    assert frame.active == "claude" and frame.others == ["codex"]
    assert r.flip_requested is False


def test_runner_labels_a_rebound_flip_key_for_the_bar(env_factory, monkeypatch):
    # [frame] flip_key = "ctrl-t" must reach the bar as "^T", not just as the
    # byte the detector watches.
    env = env_factory(active="claude")
    (paths.tandem_home() / "config.toml").write_text(
        '[frame]\nflip_key = "ctrl-t"\n'
    )
    seen = {}
    monkeypatch.setattr(
        runner, "run_in_pty",
        lambda argv, cwd=None, frame=None, control=None, child=None:
            seen.update(frame=frame) or 0,
    )
    runner.InteractiveRunner(env.session, lambda st, se, so, tg: _Sink()).run()
    assert seen["frame"].flip_byte == 0x14
    assert seen["frame"].key_label == "^T"


def test_key_label_spells_control_bytes_with_a_caret():
    assert runner._key_label(0x1D) == "^]"
    assert runner._key_label(0x14) == "^T"
    assert runner._key_label(0x01) == "^A"
    assert runner._key_label(0x7F) == "0x7f"   # not a control byte: fallback


def test_runner_reports_flip_requested(env_factory, monkeypatch):
    env = env_factory(active="claude")
    sentinel = paths.tandem_home() / "tmp" / f"{env.session.tandem_id}-claude.turn"

    def fake_run_in_pty(argv, cwd=None, frame=None, control=None, child=None):
        control.attach(_DeadChild())
        frame.on_flip()  # user pressed the keybind
        deadline = time.time() + 3
        while not frame.armed() and time.time() < deadline:
            time.sleep(0.02)
        sentinel.touch()  # turn-complete marker: the wait releases
        # the monitor's ladder finds no real child: control.terminate
        # returns "dead", flip_requested still set
        deadline = time.time() + 3
        while frame.armed() and time.time() < deadline:
            time.sleep(0.02)
        return 0

    monkeypatch.setattr(runner, "run_in_pty", fake_run_in_pty)
    r = runner.InteractiveRunner(env.session, lambda st, se, so, tg: _Sink())
    r.run()
    assert r.flip_requested is True


def test_runner_writes_bar_drop_marker(env_factory, monkeypatch, capsys):
    env = env_factory(active="claude")

    def fake_run_in_pty(argv, cwd=None, frame=None, control=None, child=None):
        frame.bar_dropped = True
        return 0

    monkeypatch.setattr(runner, "run_in_pty", fake_run_in_pty)
    r = runner.InteractiveRunner(env.session, lambda st, se, so, tg: _Sink())
    r.run()
    marker = paths.tandem_home() / "tmp" / f"{env.session.tandem_id}-bar-dropped"
    assert marker.exists()
    # A dropped bar is a note, not a sync failure: labelling it "sync error"
    # sends the user hunting a transcript divergence that never happened.
    out = capsys.readouterr().out
    assert "status bar disabled for this session" in out
    assert "sync error" not in out
    assert out.count("status bar disabled for this session") == 1  # no dupes
    assert r.reports == [
        "tandem: status bar disabled for this session (terminal conflict);"
        " set [frame] bar = false to silence"
    ]


def test_runner_holds_its_reports_back_for_a_flip(env_factory, monkeypatch, capsys):
    # A flip clears the screen a moment after run() returns, so anything
    # printed here is wiped before the user can read it. Collect, don't print
    # — the flip loop reprints onto the fresh screen.
    env = env_factory(active="claude")
    sentinel = paths.tandem_home() / "tmp" / f"{env.session.tandem_id}-claude.turn"

    def fake_run_in_pty(argv, cwd=None, frame=None, control=None, child=None):
        control.attach(_DeadChild())
        frame.bar_dropped = True
        frame.on_flip()
        deadline = time.time() + 3
        while not frame.armed() and time.time() < deadline:
            time.sleep(0.02)
        sentinel.touch()
        deadline = time.time() + 3
        while frame.armed() and time.time() < deadline:
            time.sleep(0.02)
        return 0

    monkeypatch.setattr(runner, "run_in_pty", fake_run_in_pty)
    r = runner.InteractiveRunner(env.session, lambda st, se, so, tg: _Sink())
    r.run()
    assert r.flip_requested is True
    assert any("status bar disabled" in line for line in r.reports)
    assert "status bar disabled" not in capsys.readouterr().out


def _run_capturing_monitor(env, monkeypatch):
    made = {}
    real = runner.FlipMonitor

    def capture(*a, **kw):
        made["monitor"] = real(*a, **kw)
        return made["monitor"]

    monkeypatch.setattr(runner, "FlipMonitor", capture)
    monkeypatch.setattr(
        runner, "run_in_pty",
        lambda argv, cwd=None, frame=None, control=None, child=None: 0,
    )
    runner.InteractiveRunner(env.session, lambda st, se, so, tg: _Sink()).run()
    return made["monitor"]


def test_marker_wired_derived_from_the_argv_hook_extras(env_factory, monkeypatch):
    # The hook extras that went into argv decide the mode, and they are read
    # once: hook_argv_extra re-reads config per call, so a second call could
    # disagree with what was actually launched.
    from tandem.harness.claude_code import ClaudeCodeAdapter

    env = env_factory(active="claude")
    calls = []
    real_extra = ClaudeCodeAdapter.hook_argv_extra

    def counting(self, sentinel):
        calls.append(sentinel)
        return real_extra(self, sentinel)

    monkeypatch.setattr(ClaudeCodeAdapter, "hook_argv_extra", counting)
    monitor = _run_capturing_monitor(env, monkeypatch)
    assert monitor.marker_wired is True
    assert len(calls) == 1


def test_marker_wired_false_when_adapter_injects_no_hook(env_factory, monkeypatch):
    # The marker-less shape: the adapter declined to wire a hook (codex
    # refusing to clobber a user notify), so quiescence is the only boundary.
    from tandem.harness.claude_code import ClaudeCodeAdapter

    env = env_factory(active="claude")
    monkeypatch.setattr(ClaudeCodeAdapter, "hook_argv_extra", lambda self, s: [])
    monitor = _run_capturing_monitor(env, monkeypatch)
    assert monitor.marker_wired is False


# ---- claude session-status probe -------------------------------------------


def _registry(tmp_path, monkeypatch, entries):
    """Fake ~/.claude/sessions with the given {filename: dict-or-raw} files."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
    d = tmp_path / ".claude" / "sessions"
    d.mkdir(parents=True)
    for name, entry in entries.items():
        text = entry if isinstance(entry, str) else json.dumps(entry)
        (d / name).write_text(text)
    return d


_SID = "11111111-1111-4111-8111-111111111111"


def _claude_adapter():
    from tandem.harness import get_adapter
    return get_adapter("claude")


def test_pid_alive_own_pid():
    from tandem.harness.claude_code import _pid_alive
    assert _pid_alive(os.getpid()) is True


def test_session_status_reads_busy_and_waiting(tmp_path, monkeypatch):
    me = os.getpid()
    _registry(tmp_path, monkeypatch, {
        f"{me}.json": {"pid": me, "sessionId": _SID, "status": "busy"},
    })
    assert _claude_adapter().session_status(_SID) == "busy"
    _registry(tmp_path.joinpath("b"), monkeypatch, {
        f"{me}.json": {"pid": me, "sessionId": _SID, "status": "waiting",
                       "waitingFor": "input needed"},
    })
    assert _claude_adapter().session_status(_SID) == "waiting"


def test_session_status_none_when_no_match(tmp_path, monkeypatch):
    me = os.getpid()
    _registry(tmp_path, monkeypatch, {
        f"{me}.json": {"pid": me, "sessionId": "someone-else", "status": "busy"},
    })
    assert _claude_adapter().session_status(_SID) is None


def test_session_status_none_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
    assert _claude_adapter().session_status(_SID) is None


def test_session_status_skips_stale_dead_pid_entry(tmp_path, monkeypatch):
    # A crashed run of this same resumed session leaves a dead-pid file
    # frozen at "busy"; the live entry must win.
    from tandem.harness import claude_code
    me = os.getpid()
    _registry(tmp_path, monkeypatch, {
        "99999.json": {"pid": 99999, "sessionId": _SID, "status": "busy"},
        f"{me}.json": {"pid": me, "sessionId": _SID, "status": "waiting"},
    })
    monkeypatch.setattr(claude_code, "_pid_alive", lambda pid: pid == me)
    assert claude_code.ClaudeCodeAdapter().session_status(_SID) == "waiting"
    # dead-only: no live entry at all reads as no answer
    monkeypatch.setattr(claude_code, "_pid_alive", lambda pid: False)
    assert claude_code.ClaudeCodeAdapter().session_status(_SID) is None


def test_session_status_busy_wins_among_live_matches(tmp_path, monkeypatch):
    from tandem.harness import claude_code
    me = os.getpid()
    _registry(tmp_path, monkeypatch, {
        "11.json": {"pid": 11, "sessionId": _SID, "status": "waiting"},
        "22.json": {"pid": 22, "sessionId": _SID, "status": "busy"},
    })
    monkeypatch.setattr(claude_code, "_pid_alive", lambda pid: True)
    assert claude_code.ClaudeCodeAdapter().session_status(_SID) == "busy"


def test_session_status_tolerates_garbage_files(tmp_path, monkeypatch):
    me = os.getpid()
    _registry(tmp_path, monkeypatch, {
        "junk.json": "not json{",
        "list.json": '["not", "a", "dict"]',
        "nopid.json": {"sessionId": _SID, "status": "busy"},
        f"{me}.json": {"pid": me, "sessionId": _SID, "status": "waiting"},
    })
    assert _claude_adapter().session_status(_SID) == "waiting"


# ---- status probe replaces the mtime rules ---------------------------------


def test_wait_probe_waiting_overrides_busy_mtimes(tmp_path):
    # transcript newer than sentinel: the mtime rules read mid-turn and
    # would hold for completion. The probe says the session is at its
    # prompt, and the probe is the whole test now.
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(t, time.time())
    _touch(s, time.time() - 30)
    assert wait_until_safe(t, s, cancelled=lambda: False, marker_wired=True,
                           status_probe=lambda: "waiting") is True


def test_wait_probe_no_answer_flips_eagerly(tmp_path):
    # single tier by spec: registry missing/unreadable -> flip now.
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(t, time.time())
    _touch(s, time.time() - 30)
    assert wait_until_safe(t, s, cancelled=lambda: False, marker_wired=True,
                           status_probe=lambda: None) is True


def test_wait_probe_busy_blocks_then_releases(tmp_path):
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(s, time.time())  # mtime rules would say idle (sentinel newest)
    _touch(t, time.time() - 30)
    state = {"status": "busy"}
    result = {}
    done = threading.Event()

    def wait():
        result["ok"] = wait_until_safe(t, s, cancelled=lambda: False,
                                       marker_wired=True, poll=0.05,
                                       status_probe=lambda: state["status"])
        done.set()

    threading.Thread(target=wait, daemon=True).start()
    assert not done.wait(timeout=0.4)   # busy verdict outranks idle mtimes
    state["status"] = "waiting"
    assert done.wait(timeout=3)
    assert result["ok"] is True


def test_wait_probe_busy_suppresses_valve(tmp_path):
    # A long-silent tool call: transcript ancient, quiesce tiny — the old
    # valve would fire and kill the live turn. The probe's "busy" holds.
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(t, time.time() - 3600)
    _touch(s, time.time() - 7200)
    state = {"status": "busy"}
    result = {}
    done = threading.Event()

    def wait():
        result["ok"] = wait_until_safe(t, s, cancelled=lambda: False,
                                       marker_wired=True, quiesce=0.1,
                                       poll=0.05,
                                       status_probe=lambda: state["status"])
        done.set()

    threading.Thread(target=wait, daemon=True).start()
    assert not done.wait(timeout=0.5)   # outlives quiesce: no valve
    state["status"] = "waiting"
    assert done.wait(timeout=3)
    assert result["ok"] is True


def test_wait_probe_busy_cancel_honored(tmp_path):
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(t, time.time())
    cancel = threading.Event()
    result = {}
    done = threading.Event()

    def wait():
        result["ok"] = wait_until_safe(t, s, cancelled=cancel.is_set,
                                       marker_wired=True, poll=0.05,
                                       status_probe=lambda: "busy")
        done.set()

    threading.Thread(target=wait, daemon=True).start()
    assert not done.wait(timeout=0.3)
    cancel.set()
    assert done.wait(timeout=3)
    assert result["ok"] is False


def test_monitor_probe_waiting_fires_immediately(tmp_path):
    # End-to-end through FlipMonitor: mtimes scream mid-turn, probe says
    # waiting -> the ladder runs. Mirrors test_monitor_arm_wait_terminate.
    t, s = tmp_path / "t.jsonl", tmp_path / "s.turn"
    _touch(t, time.time())
    control = _StubControl()
    m = FlipMonitor(control, [b"\x04"], transcript=t, sentinel=s,
                    marker_wired=True, poll=0.05,
                    status_probe=lambda: "waiting")
    m.start()
    m.flip_pressed()
    deadline = time.time() + 3
    while not m.flip_requested and time.time() < deadline:
        time.sleep(0.05)
    m.stop()
    assert m.flip_requested is True
    assert m.how == "soft"
    assert control.calls == [[b"\x04"]]


def test_runner_wires_status_probe_for_claude(env_factory, monkeypatch):
    env = env_factory(active="claude")
    made = {}
    real = runner.FlipMonitor

    def capture(*a, **kw):
        made["kw"] = kw
        return real(*a, **kw)

    monkeypatch.setattr(runner, "FlipMonitor", capture)
    monkeypatch.setattr(
        runner, "run_in_pty",
        lambda argv, cwd=None, frame=None, control=None, child=None: 0,
    )
    runner.InteractiveRunner(env.session, lambda st, se, so, tg: _Sink()).run()
    probe = made["kw"]["status_probe"]
    assert probe is not None
    # the probe closes over the claude sid: feed the registry and ask it
    me = os.getpid()
    d = Path(os.environ["CLAUDE_CONFIG_DIR"]) / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{me}.json").write_text(json.dumps(
        {"pid": me, "sessionId": env.session.native_id("claude"),
         "status": "busy"}))
    assert probe() == "busy"


def test_runner_wires_no_probe_for_codex(env_factory, monkeypatch):
    env = env_factory(active="codex")
    made = {}
    real = runner.FlipMonitor

    def capture(*a, **kw):
        made["kw"] = kw
        return real(*a, **kw)

    monkeypatch.setattr(runner, "FlipMonitor", capture)
    monkeypatch.setattr(
        runner, "run_in_pty",
        lambda argv, cwd=None, frame=None, control=None, child=None: 0,
    )
    runner.InteractiveRunner(env.session, lambda st, se, so, tg: _Sink()).run()
    assert made["kw"]["status_probe"] is None


def test_session_status_rejects_nonpositive_and_bool_pids(tmp_path, monkeypatch):
    # os.kill(0,0)/os.kill(-N,0) probe process groups and would read
    # "alive"; with busy-wins and no valve a pid-0 busy entry pins the
    # flip forever. The guard drops them before _pid_alive runs.
    _registry(tmp_path, monkeypatch, {
        "zero.json": {"pid": 0, "sessionId": _SID, "status": "busy"},
        "neg.json": {"pid": -1, "sessionId": _SID, "status": "busy"},
        "bool.json": {"pid": True, "sessionId": _SID, "status": "busy"},
    })
    assert _claude_adapter().session_status(_SID) is None


def test_runner_probe_swallows_raising_session_status(env_factory, monkeypatch):
    # OverflowError from os.kill on an absurd pid is not an OSError; a
    # probe that raises would kill the flip thread. The wiring maps any
    # escape to None (single tier: no answer -> flippable).
    env = env_factory(active="claude")
    made = {}
    real = runner.FlipMonitor

    def capture(*a, **kw):
        made["kw"] = kw
        return real(*a, **kw)

    monkeypatch.setattr(runner, "FlipMonitor", capture)
    monkeypatch.setattr(
        runner, "run_in_pty",
        lambda argv, cwd=None, frame=None, control=None, child=None: 0,
    )
    runner.InteractiveRunner(env.session, lambda st, se, so, tg: _Sink()).run()
    from tandem.harness.claude_code import ClaudeCodeAdapter
    monkeypatch.setattr(
        ClaudeCodeAdapter, "session_status",
        lambda self, sid: (_ for _ in ()).throw(OverflowError()),
    )
    assert made["kw"]["status_probe"]() is None


# -- the fire-at-flip warm spawn ------------------------------------------


class _StdinWithFileno:
    """Stand-in for the terminal the fire measures. pytest replaces
    `sys.stdin` with a pseudofile whose `fileno()` raises, so a fire that
    reads the window size would blow up before it ever spawned. In
    production the gate the fire checks first — `_stdin_tty()` — is exactly
    what guarantees the fileno is there, so a test that forces that gate
    open has to supply the fileno too."""

    def fileno(self):
        return 0


class _QuietWatcher:
    """No native FSEvents thread in tests whose runner exits immediately."""

    def watch(self, path):
        pass

    def start(self):
        pass

    def wait(self):
        time.sleep(0.01)

    def stop(self):
        pass


def _flip_driver(env):
    """The file's established flip-driving `run_in_pty` stand-in: attach a
    dead child (the ladder reads it as already gone), press the keybind,
    touch the turn-complete marker, then hold until the monitor has actually
    decided the flip — `armed()` goes False the moment `flip_requested`
    lands. Without that settle a `run_in_pty` that returns first would race
    `monitor.stop()` and the flip would never fire. The fire itself finishes
    under `stop()`'s join, which is what the runner's finally reads the slot
    after."""
    sentinel = paths.tandem_home() / "tmp" / \
        f"{env.session.tandem_id}-{env.session.active}.turn"

    def fake_run_in_pty(argv, cwd=None, env=None, frame=None, control=None,
                        child=None):
        control.attach(_DeadChild())
        frame.on_flip()                      # arm the flip
        deadline = time.time() + 3
        while not frame.armed() and time.time() < deadline:
            time.sleep(0.02)
        # the sentinel touch satisfies the boundary wait; then wait for the
        # monitor to fire and run its (stubbed-fast) ladder
        sentinel.touch()
        deadline = time.time() + 3
        while frame.armed() and time.time() < deadline:
            time.sleep(0.02)
        return 0

    return fake_run_in_pty


def _drive_flip(monkeypatch, env, *, warm_cfg=True, tty=True, spawns=None,
                expect_spawn=True):
    """Run a real InteractiveRunner through a driven flip (the file's
    established _DeadChild/on_flip/sentinel recipe), recording fire-time
    spawns. Returns the runner after run() completes."""
    import tandem.runner as runner_mod

    if not warm_cfg:
        (paths.tandem_home() / "config.toml").write_text("[frame]\nwarm = false\n")
    monkeypatch.setattr(runner_mod, "_stdin_tty", lambda: tty)
    monkeypatch.setattr(sys, "stdin", _StdinWithFileno())
    recorded = spawns if spawns is not None else []
    spawn_done = threading.Event()

    class FakeChild:
        def __init__(self, recipe, dims, shadow_size):
            self.recipe = recipe
            self.dims = dims
            self.shadow_size = shadow_size
            self.killed = False

        def alive(self):
            return True

        def kill(self):
            self.killed = True

    def fake_spawn_hidden(recipe, dims, shadow_size):
        child = FakeChild(recipe, dims, shadow_size)
        recorded.append(child)
        spawn_done.set()
        return child

    monkeypatch.setattr(runner_mod, "spawn_hidden", fake_spawn_hidden)
    monkeypatch.setattr(runner_mod, "TranscriptWatcher", _QuietWatcher)
    drive_flip = _flip_driver(env)

    def fake_run_in_pty(*args, **kwargs):
        code = drive_flip(*args, **kwargs)
        if expect_spawn and warm_cfg and tty and env.codex_shadow.exists():
            assert spawn_done.wait(3)
        return code

    monkeypatch.setattr(runner_mod, "run_in_pty", fake_run_in_pty)
    r = runner_mod.InteractiveRunner(env.session, sink_factory=_null_sink)
    r.run()
    return r, recorded


def test_flip_fire_spawns_the_shadow_hidden(env_factory, monkeypatch):
    env = env_factory(active="claude")
    r, spawns = _drive_flip(monkeypatch, env)
    assert r.flip_requested
    assert len(spawns) == 1
    assert spawns[0].recipe.side == "codex"
    assert spawns[0].shadow_size == env.codex_shadow.stat().st_size
    assert r.warm_child is spawns[0]


def test_no_fire_when_config_off(env_factory, monkeypatch):
    env = env_factory(active="claude")
    r, spawns = _drive_flip(monkeypatch, env, warm_cfg=False)
    assert r.flip_requested and spawns == [] and r.warm_child is None


def test_no_fire_without_a_tty(env_factory, monkeypatch):
    env = env_factory(active="claude")
    r, spawns = _drive_flip(monkeypatch, env, tty=False)
    assert r.flip_requested                 # or the gate is never reached
    assert spawns == [] and r.warm_child is None


def test_no_fire_when_shadow_transcript_is_missing(env_factory, monkeypatch):
    env = env_factory(active="claude")
    env.codex_shadow.unlink()               # never fresh-mint codex
    r, spawns = _drive_flip(monkeypatch, env)
    assert r.flip_requested                 # or the gate is never reached
    assert spawns == [] and r.warm_child is None


def test_a_raising_fire_still_flips(env_factory, monkeypatch):
    import tandem.runner as runner_mod
    env = env_factory(active="claude")
    monkeypatch.setattr(runner_mod, "spawn_hidden",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("boom")))
    monkeypatch.setattr(runner_mod, "TranscriptWatcher", _QuietWatcher)
    monkeypatch.setattr(runner_mod, "_stdin_tty", lambda: True)
    monkeypatch.setattr(sys, "stdin", _StdinWithFileno())
    monkeypatch.setattr(runner_mod, "run_in_pty", _flip_driver(env))
    r = runner_mod.InteractiveRunner(env.session, sink_factory=_null_sink)
    assert r.run() == 0                     # monitor thread survived the raise
    assert r.flip_requested and r.warm_child is None


def test_a_slow_fire_does_not_delay_the_termination_ladder(env_factory,
                                                           monkeypatch):
    import tandem.runner as runner_mod
    env = env_factory(active="claude")
    order = []
    release_spawn = threading.Event()
    spawn_done = threading.Event()

    def slow_spawn(*args, **kwargs):
        order.append("spawn-start")
        release_spawn.wait(timeout=1)
        order.append("spawn-done")
        spawn_done.set()
        return _DeadChild()

    def terminate(self, soft, **kwargs):
        order.append("ladder")
        release_spawn.set()
        return "soft"

    monkeypatch.setattr(runner_mod, "spawn_hidden", slow_spawn)
    monkeypatch.setattr(runner_mod, "TranscriptWatcher", _QuietWatcher)
    monkeypatch.setattr(runner_mod.PtyControl, "terminate", terminate)
    monkeypatch.setattr(runner_mod, "_stdin_tty", lambda: True)
    monkeypatch.setattr(sys, "stdin", _StdinWithFileno())
    monkeypatch.setattr(runner_mod, "run_in_pty", _flip_driver(env))
    r = runner_mod.InteractiveRunner(env.session, sink_factory=_null_sink)
    r.run()

    assert r.flip_requested
    assert spawn_done.wait(3)
    assert order.index("ladder") < order.index("spawn-done")


def test_a_slow_recipe_build_does_not_delay_the_termination_ladder(
        env_factory, monkeypatch):
    # The fire's filesystem setup — the shadow stat, build_launch's config
    # reads, the sentinel mkdir — belongs to the launch worker: run on the
    # monitor thread it would sit between the flip decision and the ladder,
    # delaying the very teardown the fire exists to overlap.
    import tandem.runner as runner_mod
    env = env_factory(active="claude")
    order = []
    release_build = threading.Event()
    build_done = threading.Event()
    real_build = runner_mod.build_launch

    def slow_build(session, side):
        if side != "codex":
            return real_build(session, side)   # the active side's own launch
        order.append("build-start")
        release_build.wait(timeout=1)
        order.append("build-done")
        build_done.set()
        return real_build(session, side)

    def terminate(self, soft, **kwargs):
        order.append("ladder")
        release_build.set()
        return "soft"

    monkeypatch.setattr(runner_mod, "build_launch", slow_build)
    monkeypatch.setattr(runner_mod, "spawn_hidden",
                        lambda *a, **kw: _DeadChild())
    monkeypatch.setattr(runner_mod, "TranscriptWatcher", _QuietWatcher)
    monkeypatch.setattr(runner_mod.PtyControl, "terminate", terminate)
    monkeypatch.setattr(runner_mod, "_stdin_tty", lambda: True)
    monkeypatch.setattr(sys, "stdin", _StdinWithFileno())
    monkeypatch.setattr(runner_mod, "run_in_pty", _flip_driver(env))
    r = runner_mod.InteractiveRunner(env.session, sink_factory=_null_sink)
    r.run()

    assert r.flip_requested
    assert build_done.wait(3)
    assert order.index("ladder") < order.index("build-done")


def test_no_flip_leaves_no_fire_spawn(env_factory, monkeypatch):
    env = env_factory(active="claude")
    import tandem.runner as runner_mod
    spawns = []
    monkeypatch.setattr(runner_mod, "spawn_hidden",
                        lambda *a, **kw: spawns.append(1))
    monkeypatch.setattr(runner_mod, "TranscriptWatcher", _QuietWatcher)
    monkeypatch.setattr(runner_mod, "_stdin_tty", lambda: True)
    monkeypatch.setattr(runner_mod, "run_in_pty", lambda *a, **kw: 0)
    r = runner_mod.InteractiveRunner(env.session, sink_factory=_null_sink)
    r.run()
    assert not r.flip_requested and spawns == [] and r.warm_child is None


def test_fire_kills_its_child_when_the_slot_is_already_closed(env_factory,
                                                             monkeypatch):
    # The one case monitor.stop()'s join cannot cover: its timeout expires
    # with a spawn still in flight, so the runner's finally reads and closes
    # the slot first and the fire lands after. Driven by calling the fire
    # hook once run() has returned — the slot is then closed for good, which
    # is exactly the state an expired join leaves behind. The child must be
    # killed by the fire itself; nothing else is left to reap it.
    import tandem.runner as runner_mod
    env = env_factory(active="claude")
    monitors = []

    class CapturingMonitor(runner_mod.FlipMonitor):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            monitors.append(self)

    class FakeChild:
        def __init__(self):
            self.killed = False

        def kill(self):
            self.killed = True

    spawns = []

    def fake_spawn_hidden(recipe, dims, shadow_size):
        spawns.append(FakeChild())
        return spawns[-1]

    monkeypatch.setattr(runner_mod, "FlipMonitor", CapturingMonitor)
    monkeypatch.setattr(runner_mod, "spawn_hidden", fake_spawn_hidden)
    monkeypatch.setattr(runner_mod, "TranscriptWatcher", _QuietWatcher)
    monkeypatch.setattr(runner_mod, "_stdin_tty", lambda: True)
    monkeypatch.setattr(sys, "stdin", _StdinWithFileno())
    monkeypatch.setattr(runner_mod, "run_in_pty", lambda *a, **kw: 0)
    r = runner_mod.InteractiveRunner(env.session, sink_factory=_null_sink)
    r.run()
    assert spawns == [] and r.warm_child is None   # nothing fired during the run

    monitors[0].on_flip_decided()                  # the in-flight spawn lands
    deadline = time.time() + 3
    while not (spawns and spawns[0].killed) and time.time() < deadline:
        time.sleep(0.02)
    assert len(spawns) == 1 and spawns[0].killed   # reaped, not stranded
    assert r.warm_child is None                    # and never adopted


def test_runner_adopts_a_live_child(env_factory, monkeypatch):
    import tandem.runner as runner_mod
    from tandem.warm import build_launch
    env = env_factory(active="claude")
    recipe = build_launch(env.session, "claude")
    seen = {}

    class FakeWarmChild:
        def __init__(self):
            self.recipe = recipe
            self.released = False

        def alive(self):
            return True

        def release(self):
            self.released = True
            return "raw-child"

    def fake_run_in_pty(argv, cwd=None, env=None, frame=None, control=None,
                        child=None):
        seen["argv"] = argv
        seen["child"] = child
        return 0

    def no_rebuild(*a, **kw):
        # The recipe is bound once, at spawn time: hook_argv_extra re-reads
        # config per call, so a rebuild here could disagree with the argv the
        # adopted child is already running under.
        raise AssertionError("adoption must not rebuild the launch recipe")

    monkeypatch.setattr(runner_mod, "build_launch", no_rebuild)
    monkeypatch.setattr(runner_mod, "run_in_pty", fake_run_in_pty)
    wc = FakeWarmChild()
    r = runner_mod.InteractiveRunner(env.session, sink_factory=_null_sink,
                                     adopt_child=wc)
    r.run()
    assert wc.released
    assert seen["child"] == "raw-child"
    assert seen["argv"] == recipe.argv     # the recorded recipe, not a rebuild


def test_runner_kills_a_child_that_refuses_to_release(env_factory, monkeypatch):
    # release() returns None when the discard reader still owns the fd. The
    # raw None goes to run_in_pty (a cold spawn), and the WarmChild must be
    # killed: nothing else is left to reap the hidden process.
    import tandem.runner as runner_mod
    from tandem.warm import build_launch
    env = env_factory(active="claude")
    recipe = build_launch(env.session, "claude")
    seen = {}

    class WedgedWarmChild:
        def __init__(self):
            self.recipe = recipe
            self.killed = False

        def alive(self):
            return True

        def release(self):
            return None          # reader refused to join

        def kill(self):
            self.killed = True

    def fake_run_in_pty(argv, cwd=None, env=None, frame=None, control=None,
                        child=None):
        seen["child"] = child
        return 0

    monkeypatch.setattr(runner_mod, "run_in_pty", fake_run_in_pty)
    wc = WedgedWarmChild()
    runner_mod.InteractiveRunner(env.session, sink_factory=_null_sink,
                                 adopt_child=wc).run()
    assert seen["child"] is None    # cold spawn, not a half-owned fd
    assert wc.killed


def test_runner_kills_an_adoptee_it_never_handed_over(env_factory, monkeypatch):
    """Anything raising between "we will adopt this" and the handover leaves
    the hidden child unowned: the flip loop popped it out of its carry to
    build this runner and only ever gets `warm_child` back. So the runner
    reaps it — and, on the other side of the same guard, never reaps one it
    did hand over, which would terminate the harness the user is looking at."""
    import tandem.runner as runner_mod
    from tandem.warm import build_launch
    env = env_factory(active="claude")
    recipe = build_launch(env.session, "claude")

    class Adoptee:
        def __init__(self):
            self.recipe = recipe
            self.released = False
            self.killed = False

        def alive(self):
            return True

        def release(self):
            self.released = True
            return "raw-child"

        def kill(self):
            self.killed = True

    def boom():
        raise ValueError("bad [frame] table in config.toml")

    with monkeypatch.context() as m:
        # a pre-handover step that raises; the config parse is the realistic
        # one, but the guard is not specific to it
        m.setattr(runner_mod, "load_frame_config", boom)
        never = Adoptee()
        with pytest.raises(ValueError):
            runner_mod.InteractiveRunner(env.session, sink_factory=_null_sink,
                                         adopt_child=never).run()
    assert never.killed
    assert not never.released

    monkeypatch.setattr(runner_mod, "run_in_pty", lambda *a, **kw: 0)
    handed = Adoptee()
    runner_mod.InteractiveRunner(env.session, sink_factory=_null_sink,
                                 adopt_child=handed).run()
    assert handed.released
    assert not handed.killed   # it is the running harness now


def test_runner_ignores_a_dead_or_mismatched_adoptee(env_factory, monkeypatch):
    # Neither a dead child nor one warmed for the other side is adoptable:
    # the runner rebuilds its own recipe and spawns cold. It still reaps
    # them — the carry was emptied to build this runner, so a live wrong-side
    # standby dropped here would outlive the session with nobody to kill it.
    import tandem.runner as runner_mod
    from tandem.warm import build_launch
    env = env_factory(active="claude")
    seen = {}

    class Adoptee:
        def __init__(self, side, is_alive):
            self.recipe = build_launch(env.session, side)
            self._alive = is_alive
            self.released = False
            self.killed = False

        def alive(self):
            return self._alive

        def release(self):
            self.released = True
            return "raw-child"

        def kill(self):
            self.killed = True

    def fake_run_in_pty(argv, cwd=None, env=None, frame=None, control=None,
                        child=None):
        seen["child"] = child
        return 0

    monkeypatch.setattr(runner_mod, "run_in_pty", fake_run_in_pty)
    for adoptee in (Adoptee("claude", False), Adoptee("codex", True)):
        seen.clear()   # or the second iteration could pass on stale evidence
        runner_mod.InteractiveRunner(env.session, sink_factory=_null_sink,
                                     adopt_child=adoptee).run()
        assert seen["child"] is None
        assert not adoptee.released
        assert adoptee.killed


def test_a_fired_child_is_not_leaked_when_run_in_pty_raises(env_factory,
                                                            monkeypatch):
    # The fire lands on the monitor thread, so a run_in_pty that explodes
    # after it must still surrender the child: the finally publishes it to
    # `warm_child` (the flip loop fills its carry from that same finally) or
    # a hidden harness outlives the session with nobody left to reap it.
    import tandem.runner as runner_mod
    env = env_factory(active="claude")
    spawned = []

    class FakeChild:
        def __init__(self, shadow_size):
            self.shadow_size = shadow_size
            self.killed = False

        def alive(self):
            return True

        def kill(self):
            self.killed = True

    def fake_spawn_hidden(recipe, dims, shadow_size):
        spawned.append(FakeChild(shadow_size))
        return spawned[-1]

    driver = _flip_driver(env)

    def boom(argv, cwd=None, env=None, frame=None, control=None, child=None):
        driver(argv, cwd=cwd, frame=frame, control=control, child=child)
        raise RuntimeError("pty exploded")

    monkeypatch.setattr(runner_mod, "_stdin_tty", lambda: True)
    monkeypatch.setattr(sys, "stdin", _StdinWithFileno())
    monkeypatch.setattr(runner_mod, "spawn_hidden", fake_spawn_hidden)
    monkeypatch.setattr(runner_mod, "run_in_pty", boom)
    r = runner_mod.InteractiveRunner(env.session, sink_factory=_null_sink)
    with pytest.raises(RuntimeError):
        r.run()
    assert len(spawned) == 1
    assert r.flip_requested
    assert r.warm_child is spawned[0]   # the carry's only reference to it
    assert not spawned[0].killed


def test_tail_thread_drains_all_directions(tmp_path, monkeypatch):
    """One source line lands in BOTH shadows via the runner's TailLoop set."""
    from conftest import Env3, claude_user, write_line
    from tandem.runner import TailLoop
    from tandem.sync import SyncEngine

    env = Env3(tmp_path, monkeypatch)
    write_line(env.claude_shadow, claude_user("fan out"))
    for target in env.session.targets_for("claude"):
        engine = SyncEngine(env.store, env.session, "claude", target)
        loop = TailLoop(env.store, env.session, "claude", target,
                        env.claude_shadow, engine)
        assert loop.drain() >= 1


def test_warm_skips_opencode_target(tmp_path, monkeypatch):
    """fire_warm never spawns when next-in-cycle is opencode (v1 carve-out):
    an opencode TUI booted pre-drain would cache the session pre-drain and
    never show the last turn, so opencode-bound flips run cold."""
    from conftest import Env3

    env = Env3(tmp_path, monkeypatch, active="codex")
    assert env.session.next_active("codex") == "opencode"
    r, spawns = _drive_flip(monkeypatch, env, expect_spawn=False)
    assert spawns == [] and r.warm_child is None


# -- rate limits on the bar ---------------------------------------------------


def _wait_for(pred, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def test_runner_polls_rate_limits_for_every_participant(env_factory, monkeypatch):
    from tandem import ratelimit
    env = env_factory(active="claude")
    monkeypatch.setitem(ratelimit.DEFAULT_FETCHERS, "claude",
                        lambda: [ratelimit.Window("5h", 4), ratelimit.Window("7d", 41)])
    monkeypatch.setitem(ratelimit.DEFAULT_FETCHERS, "codex",
                        lambda: [ratelimit.Window("7d", 12)])
    seen = {}

    def fake_run_in_pty(argv, cwd=None, frame=None, control=None, child=None):
        # the pump reports the bar drawn; only then does the poller start,
        # publishing on its own thread
        assert not any(frame.limits().values())   # seeded blank, nothing fetched yet
        frame.on_bar(True)
        assert _wait_for(lambda: frame.limits() and "codex" in frame.limits())
        seen["limits"] = dict(frame.limits())
        return 0

    monkeypatch.setattr(runner, "run_in_pty", fake_run_in_pty)
    runner.InteractiveRunner(env.session, lambda st, se, so, tg: _Sink()).run()
    assert seen["limits"] == {"claude": "5h 4% 7d 41%", "codex": "7d 12%"}
    # and it is torn down with the run
    assert not any(t.name == "tandem-ratelimit" and t.is_alive()
                   for t in threading.enumerate())


def test_runner_skips_the_rate_limit_poll_when_disabled(env_factory, monkeypatch):
    from tandem import ratelimit
    env = env_factory(active="claude")
    (paths.tandem_home() / "config.toml").write_text("[frame]\nrate_limits = false\n")
    called = []
    monkeypatch.setitem(ratelimit.DEFAULT_FETCHERS, "claude", lambda: called.append(1))
    monkeypatch.setitem(ratelimit.DEFAULT_FETCHERS, "codex", lambda: called.append(1))
    seen = {}
    monkeypatch.setattr(
        runner, "run_in_pty",
        lambda argv, cwd=None, frame=None, control=None, child=None:
            seen.update(frame=frame) or 0,
    )
    runner.InteractiveRunner(env.session, lambda st, se, so, tg: _Sink()).run()
    assert seen["frame"].limits is None
    assert called == []


def test_runner_skips_the_rate_limit_poll_without_a_bar(env_factory, monkeypatch):
    # nothing to paint the figures on: no network calls either
    from tandem import ratelimit
    env = env_factory(active="claude")
    (paths.tandem_home() / "config.toml").write_text("[frame]\nbar = false\n")
    called = []
    monkeypatch.setitem(ratelimit.DEFAULT_FETCHERS, "claude", lambda: called.append(1))
    monkeypatch.setitem(ratelimit.DEFAULT_FETCHERS, "codex", lambda: called.append(1))
    monkeypatch.setattr(runner, "run_in_pty",
                        lambda argv, cwd=None, frame=None, control=None, child=None: 0)
    runner.InteractiveRunner(env.session, lambda st, se, so, tg: _Sink()).run()
    assert called == []


def test_runner_makes_no_rate_limit_calls_when_the_pump_never_draws_a_bar(
        env_factory, monkeypatch):
    # config says bar, but the pump found no tty / too few rows: no bar, no calls
    from tandem import ratelimit
    env = env_factory(active="claude")
    called = []
    monkeypatch.setitem(ratelimit.DEFAULT_FETCHERS, "claude", lambda: called.append(1))
    monkeypatch.setitem(ratelimit.DEFAULT_FETCHERS, "codex", lambda: called.append(1))

    def fake_run_in_pty(argv, cwd=None, frame=None, control=None, child=None):
        frame.on_bar(False)
        time.sleep(0.2)
        return 0

    monkeypatch.setattr(runner, "run_in_pty", fake_run_in_pty)
    runner.InteractiveRunner(env.session, lambda st, se, so, tg: _Sink()).run()
    assert called == []


def test_runner_halts_the_poll_when_the_bar_drops(env_factory, monkeypatch):
    from tandem import ratelimit
    env = env_factory(active="claude")
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return [ratelimit.Window("7d", 12)]

    monkeypatch.setitem(ratelimit.DEFAULT_FETCHERS, "claude", fetch)
    monkeypatch.setitem(ratelimit.DEFAULT_FETCHERS, "codex", fetch)

    def fake_run_in_pty(argv, cwd=None, frame=None, control=None, child=None):
        frame.on_bar(True)
        assert _wait_for(lambda: calls["n"] >= 2)
        frame.on_bar(False)
        assert _wait_for(lambda: not any(
            t.name == "tandem-ratelimit" and t.is_alive() for t in threading.enumerate()))
        return 0

    monkeypatch.setattr(runner, "run_in_pty", fake_run_in_pty)
    runner.InteractiveRunner(env.session, lambda st, se, so, tg: _Sink()).run()


def test_runner_pokes_the_poller_when_a_response_lands(env_factory, monkeypatch):
    """The tail thread sees the active transcript grow; a change in the usage
    text means a response just landed, so the account figures get refreshed
    ahead of the interval."""
    import functools

    from tandem import ratelimit
    env = env_factory(active="claude")
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return [ratelimit.Window("5h", 4)]

    monkeypatch.setitem(ratelimit.DEFAULT_FETCHERS, "claude", fetch)
    monkeypatch.setitem(ratelimit.DEFAULT_FETCHERS, "codex", lambda: None)
    monkeypatch.setattr(runner, "RateLimitPoller",
                        functools.partial(ratelimit.RateLimitPoller,
                                          interval=3600, min_gap=0))

    def fake_run_in_pty(argv, cwd=None, frame=None, control=None, child=None):
        frame.on_bar(True)
        assert _wait_for(lambda: calls["n"] == 1)
        time.sleep(0.2)
        assert calls["n"] == 1
        with open(env.claude_shadow, "a") as f:
            f.write(json.dumps({
                "type": "assistant", "uuid": "a-1", "sessionId": "x",
                "message": {"id": "m1", "role": "assistant",
                            "content": [{"type": "text", "text": "hi"}],
                            "usage": {"input_tokens": 10, "output_tokens": 5}},
            }) + "\n")
        assert _wait_for(lambda: calls["n"] >= 2, timeout=5)
        return 0

    monkeypatch.setattr(runner, "run_in_pty", fake_run_in_pty)
    runner.InteractiveRunner(env.session, lambda st, se, so, tg: _Sink()).run()


def test_codex_auxiliary_notify_does_not_flip_running_primary_turn(env_factory, monkeypatch):
    env = env_factory(active="codex")
    env.codex_shadow.write_text(json.dumps({
        "type": "event_msg", "payload": {"type": "task_started", "turn_id": "primary"},
    }) + "\n")
    made = {}
    real_monitor = runner.FlipMonitor

    def capture(*args, **kwargs):
        made["monitor"] = real_monitor(*args, **kwargs)
        return made["monitor"]

    monkeypatch.setattr(runner, "FlipMonitor", capture)
    monkeypatch.setattr(runner.PtyControl, "terminate", lambda *a, **kw: "test")

    def native_turn(argv, **kwargs):
        monitor = made["monitor"]
        monitor.sentinel.touch()  # native title thread's notify, not primary completion
        monitor.flip_pressed()
        time.sleep(0.3)
        assert not monitor.flip_requested
        with env.codex_shadow.open("a") as stream:
            stream.write(json.dumps({"type": "event_msg", "payload": {
                "type": "task_complete", "turn_id": "primary",
            }}) + "\n")
        deadline = time.monotonic() + 2
        while not monitor.flip_requested and time.monotonic() < deadline:
            time.sleep(0.01)
        assert monitor.flip_requested
        return 0

    monkeypatch.setattr(runner, "run_in_pty", native_turn)
    result = runner.InteractiveRunner(env.session, _null_sink)
    assert result.run() == 0
    assert result.flip_requested


def _codex_lifecycle(path, kind, turn_id="primary", mode="a"):
    with path.open(mode) as stream:
        stream.write(json.dumps({"type": "event_msg", "payload": {
            "type": kind, "turn_id": turn_id,
        }}) + "\n")


@pytest.mark.parametrize("terminal", ["task_complete", "turn_aborted"])
def test_codex_primary_boundary_terminal_and_late_writes(tmp_path, terminal):
    path = tmp_path / "rollout.jsonl"
    probe = runner.CodexTurnBoundary()
    assert not probe(None)
    assert not probe(path)
    _codex_lifecycle(path, "task_started", mode="w")
    assert not probe(path)
    _codex_lifecycle(path, terminal, turn_id="auxiliary")
    assert not probe(path)
    _codex_lifecycle(path, terminal)
    assert probe(path)
    _codex_lifecycle(path, "token_count")
    assert probe(path)  # housekeeping after the terminal event is not a new turn
    _codex_lifecycle(path, "task_started", turn_id="next")
    assert not probe(path)


def test_codex_primary_boundary_partial_corrupt_replaced_and_truncated(tmp_path):
    path = tmp_path / "rollout.jsonl"
    probe = runner.CodexTurnBoundary()
    _codex_lifecycle(path, "task_started", mode="w")
    partial = json.dumps({"type": "event_msg", "payload": {
        "type": "task_complete", "turn_id": "primary",
    }})
    with path.open("a") as stream:
        stream.write(partial)
    assert not probe(path)
    with path.open("a") as stream:
        stream.write("\n")
    assert probe(path)
    with path.open("a") as stream:
        stream.write("invalid\n")
    assert not probe(path)
    _codex_lifecycle(path, "task_complete")
    assert not probe(path)
    _codex_lifecycle(path, "task_started", mode="w")
    assert not probe(path)
    _codex_lifecycle(path, "task_complete")
    assert probe(path)
    replacement = tmp_path / "replacement"
    _codex_lifecycle(replacement, "task_started", mode="w")
    replacement.replace(path)
    assert not probe(path)
    path.unlink()
    assert not probe(path)


def test_codex_primary_boundary_requires_tool_result_and_terminal(tmp_path):
    path = tmp_path / "rollout.jsonl"
    probe = runner.CodexTurnBoundary()
    _codex_lifecycle(path, "task_started", mode="w")
    with path.open("a") as stream:
        stream.write(json.dumps({"type": "response_item", "payload": {
            "type": "custom_tool_call", "call_id": "sleep",
        }}) + "\n")
    _codex_lifecycle(path, "task_complete")
    assert not probe(path)
    with path.open("a") as stream:
        stream.write(json.dumps({"type": "response_item", "payload": {
            "type": "custom_tool_call_output", "call_id": "sleep", "output": "done",
        }}) + "\n")
    assert probe(path)


def test_codex_primary_boundary_no_lifecycle_and_error_without_terminal_hold(tmp_path):
    path = tmp_path / "rollout.jsonl"
    probe = runner.CodexTurnBoundary()
    path.write_text(json.dumps({"type": "session_meta", "payload": {"id": "primary"}}) + "\n")
    assert not probe(path)
    _codex_lifecycle(path, "task_started")
    _codex_lifecycle(path, "error")
    assert not probe(path)


def test_codex_custom_notify_retains_markerless_fallback(env_factory, monkeypatch):
    from tandem.harness.codex import CodexAdapter
    env = env_factory(active="codex")
    monkeypatch.setattr(CodexAdapter, "hook_argv_extra", lambda self, sentinel: [])
    monitor = _run_capturing_monitor(env, monkeypatch)
    assert not monitor.marker_wired
    assert monitor.turn_boundary is None


@pytest.mark.parametrize("call_id", [None, "", 123])
def test_codex_primary_boundary_malformed_call_identity_holds(tmp_path, call_id):
    path = tmp_path / "rollout.jsonl"
    probe = runner.CodexTurnBoundary()
    _codex_lifecycle(path, "task_started", mode="w")
    with path.open("a") as stream:
        stream.write(json.dumps({"type": "response_item", "payload": {
            "type": "custom_tool_call", "call_id": call_id,
        }}) + "\n")
    _codex_lifecycle(path, "task_complete")
    assert not probe(path)


@pytest.mark.parametrize("payload", [None, 1, [], "missing"])
def test_codex_primary_boundary_malformed_payload_cannot_close_old_turn(tmp_path, payload):
    path = tmp_path / "rollout.jsonl"
    probe = runner.CodexTurnBoundary()
    _codex_lifecycle(path, "task_started", mode="w")
    record = {"type": "event_msg"}
    if payload != "missing":
        record["payload"] = payload
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    _codex_lifecycle(path, "task_complete")
    assert not probe(path)


@pytest.mark.parametrize("record", [
    {"type": "event_msg", "payload": {"type": "user_message", "message": "queued"}},
    {"type": "response_item", "payload": {"type": "message", "role": "user"}},
    {"type": "event_msg", "payload": {"type": "future_lifecycle"}},
    {"type": "future_record", "payload": {}},
    {"type": "event_msg", "payload": {}},
    {"type": "response_item", "payload": {}},
])
def test_codex_primary_boundary_queued_or_unknown_record_invalidates_terminal(tmp_path, record):
    path = tmp_path / "rollout.jsonl"
    probe = runner.CodexTurnBoundary()
    _codex_lifecycle(path, "task_started", mode="w")
    _codex_lifecycle(path, "task_complete")
    assert probe(path)
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    assert not probe(path)
    _codex_lifecycle(path, "task_complete")
    assert not probe(path)
    _codex_lifecycle(path, "task_started", turn_id="next")
    _codex_lifecycle(path, "task_complete", turn_id="next")
    assert probe(path)
