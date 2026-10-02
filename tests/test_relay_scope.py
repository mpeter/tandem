"""The plugin scopes relay enforcement before invoking the installed CLI."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parent.parent / "plugin" / "hooks" / "relay-scope.py"


def invoke(tmp_path, phase, payload, cli=None):
    if cli is not None:
        command = tmp_path / "tandem"
        command.write_text("#!/usr/bin/python3\n" + cli)
        command.chmod(0o700)
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return subprocess.run(
        [sys.executable, str(SCRIPT), phase], input=raw,
        capture_output=True, env={**os.environ, "PATH": str(tmp_path)},
    )


@pytest.mark.parametrize("phase,payload", [
    ("pre", {"tool_name": "Bash", "tool_input": {"command": "printf ordinary"}}),
    ("pre", {"tool_name": "Agent", "tool_input": {"subagent_type": "tandem:gpt"}}),
    ("pre", {"tool_name": "MCP", "tool_input": {"agent_type": "tandem:gpt"}}),
    ("pre", {"agent_type": "other", "tool_input": {"agent_type": "tandem:codex-worker"}}),
    ("post", {"tool_name": "MCP", "tool_input": {"subagent_type": "tandem:gpt"}}),
    ("post", {"tool_name": "Agent", "tool_input": {"subagent_type": "other", "nested": {"subagent_type": "tandem:gpt"}}}),
    ("post", {"tool_name": "Agent", "tool_input": {"prompt": '{"subagent_type":"tandem:gpt"}'}}),
    ("stop", {"agent_type": "other"}),
])
def test_ordinary_scope_never_invokes_old_cli(tmp_path, phase, payload):
    result = invoke(tmp_path, phase, payload, "raise RuntimeError('must not run')\n")
    assert result.returncode == 0
    assert result.stdout == result.stderr == b""


@pytest.mark.parametrize("phase", ["pre", "stop", "post"])
@pytest.mark.parametrize("cli", [None, "raise SystemExit(2)\n", "raise SystemExit(1)\n"])
def test_relay_scope_reports_missing_or_failed_cli(tmp_path, phase, cli):
    payload = {"agent_type": "tandem:gpt", "tool_name": "Agent", "tool_input": {"subagent_type": "tandem:gpt"}}
    result = invoke(tmp_path, phase, payload, cli)
    assert result.returncode == 0
    response = json.loads(result.stdout)
    if phase == "pre":
        assert response["hookSpecificOutput"]["permissionDecision"] == "deny"
    elif phase == "stop":
        assert response["decision"] == "block"
    else:
        assert response["hookSpecificOutput"]["additionalContext"].startswith("UNVERIFIED CODEX RELAY:")


@pytest.mark.parametrize("raw", [b"broken", b"[]", b"null"])
def test_malformed_native_input_fails_closed(tmp_path, raw):
    result = invoke(tmp_path, "pre", raw)
    assert result.returncode == 0
    assert json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_relay_command_forwards_stdin_and_output_without_shell(tmp_path):
    raw = b'{"agent_type":"tandem:gpt","tool_input":{"command":"$(printf never); opaque"}}\r\n'
    output = b'{"hookSpecificOutput":{"hookEventName":"PreToolUse","updatedInput":{"message":"  exact\\r\\n\\n"}}}\r\n'
    seen = tmp_path / "seen.json"
    cli = (
        "import json,sys\n"
        f"open({str(seen)!r},'w').write(json.dumps([sys.argv[1:],sys.stdin.buffer.read().decode()]))\n"
        f"sys.stdout.buffer.write({output!r})\n"
    )
    result = invoke(tmp_path, "pre", raw, cli)
    assert result.returncode == 0
    assert result.stdout == output
    assert json.loads(seen.read_text()) == [["hook-relay", "pre"], raw.decode()]


@pytest.mark.parametrize("stdout", [b"not JSON", b"[]", b"null"])
def test_invalid_guard_output_fails_closed(tmp_path, stdout):
    result = invoke(tmp_path, "pre", {"agent_type": "tandem:codex-worker"},
                    f"import sys\nsys.stdout.buffer.write({stdout!r})\n")
    assert json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_missing_scope_interpreter_is_a_blocking_hook_failure(tmp_path):
    from tandem.relayguard import PRE_HOOK_COMMAND

    result = subprocess.run(
        ["/bin/sh", "-c", PRE_HOOK_COMMAND], input=b'{"agent_type":"tandem:gpt"}',
        capture_output=True,
        env={**os.environ, "PATH": str(tmp_path), "CLAUDE_PLUGIN_ROOT": str(SCRIPT.parent.parent)},
    )
    assert result.returncode == 2
