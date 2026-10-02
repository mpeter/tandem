"""Scope plugin relay hooks before calling the installed Tandem CLI."""

import json
import subprocess
import sys


RELAY_TYPES = ("tandem:gpt", "tandem:codex-worker")


def failure(phase: str, reason: str) -> None:
    response: dict[str, object]
    if phase == "pre":
        response = {"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }}
    elif phase == "stop":
        response = {"decision": "block", "reason": reason}
    else:
        response = {"hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": "UNVERIFIED CODEX RELAY: " + reason,
        }}
    print(json.dumps(response))


def main(phase: str) -> None:
    raw = sys.stdin.buffer.read()
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Expected a hook object")
    except ValueError:
        failure(phase, "Cannot validate relay hook input.")
        return
    if phase == "post":
        tool_input = payload.get("tool_input")
        scoped = (payload.get("tool_name") in ("Agent", "Task")
                  and isinstance(tool_input, dict)
                  and tool_input.get("subagent_type") in RELAY_TYPES)
    else:
        scoped = payload.get("agent_type") in RELAY_TYPES
    if not scoped:
        return
    reason = "Compatible Tandem relay guard unavailable; update the CLI and plugin together."
    try:
        result = subprocess.run(
            ["tandem", "hook-relay", phase], input=raw, capture_output=True,
        )
    except OSError:
        failure(phase, reason)
        return
    sys.stderr.buffer.write(result.stderr)
    if result.returncode:
        failure(phase, reason)
        return
    if result.stdout:
        try:
            if not isinstance(json.loads(result.stdout), dict):
                raise ValueError("Expected guard output")
        except ValueError:
            failure(phase, reason)
            return
    sys.stdout.buffer.write(result.stdout)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("pre", "stop", "post"):
        raise SystemExit(2)
    main(sys.argv[1])
