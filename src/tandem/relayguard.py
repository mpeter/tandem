"""Agent-scoped enforcement for the Claude-to-Codex relay."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from uuid import UUID

from . import modelcat
from .hookroute import sandbox_for_mode

_COMMAND = re.compile(
    r"tandem sub -q(?: --sandbox (?:read-only|workspace-write))? <<'"
    r"(TANDEM_TASK_EOF(?:_[0-9]+)?)'\n"
)
_RECEIPT = "[tandem-sub receipt] "
PRE_HOOK_COMMAND = 'python3 "${CLAUDE_PLUGIN_ROOT}/hooks/relay-scope.py" pre || exit 2'
_RETRY = "Relay only: run tandem sub -q with the entire brief in a single quoted heredoc."


def dispatch_brief(tool_name: object, tool_input: object) -> str | None:
    """Accept one literal stdin dispatch, with no surrounding shell program."""
    if tool_name != "Bash" or not isinstance(tool_input, dict):
        return None
    if tool_input.get("run_in_background"):
        return None
    command = tool_input.get("command")
    if not isinstance(command, str):
        return None
    match = _COMMAND.match(command)
    if match is None:
        return None
    rest = command[match.end():].removesuffix("\n").split("\n")
    if not rest or rest[-1] != match[1] or match[1] in rest[:-1]:
        return None
    if "\r" in command or "\x00" in command:
        return None
    suffix = "\n" + match[1]
    if not command.endswith(suffix) and not command.endswith(suffix + "\n"):
        return None
    brief = "\n".join(rest[:-1])
    return brief if brief.strip() else None


def task_digest(task: str) -> str:
    return hashlib.sha256(task.encode()).hexdigest()


def receipt(worker_id: str, model: str, code: int, task: str) -> str:
    return _RECEIPT + json.dumps({
        "version": 1, "worker_id": worker_id, "model": model,
        "exit_code": code, "task_sha256": task_digest(task),
    }, sort_keys=True)


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text"
                         and isinstance(b.get("text"), str))
    return ""


def _receipt_code(output: str, brief: str) -> int | None:
    try:
        _, task = modelcat.split_model_header(brief.strip())
    except modelcat.MalformedHeader:
        return None
    digest = task_digest(task.strip())
    for line in reversed(output.splitlines()):
        if not line.startswith(_RECEIPT):
            continue
        try:
            value = json.loads(line[len(_RECEIPT):])
            if not isinstance(value, dict):
                continue
            UUID(value.get("worker_id", ""))
        except (ValueError, TypeError, AttributeError):
            continue
        code = value.get("exit_code")
        if (value.get("version") == 1 and isinstance(value.get("model"), str)
                and type(code) is int and value.get("task_sha256") == digest):
            return code
    return None


def _task_text(entry: dict) -> str | None:
    if entry.get("isMeta") is True and entry.get("turnCompanion") is True:
        return None
    content = entry.get("message", {}).get("content")
    if isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    ):
        return None
    text = _text(content).strip()
    # Claude injects hook feedback as user text; it continues the same task.
    while text.startswith("<system-reminder>"):
        end = text.find("</system-reminder>")
        if end < 0:
            break
        text = text[end + len("</system-reminder>"):].strip()
    return text if text and not text.startswith("Stop hook feedback:") else None


def _retry_control(text: str) -> bool:
    return text == "tandem-retry: workspace-write" or re.fullmatch(
        r"(?:Please )?Retry (?:the same task )?with write access[.!?]?",
        text, re.IGNORECASE,
    ) is not None


def _transcript_path(payload: dict) -> Path | None:
    path = payload.get("agent_transcript_path")
    if not path and payload.get("agent_id"):
        agent_id, main_path = payload.get("agent_id"), payload.get("transcript_path")
        if isinstance(agent_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", agent_id) and isinstance(main_path, str):
            main = Path(main_path).expanduser()
            path = str(main.parent / main.stem / "subagents" / f"agent-{agent_id}.jsonl")
    return Path(path).expanduser() if isinstance(path, str) and path else None


def assigned_task(payload: dict) -> str | None:
    path = _transcript_path(payload)
    if path is None:
        return None
    task = None
    try:
        with path.open() as stream:
            for line in stream:
                entry = json.loads(line)
                if entry.get("type") == "user":
                    text = _task_text(entry)
                    if text is not None and not _retry_control(text):
                        task = text
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return task


def _matches_output(message: object, output: str, failed: bool) -> bool:
    if not isinstance(message, str):
        return False
    canonical = "[tandem-sub failed]\n" + output if failed else output
    return message == canonical


def _worker_output(payload: dict) -> tuple[str | None, bool, str | None]:
    """Canonical current worker output, delivered flag, or validation error."""
    path = _transcript_path(payload)
    if path is None:
        return None, False, "Cannot verify the relay's own transcript. " + _RETRY
    calls: dict[str, str] = {}
    completed: list[tuple[int | None, bool, str]] = []
    task = None
    handbacks: dict[str, str] = {}
    handback_attempts: set[str] = set()
    delivered = False
    try:
        with path.open() as stream:
            for line in stream:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    return None, False, "Malformed relay transcript. " + _RETRY
                attachment = entry.get("attachment", {})
                if (entry.get("type") == "attachment" and isinstance(attachment, dict)
                        and attachment.get("type") == "hook_success"
                        and attachment.get("hookName") == "PreToolUse:SubagentHandback"
                        and attachment.get("command") in ("tandem hook-relay pre", PRE_HOOK_COMMAND)
                        and completed and not calls):
                    code, failed, output = completed[-1]
                    hook_output = json.loads(attachment.get("stdout", "{}"))
                    specific = hook_output.get("hookSpecificOutput", {})
                    updated = specific.get("updatedInput", {})
                    report = updated.get("message")
                    call_id = attachment.get("toolUseID")
                    if ((code is not None or failed) and isinstance(call_id, str)
                            and call_id in handback_attempts and _matches_output(report, output, failed or code != 0)):
                        handbacks[call_id] = output
                message = entry.get("message", {})
                if not isinstance(message, dict):
                    continue
                blocks = message.get("content")
                if entry.get("type") == "user":
                    text = _task_text(entry)
                    if text is not None:
                        if not _retry_control(text):
                            task = text
                        calls.clear()
                        completed.clear()
                        handbacks.clear()
                        handback_attempts.clear()
                        delivered = False
                if not isinstance(blocks, list):
                    continue
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use" and entry.get("type") == "assistant":
                        brief = dispatch_brief(block.get("name"), block.get("input"))
                        if brief is not None and brief.strip() == task and isinstance(block.get("id"), str):
                            calls[block["id"]] = brief
                        elif block.get("name") == "SubagentHandback" and completed and not calls:
                            code, failed, output = completed[-1]
                            if isinstance(block.get("id"), str):
                                handback_attempts.add(block["id"])
                            handback_input = block.get("input")
                            report = handback_input.get("message") if isinstance(handback_input, dict) else None
                            if ((code is not None or failed) and isinstance(block.get("id"), str)
                                    and _matches_output(report, output, failed or code != 0)):
                                handbacks[block["id"]] = output
                    elif block.get("type") == "tool_result" and entry.get("type") == "user":
                        call_id = block.get("tool_use_id")
                        brief = calls.pop(call_id, None) if isinstance(call_id, str) else None
                        if brief is not None:
                            output = _text(block.get("content"))
                            completed.append((_receipt_code(output, brief), block.get("is_error") is True, output))
                            handbacks.clear()
                            handback_attempts.clear()
                            delivered = False
                        elif isinstance(call_id, str) and call_id in handbacks:
                            handbacks.pop(call_id)
                            handback_attempts.discard(call_id)
                            delivered = block.get("is_error") is not True
    except (OSError, ValueError, TypeError, AttributeError):
        return None, False, "Cannot read a valid relay transcript. " + _RETRY
    if completed and not calls:
        code, failed, output = completed[-1]
        if code is not None or failed:
            canonical = "[tandem-sub failed]\n" + output if failed or code != 0 else output
            return canonical, delivered, None
    return None, False, "No completed Codex worker dispatch for the assigned task was verified. " + _RETRY


def stop_reason(payload: dict) -> str | None:
    output, delivered, reason = _worker_output(payload)
    if reason is not None:
        return reason
    if delivered or (output is not None and _matches_output(payload.get("last_assistant_message"), output, False)):
        return None
    return "Return the worker output unchanged, including its receipt; prefix failures with [tandem-sub failed]."


def decision(payload: dict, phase: str) -> dict | None:
    if phase == "post":
        tool_input = payload.get("tool_input")
        if payload.get("tool_name") not in {"Agent", "Task"} or not isinstance(tool_input, dict):
            return None
        agent_type = tool_input.get("subagent_type")
        if agent_type not in {"tandem:gpt", "tandem:codex-worker"}:
            return None
        response = payload.get("tool_response")
        if not isinstance(response, dict):
            response = {}
        if response.get("isAsync") or response.get("status") == "async_launched":
            context = "Codex relay is pending and unverified until its guarded handback includes a worker receipt."
        else:
            check = dict(payload)
            check.update(agent_id=response.get("agentId"),
                         agent_transcript_path=response.get("agent_transcript_path"),
                         last_assistant_message=_text(response.get("content")))
            reason = stop_reason(check)
            if reason is None:
                return None
            context = "UNVERIFIED CODEX RELAY: " + reason + " Do not present this report as verified Codex work."
        return {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": context}}
    # Plugin hooks are global; these guards apply only inside our relays.
    if payload.get("agent_type") not in {"tandem:gpt", "tandem:codex-worker"}:
        return None
    if phase == "pre":
        tool_name = payload.get("tool_name")
        if tool_name == "SubagentHandback":
            output, _, reason = _worker_output(payload)
            tool_input = payload.get("tool_input")
            if output is not None and isinstance(tool_input, dict):
                updated = dict(tool_input)
                updated["message"] = output
                return {"hookSpecificOutput": {
                    "hookEventName": "PreToolUse", "updatedInput": updated,
                }}
            reason = reason or "Cannot validate the relay handback input."
        elif dispatch_brief(tool_name, payload.get("tool_input")) is not None:
            tool_input = payload["tool_input"]
            command = tool_input["command"]
            first_line = command.split("\n", 1)[0]
            brief = dispatch_brief(tool_name, tool_input)
            task = assigned_task(payload)
            if task is None or brief is None or brief.strip() != task:
                return {"hookSpecificOutput": {
                    "hookEventName": "PreToolUse", "permissionDecision": "deny",
                    "permissionDecisionReason": "Dispatch the assigned task unchanged, including its model header. " + _RETRY,
                }}
            if " --sandbox " not in first_line:
                mode = sandbox_for_mode(payload.get("permission_mode"))
                updated = dict(tool_input)
                updated["command"] = command.replace("tandem sub -q", f"tandem sub -q --sandbox {mode}", 1)
                return {"hookSpecificOutput": {
                    "hookEventName": "PreToolUse", "updatedInput": updated,
                }}
            if " --sandbox workspace-write " in first_line and sandbox_for_mode(payload.get("permission_mode")) != "workspace-write":
                return {"hookSpecificOutput": {
                    "hookEventName": "PreToolUse", "permissionDecision": "ask",
                    "permissionDecisionReason": "Approve workspace-write access for this Codex worker retry.",
                }}
            return None
        else:
            reason = _RETRY
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }}
    reason = stop_reason(payload)
    return {"decision": "block", "reason": reason} if reason else None
