"""The relay executes Codex rather than doing its delegated task itself."""

import json
from uuid import uuid4

import pytest
from click.testing import CliRunner

from tandem import cli, relayguard


def command(brief="review this", sandbox=""):
    flag = f" --sandbox {sandbox}" if sandbox else ""
    return f"tandem sub -q{flag} <<'TANDEM_TASK_EOF'\n{brief}\nTANDEM_TASK_EOF"


def payload(tool="Bash", cmd=None):
    return {"agent_type": "tandem:gpt", "tool_name": tool,
            "tool_input": {"command": command() if cmd is None else cmd}}


@pytest.mark.parametrize("sandbox", ["", "read-only", "workspace-write"])
def test_literal_dispatch_accepts_opaque_brief(sandbox):
    brief = "tandem-model: gpt\n$(echo no) `cat secret` ; > file\n'quotes'"
    assert relayguard.dispatch_brief("Bash", {"command": command(brief, sandbox)}) == brief


@pytest.mark.parametrize("cmd", [
    "cat secret", command() + "; cat secret", command() + "\ncat secret",
    command() + "\n\n", command().replace("<<'TANDEM_TASK_EOF'", "<<TANDEM_TASK_EOF"),
    "X=1 " + command(), "echo $(" + command() + ")",
    command().replace(" -q ", " -q --model other "),
    command("before\nTANDEM_TASK_EOF\ncat secret"),
    command().replace(" <<", " & <<"), command(""),
])
def test_shell_programs_are_denied(cmd):
    result = relayguard.decision(payload(cmd=cmd), "pre")
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize("tool", ["Read", "Agent", "WebSearch", "Edit"])
def test_non_dispatch_tools_are_denied(tool):
    assert relayguard.decision(payload(tool), "pre")["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_background_dispatch_is_denied():
    value = payload()
    value["tool_input"]["run_in_background"] = True
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize("mode,sandbox", [("default", "read-only"), ("acceptEdits", "workspace-write")])
def test_default_dispatch_gets_explicit_sandbox(tmp_path, mode, sandbox):
    value = payload()
    value["agent_transcript_path"] = transcript(tmp_path)["agent_transcript_path"]
    value["permission_mode"] = mode
    updated = relayguard.decision(value, "pre")["hookSpecificOutput"]["updatedInput"]
    assert updated["command"] == command(sandbox=sandbox)


def test_explicit_write_retry_requires_approval_without_write_mode(tmp_path):
    value = payload(cmd=command(sandbox="workspace-write"))
    value["agent_transcript_path"] = transcript(tmp_path)["agent_transcript_path"]
    value["permission_mode"] = "default"
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["permissionDecision"] == "ask"
    value["permission_mode"] = "acceptEdits"
    assert relayguard.decision(value, "pre") is None


def transcript(tmp_path, output=None, code=0, brief="review this", error=False):
    path = tmp_path / "agent.jsonl"
    _, worker_task = relayguard.modelcat.split_model_header(brief.strip())
    result = output if output is not None else relayguard.receipt(str(uuid4()), "gpt", code, worker_task.strip())
    entries = [
        {"type": "user", "message": {"content": brief}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "call", "name": "Bash", "input": {"command": command(brief)}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "call", "content": result, "is_error": error}]}},
    ]
    path.write_text("\n".join(map(json.dumps, entries)) + "\n")
    return {"agent_type": "tandem:gpt", "hook_event_name": "SubagentStop",
            "agent_transcript_path": str(path), "last_assistant_message": result}


def test_completed_worker_receipt_allows_stop_and_handback(tmp_path):
    value = transcript(tmp_path)
    assert relayguard.decision(value, "stop") is None
    value["tool_name"] = "SubagentHandback"
    value["tool_input"] = {"message": value["last_assistant_message"]}
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["updatedInput"]["message"] == value["last_assistant_message"]


@pytest.mark.parametrize("code,error", [(1, False), (None, True)])
def test_failed_dispatch_requires_explicit_failure_report(tmp_path, code, error):
    value = transcript(tmp_path, code=code or 0, error=error,
                       output="Exit code 1: bad model" if error else None)
    assert relayguard.decision(value, "stop")["decision"] == "block"
    value["last_assistant_message"] = "[tandem-sub failed]\n" + value["last_assistant_message"]
    assert relayguard.decision(value, "stop") is None


@pytest.mark.parametrize("output", ["I did the task myself", "[tandem-sub receipt] {}",
                                      relayguard.receipt(str(uuid4()), "gpt", 0, "different brief")])
def test_invalid_or_unrelated_receipt_blocks_stop(tmp_path, output):
    value = transcript(tmp_path, output=output)
    assert relayguard.decision(value, "stop")["decision"] == "block"


def test_receipt_in_task_is_not_worker_evidence(tmp_path):
    value = transcript(tmp_path)
    path = tmp_path / "agent.jsonl"
    path.write_text(json.dumps({"type": "user", "message": {"content": value["last_assistant_message"]}}) + "\n")
    value["stop_hook_active"] = True
    assert relayguard.decision(value, "stop")["decision"] == "block"
    value["tool_name"] = "SubagentHandback"
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_main_transcript_is_never_used_as_subagent_evidence(tmp_path):
    value = transcript(tmp_path)
    value["transcript_path"] = value.pop("agent_transcript_path")
    assert relayguard.decision(value, "stop")["decision"] == "block"


def test_handback_derives_agent_transcript_path(tmp_path):
    value = transcript(tmp_path)
    derived = tmp_path / "main" / "subagents"
    derived.mkdir(parents=True)
    (tmp_path / "agent.jsonl").rename(derived / "agent-id123.jsonl")
    value.pop("agent_transcript_path")
    value.update(transcript_path=str(tmp_path / "main.jsonl"), agent_id="id123", tool_name="SubagentHandback", tool_input={"message": value["last_assistant_message"]})
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["updatedInput"]["message"] == value["last_assistant_message"]


@pytest.mark.parametrize("phase", ["pre", "stop"])
def test_global_hook_does_not_affect_other_agents(phase):
    value = payload("Read")
    value["agent_type"] = "Explore"
    assert relayguard.decision(value, phase) is None
    value.pop("agent_type")
    assert relayguard.decision(value, phase) is None


def test_cli_malformed_hook_input_fails_closed():
    result = CliRunner().invoke(cli.main, ["hook-relay", "pre"], input="[]")
    assert result.exit_code == 2


def test_cli_denies_native_task_attempt():
    result = CliRunner().invoke(cli.main, ["hook-relay", "pre"], input=json.dumps(payload(cmd="cat secret")))
    assert result.exit_code == 0
    assert json.loads(result.output)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_parent_backstop_marks_missing_evidence_unverified():
    value = {"tool_name": "Agent", "tool_input": {"subagent_type": "tandem:gpt"},
             "tool_response": {"content": [{"type": "text", "text": "I did this myself"}]}}
    result = relayguard.decision(value, "post")
    assert "UNVERIFIED CODEX RELAY" in result["hookSpecificOutput"]["additionalContext"]
    value["tool_input"]["subagent_type"] = "Explore"
    assert relayguard.decision(value, "post") is None


def test_parent_backstop_marks_async_dispatch_pending():
    value = {"tool_name": "Agent", "tool_input": {"subagent_type": "tandem:gpt"},
             "tool_response": {"agentId": "worker1", "isAsync": True, "status": "async_launched"}}
    result = relayguard.decision(value, "post")
    assert "pending and unverified" in result["hookSpecificOutput"]["additionalContext"]


def test_parent_backstop_accepts_worker_evidence(tmp_path):
    own = transcript(tmp_path)
    value = {"tool_name": "Agent", "tool_input": {"subagent_type": "tandem:gpt"},
             "tool_response": {"agent_transcript_path": own["agent_transcript_path"],
                               "content": own["last_assistant_message"]}}
    assert relayguard.decision(value, "post") is None


def test_followup_task_cannot_reuse_old_worker_receipt(tmp_path):
    value = transcript(tmp_path)
    path = tmp_path / "agent.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps({"type": "user", "message": {"content": "Retry with write access"}}) + "\n")
    assert relayguard.decision(value, "stop")["decision"] == "block"
    value["tool_name"] = "SubagentHandback"
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_hook_feedback_preserves_current_task_receipt(tmp_path):
    value = transcript(tmp_path)
    path = tmp_path / "agent.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps({"type": "user", "message": {"content": "Stop hook feedback: return the receipt"}}) + "\n")
    assert relayguard.decision(value, "stop") is None


@pytest.mark.parametrize("message,blocked", [
    ("<system-reminder>Use handback</system-reminder>", False),
    ("<system-reminder>Use handback</system-reminder>\nRetry with write access", True),
])
def test_reminder_wrapper_does_not_hide_followup(tmp_path, message, blocked):
    value = transcript(tmp_path)
    with (tmp_path / "agent.jsonl").open("a") as stream:
        stream.write(json.dumps({"type": "user", "message": {"content": message}}) + "\n")
    assert (relayguard.decision(value, "stop") is not None) == blocked


def test_failed_worker_handback_supplies_canonical_failure(tmp_path):
    value = transcript(tmp_path, code=1)
    value["tool_name"] = "SubagentHandback"
    output = value.pop("last_assistant_message")
    value["tool_input"] = {"message": "Incorrect success claim"}
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["updatedInput"]["message"] == "[tandem-sub failed]\n" + output


@pytest.mark.parametrize("brief", ["shorter task", "tandem-model: other\nreview this"])
def test_dispatch_cannot_replace_assigned_task(tmp_path, brief):
    value = transcript(tmp_path)
    value.update(tool_name="Bash", tool_input={"command": command(brief)})
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_receipt_for_substituted_task_cannot_verify_handoff(tmp_path):
    value = transcript(tmp_path, brief="substituted task")
    path = tmp_path / "agent.jsonl"
    lines = path.read_text().splitlines()
    lines[0] = json.dumps({"type": "user", "message": {"content": "assigned task"}})
    path.write_text("\n".join(lines) + "\n")
    assert relayguard.decision(value, "stop")["decision"] == "block"


def test_altered_success_handback_is_replaced_with_worker_output(tmp_path):
    value = transcript(tmp_path)
    value.update(tool_name="SubagentHandback", tool_input={"message": "A different answer\n" + value["last_assistant_message"]})
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["updatedInput"]["message"] == value["last_assistant_message"]


def test_success_final_text_must_equal_worker_output(tmp_path):
    value = transcript(tmp_path)
    value["last_assistant_message"] = "I improved the worker's answer"
    assert relayguard.decision(value, "stop")["decision"] == "block"


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("change", ["indentation", "blank_lines", "line_endings"])
def test_final_output_rejects_whitespace_loss(tmp_path, failed, change):
    output = "  exact answer\r\n" + relayguard.receipt(str(uuid4()), "gpt", int(failed), "review this") + "\r\n\r\n"
    value = transcript(tmp_path, output=output)
    altered = {"indentation": output.lstrip(), "blank_lines": output.rstrip(),
               "line_endings": output.replace("\r\n", "\n")}[change]
    value["last_assistant_message"] = ("[tandem-sub failed]\n" if failed else "") + altered
    assert relayguard.decision(value, "stop")["decision"] == "block"


@pytest.mark.parametrize("failed", [False, True])
def test_final_output_accepts_exact_canonical_whitespace(tmp_path, failed):
    output = "  exact answer\r\n" + relayguard.receipt(str(uuid4()), "gpt", int(failed), "review this") + "\r\n\r\n"
    value = transcript(tmp_path, output=output)
    value["last_assistant_message"] = ("[tandem-sub failed]\n" if failed else "") + output
    assert relayguard.decision(value, "stop") is None


@pytest.mark.parametrize("separator", ["", " ", "\r\n", "\n\n"])
def test_final_failure_prefix_requires_exact_newline(tmp_path, separator):
    value = transcript(tmp_path, code=1)
    value["last_assistant_message"] = "[tandem-sub failed]" + separator + value["last_assistant_message"]
    assert relayguard.decision(value, "stop")["decision"] == "block"


@pytest.mark.parametrize("failed", [False, True])
def test_altered_handback_cannot_mark_worker_output_delivered(tmp_path, failed):
    output = "  exact answer\n" + relayguard.receipt(str(uuid4()), "gpt", int(failed), "review this") + "\n\n"
    value = transcript(tmp_path, output=output)
    report = ("[tandem-sub failed]\n" if failed else "") + output.strip()
    entries = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "handback", "name": "SubagentHandback", "input": {"message": report}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "handback", "content": "Delivered", "is_error": False}]}},
    ]
    with (tmp_path / "agent.jsonl").open("a") as stream:
        stream.write("\n".join(map(json.dumps, entries)) + "\n")
    value["last_assistant_message"] = "Done."
    assert relayguard.decision(value, "stop")["decision"] == "block"


def test_delivered_handback_allows_modern_closing_text(tmp_path):
    value = transcript(tmp_path)
    path = tmp_path / "agent.jsonl"
    entries = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "handback", "name": "SubagentHandback", "input": {"message": value["last_assistant_message"]}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "handback", "content": "Delivered", "is_error": False}]}},
    ]
    with path.open("a") as stream:
        stream.write("\n".join(map(json.dumps, entries)) + "\n")
    value["last_assistant_message"] = "Done."
    assert relayguard.decision(value, "stop") is None
    value.update(tool_name="SubagentHandback", tool_input={"message": "altered follow-on report"})
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["updatedInput"]["message"] == entries[0]["message"]["content"][0]["input"]["message"]


def test_dispatch_cannot_drop_assigned_model_header(tmp_path):
    value = transcript(tmp_path, brief="tandem-model: gpt\nreview this")
    value.update(tool_name="Bash", tool_input={"command": command("review this")})
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize("control", [
    "tandem-retry: workspace-write", "Retry with write access",
    "Retry the same task with write access", " Please retry with write access. ",
    "PLEASE RETRY THE SAME TASK WITH WRITE ACCESS!",
])
def test_retry_control_preserves_original_brief_but_requires_new_worker(tmp_path, control):
    brief = "tandem-model: gpt\nReview this exact original task."
    value = transcript(tmp_path, brief=brief)
    assert relayguard.decision(value, "stop") is None  # original success is valid
    path = tmp_path / "agent.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps({"type": "user", "message": {"content": control}}) + "\n")
    assert relayguard.assigned_task(value) == brief
    assert relayguard.decision(value, "stop")["decision"] == "block"
    value.update(tool_name="Bash", permission_mode="default",
                 tool_input={"command": command(brief, "workspace-write")})
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["permissionDecision"] == "ask"
    value["permission_mode"] = "acceptEdits"
    assert relayguard.decision(value, "pre") is None
    value["tool_input"] = {"command": command(control.strip(), "workspace-write")}
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["permissionDecision"] == "deny"
    fresh = relayguard.receipt(str(uuid4()), "gpt", 0, "Review this exact original task.")
    entries = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "retry", "name": "Bash", "input": {"command": command(brief, "workspace-write")}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "retry", "content": fresh}]}},
    ]
    with path.open("a") as stream:
        stream.write("\n".join(map(json.dumps, entries)) + "\n")
    value.update(tool_name="SubagentHandback", tool_input={"message": fresh})
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["updatedInput"]["message"] == fresh


def test_other_followup_remains_new_assignment(tmp_path):
    value = transcript(tmp_path)
    text = "Retry with write access and also change the report format"
    with (tmp_path / "agent.jsonl").open("a") as stream:
        stream.write(json.dumps({"type": "user", "message": {"content": text}}) + "\n")
    assert relayguard.assigned_task(value) == text
    value.update(tool_name="Bash", tool_input={"command": command()})
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_stale_handback_result_cannot_deliver_newer_worker_output(tmp_path):
    value = transcript(tmp_path)
    newer = "new answer\n" + relayguard.receipt(str(uuid4()), "gpt", 0, "review this")
    entries = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "old-handback", "name": "SubagentHandback", "input": {"message": value["last_assistant_message"]}}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "new-worker", "name": "Bash", "input": {"command": command()}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "new-worker", "content": newer}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "old-handback", "content": "Delivered", "is_error": False}]}},
    ]
    with (tmp_path / "agent.jsonl").open("a") as stream:
        stream.write("\n".join(map(json.dumps, entries)) + "\n")
    value["last_assistant_message"] = "Done."
    assert relayguard.decision(value, "stop")["decision"] == "block"
    value.update(tool_name="SubagentHandback", tool_input={"message": newer})
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["updatedInput"]["message"] == newer


def test_handback_update_preserves_every_worker_footer_and_whitespace(tmp_path):
    output = "  exact answer\n\n" + relayguard.receipt(str(uuid4()), "gpt", 0, "review this") + "\n\n[tandem-sub model: gpt]\n"
    value = transcript(tmp_path, output=output)
    value.update(tool_name="SubagentHandback", tool_input={"message": "exact answer"})
    assert relayguard.decision(value, "pre")["hookSpecificOutput"]["updatedInput"]["message"] == output


@pytest.mark.parametrize("hook_command", ["tandem hook-relay pre", relayguard.PRE_HOOK_COMMAND])
def test_corrected_handback_hook_evidence_allows_modern_stop(tmp_path, hook_command):
    value = transcript(tmp_path)
    output = value["last_assistant_message"]
    entries = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "corrected-handback", "name": "SubagentHandback", "input": {"message": "Incomplete copy"}}]}},
        {"type": "attachment", "attachment": {
            "type": "hook_success", "hookName": "PreToolUse:SubagentHandback",
            "command": hook_command, "toolUseID": "corrected-handback",
            "stdout": json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": {"message": output}}}),
        }},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "corrected-handback", "content": "Delivered", "is_error": False}]}},
    ]
    with (tmp_path / "agent.jsonl").open("a") as stream:
        stream.write("\n".join(map(json.dumps, entries)) + "\n")
    value["last_assistant_message"] = "Done."
    assert relayguard.decision(value, "stop") is None


def test_foreign_hook_command_cannot_certify_corrected_handback(tmp_path):
    value = transcript(tmp_path)
    output = value["last_assistant_message"]
    entries = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "handback", "name": "SubagentHandback", "input": {"message": "Incomplete copy"}}]}},
        {"type": "attachment", "attachment": {
            "type": "hook_success", "hookName": "PreToolUse:SubagentHandback",
            "command": "foreign " + relayguard.PRE_HOOK_COMMAND, "toolUseID": "handback",
            "stdout": json.dumps({"hookSpecificOutput": {"updatedInput": {"message": output}}}),
        }},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "handback", "content": "Delivered"}]}},
    ]
    with (tmp_path / "agent.jsonl").open("a") as stream:
        stream.write("\n".join(map(json.dumps, entries)) + "\n")
    value["last_assistant_message"] = "Done."
    assert relayguard.decision(value, "stop")["decision"] == "block"


def test_native_turn_companion_preserves_delivered_handback(tmp_path):
    value = transcript(tmp_path)
    output = value["last_assistant_message"]
    entries = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "handback", "name": "SubagentHandback", "input": {"message": output}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "handback", "content": [{"type": "text", "text": '{"success":true,"message":"Report delivered to your caller."}'}]}]}},
        {"type": "user", "isMeta": True, "turnCompanion": True, "message": {"content": "[Your previous response had no visible output. Please continue and produce a user-visible response.]"}},
    ]
    with (tmp_path / "agent.jsonl").open("a") as stream:
        stream.write("\n".join(map(json.dumps, entries)) + "\n")
    value["last_assistant_message"] = "Done."
    assert relayguard.assigned_task(value) == "review this"
    assert relayguard.decision(value, "stop") is None


@pytest.mark.parametrize("metadata", [{}, {"isMeta": True}, {"turnCompanion": True}])
def test_real_followup_after_handback_remains_new_assignment(tmp_path, metadata):
    value = transcript(tmp_path)
    text = "Review the new task instead."
    with (tmp_path / "agent.jsonl").open("a") as stream:
        stream.write(json.dumps({"type": "user", **metadata, "message": {"content": text}}) + "\n")
    assert relayguard.assigned_task(value) == text
    assert relayguard.decision(value, "stop")["decision"] == "block"
