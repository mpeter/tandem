"""Build the window's runtime clients, one per participant."""

from __future__ import annotations

from .claude import ClaudeRuntime
from .codex import CodexRuntime
from .opencode import OpencodeRuntime

_CLASSES = {"claude": ClaudeRuntime, "codex": CodexRuntime, "opencode": OpencodeRuntime}


def make_runtimes(session, cfg) -> dict:
    from ... import compat

    classes = dict(_CLASSES)
    if "opencode" in session.participants and compat.opencode_major() == 2:
        from .opencode2 import Opencode2Runtime

        classes["opencode"] = Opencode2Runtime
    return {h: classes[h](cfg) for h in session.participants if h in classes}
