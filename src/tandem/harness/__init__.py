from .base import HarnessAdapter
from .claude_code import ClaudeCodeAdapter
from .codex import CodexAdapter
from .opencode import OpencodeAdapter

ADAPTERS: dict[str, HarnessAdapter] = {
    "claude": ClaudeCodeAdapter(),
    "codex": CodexAdapter(),
    "opencode": OpencodeAdapter(),
}


def get_adapter(harness_id: str) -> HarnessAdapter:
    adapter = ADAPTERS[harness_id]
    if harness_id == "opencode" and type(adapter) is OpencodeAdapter:
        from .. import compat

        version = compat.parse_version(adapter.detect_version() or "")
        if version and version[0] == 2:
            from .opencode2 import Opencode2Adapter

            return Opencode2Adapter()
    return adapter
