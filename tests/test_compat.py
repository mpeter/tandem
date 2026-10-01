from tandem import compat


def test_opencode_v1_versions_keep_the_floor_only_policy():
    assert not compat.version_supported("opencode", "1.17.99")
    assert compat.version_supported("opencode", "1.18.20")
    assert compat.version_supported("opencode", "1.99.99")


def test_opencode_two_has_a_verified_floor():
    version = "opencode v2.0.21"
    assert compat.version_supported("opencode", version)
    assert not compat.version_supported("opencode", "2.0.20")
    assert compat.hard_rejection_reason("opencode", version) is None


def test_opencode_future_majors_remain_hard_rejected():
    assert not compat.version_supported("opencode", "3.0.0")
    assert compat.hard_rejection_reason("opencode", "3.0.0") is not None


def test_participant_resolution_rejects_future_opencode_before_runtime_probe(
        monkeypatch, capsys):
    from tandem import cli

    class Adapter:
        def __init__(self, harness, version):
            self.id = harness
            self.display_name = harness
            self.binary = harness
            self.install_hint = ""
            self.version = version
            self.runtime_probed = False

        def detect_version(self):
            return self.version

        def version_supported(self, version):
            return compat.version_supported(self.id, version)

        def runtime_ready(self):
            self.runtime_probed = True
            return True, ""

    adapters = {
        "claude": Adapter("claude", "2.1.265"),
        "codex": Adapter("codex", "0.153.4"),
        "opencode": Adapter("opencode", "3.0.0"),
    }
    monkeypatch.setattr("tandem.config.load_harnesses",
                        lambda: ["claude", "codex", "opencode"])
    monkeypatch.setattr(cli, "get_adapter", adapters.__getitem__)

    usable, _ = cli._resolve_participants(warn_only=True)

    assert usable == ["claude", "codex"]
    assert not adapters["opencode"].runtime_probed
    assert "OpenCode 3 is unsupported" in capsys.readouterr().err
