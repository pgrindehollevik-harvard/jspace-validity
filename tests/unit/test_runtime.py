"""Fast, model-free checks for portable runtime configuration."""

import pytest

from jspace import core, scale_runtime
from jspace.profiles import COMPANION_REVISION, LENS_REVISION, get_profile
from jspace.scale_config import config_sha256, confirmatory_profiles, transferred_band


def test_cpu_can_be_requested_explicitly(monkeypatch):
    monkeypatch.delenv("JSPACE_DEVICE", raising=False)
    assert core.resolve_device("cpu") == "cpu"


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="unsupported JSPACE_DEVICE"):
        core.resolve_device("tpu")


def test_qwen14_profile_has_expected_shape_and_lens():
    profile = get_profile("QWEN3-14B")
    assert profile.n_layers == 40
    assert profile.hidden_size == 5120
    assert profile.lens_file.endswith("Qwen3-14B_jacobian_lens.pt")


def test_scale_profiles_and_bands_are_frozen():
    assert set(confirmatory_profiles()) == {"qwen3-8b", "qwen3-14b"}
    assert transferred_band(get_profile("qwen3-8b").n_layers) == (14, 19)
    assert transferred_band(get_profile("qwen3-14b").n_layers) == (15, 21)


def test_scale_artifacts_are_commit_pinned():
    assert LENS_REVISION == "0731326edff4ae730ffc5356fe1a4728c748b3a6"
    assert COMPANION_REVISION == "581d398613e5602a5af361e1c34d3a92ea82ba8e"
    assert get_profile("qwen3-8b").model_revision == ("b968826d9c46dd6066d109eabc6255188de91218")
    assert get_profile("qwen3-14b").model_revision == ("40c069824f4251a91eefaf281ebe4c544efd3e18")


def test_frozen_configuration_digest():
    assert config_sha256() == ("18581126da8946d50a7e5cef4d60cb2cc95af7807e25f3fa86b9422889a59b5f")


def test_intervention_constants_match_frozen_config():
    assert core.K_ABLATE == 10
    assert core.K_CLEAN_EXEMPT == 10


def test_companion_checkout_rejects_wrong_revision(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()

    def fake_check_output(command, **_kwargs):
        return "0" * 40 + "\n" if command[1:3] == ["rev-parse", "HEAD"] else ""

    monkeypatch.setattr(scale_runtime.subprocess, "check_output", fake_check_output)
    with pytest.raises(RuntimeError, match="revision mismatch"):
        scale_runtime.verify_companion_checkout(tmp_path)


def test_companion_checkout_rejects_modified_tracked_files(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()

    def fake_check_output(command, **_kwargs):
        return COMPANION_REVISION + "\n" if command[1:3] == ["rev-parse", "HEAD"] else " M data\n"

    monkeypatch.setattr(scale_runtime.subprocess, "check_output", fake_check_output)
    with pytest.raises(RuntimeError, match="modified tracked files"):
        scale_runtime.verify_companion_checkout(tmp_path)


def test_lens_loader_resolves_exact_cached_file_without_metadata_request(monkeypatch):
    profile = get_profile("qwen3-8b")
    calls = {}
    sentinel = object()

    def fake_download(repo_id, filename, *, revision, local_files_only):
        calls.update(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_files_only=local_files_only,
        )
        return "/cache/lens.pt"

    monkeypatch.setattr(scale_runtime, "hf_hub_download", fake_download)
    monkeypatch.setattr(scale_runtime.jlens.JacobianLens, "load", lambda path: sentinel)

    assert scale_runtime.load_scale_lens(profile, offline=True) is sentinel
    assert calls == {
        "repo_id": "neuronpedia/jacobian-lens",
        "filename": profile.lens_file,
        "revision": LENS_REVISION,
        "local_files_only": True,
    }
