import json

import pytest

from jspace.profiles import get_profile
from jspace.scale_config import (
    CONFIG_PATH,
    REPO_ROOT,
    load_scale_config,
    resolved_config_path,
    transferred_band,
)

V3_PATH = REPO_ROOT / "extension" / "config_v3.json"


def test_config_path_defaults_to_frozen_v2(monkeypatch):
    monkeypatch.delenv("JSPACE_SCALE_CONFIG", raising=False)
    assert resolved_config_path() == CONFIG_PATH


def test_config_path_env_override(monkeypatch):
    monkeypatch.setenv("JSPACE_SCALE_CONFIG", "extension/config_v3.json")
    assert resolved_config_path() == V3_PATH
    config = load_scale_config()
    assert config["models"]["primary"] == ["qwen3-32b"]
    assert config["models"]["secondary"] == ["qwen3-1.7b"]
    assert config["measure_noncoherence_unconditionally"] is True


def test_v3_shares_every_scientific_block_with_v2():
    with CONFIG_PATH.open() as handle:
        v2 = json.load(handle)
    with V3_PATH.open() as handle:
        v3 = json.load(handle)
    # Identical data/intervention/thresholds keep the staged-data digest and the
    # transferred-band rule byte-compatible with the frozen v2 protocol.
    assert v3["data"] == v2["data"]
    assert v3["intervention"] == v2["intervention"]
    assert v3["thresholds"] == v2["thresholds"]
    assert v3["schema_version"] == 2
    assert "qwen3-8b" in v3["models"]["excluded"]
    assert "qwen3-14b" in v3["models"]["excluded"]


def test_v2_config_has_no_unconditional_flag():
    with CONFIG_PATH.open() as handle:
        v2 = json.load(handle)
    assert "measure_noncoherence_unconditionally" not in v2


def test_v3_band_transfer_and_profiles(monkeypatch):
    monkeypatch.setenv("JSPACE_SCALE_CONFIG", "extension/config_v3.json")
    config = load_scale_config()
    small = get_profile("qwen3-1.7b")
    large = get_profile("qwen3-32b")
    assert (small.n_layers, small.hidden_size) == (28, 2048)
    assert (large.n_layers, large.hidden_size) == (64, 5120)
    assert transferred_band(small.n_layers, config) == (10, 14)
    assert transferred_band(large.n_layers, config) == (24, 34)
    # The frozen v2 mappings must be unaffected by the new profiles.
    assert transferred_band(36, config) == (14, 19)
    assert transferred_band(40, config) == (15, 21)


def test_unknown_profile_still_rejected():
    with pytest.raises(ValueError):
        get_profile("qwen3-0.6b")


def test_v4_config_and_lens_override(monkeypatch):
    import json

    with (REPO_ROOT / "extension" / "config_v4.json").open() as handle:
        v4 = json.load(handle)
    with CONFIG_PATH.open() as handle:
        v2 = json.load(handle)
    assert v4["models"]["primary"] == ["qwen3-32b"]
    assert v4["models"]["secondary"] == []
    assert v4["measure_noncoherence_unconditionally"] is True
    assert v4["data"] == v2["data"]
    assert v4["thresholds"] == v2["thresholds"]
    assert v4["intervention"] == v2["intervention"]

    from jspace.scale_runtime import lens_override_path

    monkeypatch.delenv("JSPACE_LENS_FILE", raising=False)
    assert lens_override_path() is None
    monkeypatch.setenv("JSPACE_LENS_FILE", "extension/state/lens_fit/artifact/x.pt")
    assert lens_override_path() == REPO_ROOT / "extension/state/lens_fit/artifact/x.pt"
