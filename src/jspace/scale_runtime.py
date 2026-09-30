"""Pinned model/lens loader used only by the version-2 extension."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import jlens
import torch
import transformers
from huggingface_hub import hf_hub_download

from jspace import core
from jspace.profiles import (
    COMPANION_REPO,
    COMPANION_REVISION,
    LENS_REPO,
    LENS_REVISION,
    Qwen3Profile,
    get_profile,
)
from jspace.scale_config import REPO_ROOT, confirmatory_profiles, transferred_band


def verify_companion_checkout(path: Path | None = None) -> dict[str, str]:
    """Reject missing, modified, or incorrectly pinned evaluation data."""
    path = path or REPO_ROOT / "vendor" / "jacobian-lens"
    if not (path / ".git").exists():
        raise RuntimeError(f"missing companion checkout at {path}")
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.PIPE
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=path,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"could not verify companion checkout at {path}") from exc
    if revision != COMPANION_REVISION:
        raise RuntimeError(
            f"companion revision mismatch: observed={revision}, expected={COMPANION_REVISION}"
        )
    if status:
        raise RuntimeError("companion checkout has modified tracked files")
    return {"repository": COMPANION_REPO, "revision": revision}


def lens_override_path() -> Path | None:
    """A registered local lens artifact (PREREGISTRATION_V4.md) instead of the Hub file.

    Absent (the frozen v2/v3 configurations), stock-lens resolution is unchanged.
    """
    override = os.environ.get("JSPACE_LENS_FILE")
    if not override:
        return None
    path = Path(override)
    return path if path.is_absolute() else REPO_ROOT / path


def load_scale_lens(profile: Qwen3Profile, offline: bool) -> jlens.JacobianLens:
    """Resolve the pinned file in the Hub cache, then load without repository metadata."""
    override = lens_override_path()
    if override is not None:
        return jlens.JacobianLens.load(str(override))
    lens_path = hf_hub_download(
        LENS_REPO,
        profile.lens_file,
        revision=LENS_REVISION,
        local_files_only=offline,
    )
    return jlens.JacobianLens.load(lens_path)


def load_scale_setup(
    profile_key: str, device: str | None = None, dtype: torch.dtype | None = None
) -> core.Setup:
    if profile_key not in confirmatory_profiles():
        raise ValueError(
            f"{profile_key!r} is outside the frozen confirmatory profiles {confirmatory_profiles()}"
        )
    profile = get_profile(profile_key)
    device = core.resolve_device(device)
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        profile.model_id,
        revision=profile.model_revision,
        dtype=dtype,
        local_files_only=offline,
    ).to(device)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(
        profile.model_id,
        revision=profile.model_revision,
        local_files_only=offline,
    )
    model = jlens.from_hf(hf, tok)
    lens = load_scale_lens(profile, offline)

    if len(model.layers) != profile.n_layers:
        raise RuntimeError(f"profile says {profile.n_layers} layers, loaded {len(model.layers)}")
    if model.d_model != profile.hidden_size:
        raise RuntimeError(f"profile says d_model={profile.hidden_size}, loaded {model.d_model}")
    band = transferred_band(profile.n_layers)
    missing = [layer for layer in range(band[0], band[1] + 1) if layer not in lens.jacobians]
    if missing:
        raise RuntimeError(f"pinned lens is missing registered band layers {missing}")
    return core.Setup(hf=hf, tok=tok, model=model, lens=lens, device=device)
