"""Registered completion of the Qwen3-32B Jacobian-lens fit (PREREGISTRATION_V4.md).

The upstream repository ships only an interrupted fit checkpoint for Qwen3-32B, so this
driver refits the lens from scratch with the released recipe at the pinned companion
code: wikitext-103 train prompts in dataset order (first eligible prompts, minimum 600
stripped characters, truncated to 2,000 characters), dim_batch 128, max_seq_len 128,
bfloat16, at most 1,000 prompts. Early stopping reimplements the released intent as a
registered block rule: after at least 100 prompts, stop at the first multiple of 10
where the running mean changed by less than 2 percent (max over layers) across the last
10 prompts, i.e. 0.2 percent per prompt, matching the released 0.002 per-prompt stop.

The fit checkpoints after every prompt (atomic) and resumes on requeue. The finished
lens is saved in the stock format with a manifest recording its SHA-256, prompt count,
recipe, and code commit. This driver writes no gate result and reads no registered
evaluation item; the eligible-prompt stream (ranks 0..999) is disjoint by construction
from the evaluation slice, which skips the first 2,000 eligible rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from jspace.profiles import get_profile
from jspace.scale_config import REPO_ROOT

FIT_MODEL = "qwen3-32b"
N_PROMPTS = 1000
MIN_CHARS = 600
MAX_CHARS = 2000
# The released recipe used dim_batch=128 on a 179 GB B200; the H200 (140 GB) OOMs at
# 128. Chunk width is mathematically equivalent at any value (deviation-logged).
DIM_BATCH = int(os.environ.get("JSPACE_FIT_DIM_BATCH", "64"))
MAX_SEQ_LEN = 128
BLOCK = 10
MIN_PROMPTS = 100
BLOCK_STOP_DELTA = 0.02  # 10 prompts x the released 0.002 per-prompt stop
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


PROMPTS_PATH = REPO_ROOT / "extension" / "state" / "lens_fit" / "fit_prompts.json"


def load_fit_prompts() -> list[str]:
    """First N_PROMPTS eligible wikitext-103 train rows, in dataset order, truncated.

    Offline GPU jobs read the login-node-staged file (the repository's standard
    staging pattern); staging itself resolves the pinned dataset revision online.
    """
    if PROMPTS_PATH.exists():
        with PROMPTS_PATH.open() as handle:
            prompts = json.load(handle)["prompts"]
        if len(prompts) != N_PROMPTS:
            raise RuntimeError(f"staged prompt file has {len(prompts)} prompts")
        return prompts

    from datasets import load_dataset

    rows = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        revision=WIKITEXT_REVISION,
        split="train",
    )
    prompts = []
    for row in rows:
        text = row["text"]
        if len(text.strip()) >= MIN_CHARS:
            prompts.append(text[:MAX_CHARS])
            if len(prompts) == N_PROMPTS:
                break
    if len(prompts) != N_PROMPTS:
        raise RuntimeError(f"only {len(prompts)} eligible fit prompts")
    return prompts


def prompts_sha256(prompts: list[str]) -> str:
    encoded = json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def stage_prompts() -> None:
    prompts = load_fit_prompts()
    PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROMPTS_PATH.open("w") as handle:
        json.dump({"prompts": prompts, "prompts_sha256": prompts_sha256(prompts)}, handle)
    print(f"staged {len(prompts)} prompts sha256={prompts_sha256(prompts)}", flush=True)


def max_relative_change(
    previous: dict[int, torch.Tensor] | None, current: dict[int, torch.Tensor]
) -> float:
    if previous is None:
        return float("inf")
    return max(
        ((current[layer] - previous[layer]).norm() / previous[layer].norm()).item()
        for layer in current
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    import jlens
    import transformers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[FIT_MODEL], default=FIT_MODEL)
    parser.add_argument(
        "--stage-prompts",
        action="store_true",
        help="Stage the fit prompt file on a login node (online) and exit.",
    )
    parser.add_argument(
        "--wall-seconds",
        type=float,
        default=float(os.environ.get("JSPACE_FIT_WALL_SECONDS", 82800)),
        help="Checkpoint and exit 99 when exceeded, for chained or requeued jobs.",
    )
    args = parser.parse_args()
    if args.stage_prompts:
        stage_prompts()
        return
    started = time.monotonic()

    profile = get_profile(args.model)
    state_dir = REPO_ROOT / "extension" / "state" / "lens_fit"
    state_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = state_dir / f"{profile.key}_fit_checkpoint.pt"
    out_dir = REPO_ROOT / "extension" / "state" / "lens_fit" / "artifact"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Qwen3-32B_jacobian_lens_refit.pt"
    manifest_path = out_dir / "Qwen3-32B_jacobian_lens_refit.json"

    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    prompts = load_fit_prompts()
    print(f"fit prompts: {len(prompts)} (eligible ranks 0..{len(prompts) - 1})", flush=True)

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        profile.model_id,
        revision=profile.model_revision,
        dtype=torch.bfloat16,
        local_files_only=offline,
    ).to("cuda")
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(
        profile.model_id, revision=profile.model_revision, local_files_only=offline
    )
    model = jlens.from_hf(hf, tok)

    previous_mean: dict[int, torch.Tensor] | None = None
    deltas: list[dict] = []
    lens = None
    for target in range(BLOCK, N_PROMPTS + 1, BLOCK):
        lens = jlens.fitting.fit(
            model,
            prompts[:target],
            dim_batch=DIM_BATCH,
            max_seq_len=MAX_SEQ_LEN,
            checkpoint_path=str(checkpoint_path),
            checkpoint_every=1,
            resume=True,
        )
        delta = max_relative_change(previous_mean, lens.jacobians)
        previous_mean = {layer: J.clone() for layer, J in lens.jacobians.items()}
        deltas.append({"prompts": target, "block_delta": delta})
        print(f"block to {target} prompts: block_delta={delta:.5f}", flush=True)
        if target >= MIN_PROMPTS and delta < BLOCK_STOP_DELTA:
            print(f"registered stop rule met at {target} prompts", flush=True)
            break
        if time.monotonic() - started > args.wall_seconds:
            print("wall limit reached; checkpoint retained for resume", flush=True)
            raise SystemExit(99)

    assert lens is not None
    lens.save(str(out_path))
    manifest = {
        "model_id": profile.model_id,
        "model_revision": profile.model_revision,
        "code_commit": commit,
        "wikitext_revision": WIKITEXT_REVISION,
        "recipe": {
            "n_prompts_cap": N_PROMPTS,
            "min_chars": MIN_CHARS,
            "max_chars": MAX_CHARS,
            "dim_batch": DIM_BATCH,
            "max_seq_len": MAX_SEQ_LEN,
            "dtype": "bfloat16",
            "compile": False,
            "block": BLOCK,
            "min_prompts": MIN_PROMPTS,
            "block_stop_delta": BLOCK_STOP_DELTA,
        },
        "prompts_sha256": prompts_sha256(prompts),
        "prompts_fitted": lens.n_prompts,
        "block_deltas": deltas,
        "lens_sha256": file_sha256(out_path),
    }
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=1)
        handle.write("\n")
    print(json.dumps(manifest, indent=1), flush=True)


if __name__ == "__main__":
    main()
