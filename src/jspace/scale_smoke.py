"""Profile-specific CUDA compatibility smoke test; writes no result artifact."""

from __future__ import annotations

import argparse
import json
import time

import torch

from jspace import core
from jspace.ablation import JSpaceAblator
from jspace.generate import generate
from jspace.profiles import get_profile
from jspace.scale_config import (
    confirmatory_profiles,
    load_scale_config,
    transferred_band,
)
from jspace.scale_runtime import load_scale_setup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=confirmatory_profiles(), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("scale smoke test requires an allocated CUDA GPU")

    profile = get_profile(args.model)
    config = load_scale_config()
    band = transferred_band(profile.n_layers, config)
    torch.cuda.reset_peak_memory_stats()
    setup = load_scale_setup(args.model, device="cuda")
    prompt = core.chat_prompt(setup, "What is 6 times 7?", "cot")

    started = time.perf_counter()
    clean = generate(setup, prompt, max_new_tokens=16, seed=17)
    clean_seconds = time.perf_counter() - started
    ablator = JSpaceAblator(setup, band, k=config["intervention"]["k"]).install()
    try:
        started = time.perf_counter()
        ablated = generate(setup, prompt, max_new_tokens=16, seed=17, ablator=ablator)
        ablated_seconds = time.perf_counter() - started
    finally:
        ablator.remove()

    print(
        json.dumps(
            {
                "profile": args.model,
                "model_revision": profile.model_revision,
                "gpu": torch.cuda.get_device_name(0),
                "band": band,
                "clean_tokens": clean.n_new,
                "clean_seconds": round(clean_seconds, 3),
                "ablated_tokens": ablated.n_new,
                "ablated_seconds": round(ablated_seconds, 3),
                "peak_cuda_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                "clean_preview": clean.text[:120],
                "ablated_preview": ablated.text[:120],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
