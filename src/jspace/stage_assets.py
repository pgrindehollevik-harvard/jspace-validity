"""Prefetch public model and lens artifacts onto a cluster cache.

This module performs no model inference and writes no experimental results.  It is
intended for cluster staging so GPU compute jobs can set HF_HUB_OFFLINE=1 and never depend
on external network access once they start.

Example:
    HF_HOME=$HOME/scratch/jspace-validity/.hf \
      .venv/bin/python -m jspace.stage_assets --model qwen3-8b
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from jspace.profiles import LENS_REPO, LENS_REVISION, QWEN3_PROFILES, get_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(QWEN3_PROFILES), required=True)
    args = parser.parse_args()
    profile = get_profile(args.model)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        Path(hf_home).mkdir(parents=True, exist_ok=True)

    print(f"staging model {profile.model_id}", flush=True)
    model_revision = profile.model_revision
    lens_revision = LENS_REVISION
    model_path = snapshot_download(profile.model_id, revision=model_revision)
    print(f"staging lens {LENS_REPO}:{profile.lens_file}", flush=True)
    lens_path = hf_hub_download(LENS_REPO, profile.lens_file, revision=lens_revision)
    print(
        json.dumps(
            {
                "profile": profile.key,
                "model_id": profile.model_id,
                "model_snapshot": model_path,
                "lens_file": profile.lens_file,
                "lens_path": lens_path,
                "hf_home": hf_home,
                "model_revision": model_revision,
                "lens_revision": lens_revision,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
