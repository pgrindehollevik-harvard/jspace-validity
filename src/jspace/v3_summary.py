"""Registered summary of the version-3 results (extension/PREREGISTRATION_V3.md).

Committed before any v3 result artifact exists. It evaluates exactly the registered
endpoints: H1 anti-selectivity per model (exact two-sided sign test on the 50 paired
per-sequence WikiText deltas at alpha .025, and a 10,000-draw paired sequence
percentile-bootstrap 95% CI for the selectivity index that must lie above 1), the
complete-gate verdicts (H2/H3) read from the raw artifacts, and the conditional H4
flip-core overlap test. The locked v2 analysis is untouched.
"""

from __future__ import annotations

import argparse
import json
from random import Random

from jspace.frozen_explore import (
    BOOTSTRAP_SEED,
    bootstrap_selectivity_index,
    downflips,
    overlap_tail_p,
    selectivity_index,
    sign_test_p,
)
from jspace.scale_analyze import bootstrap_mean, bootstrap_paired_difference

V3_MODELS = ("qwen3-1.7b", "qwen3-32b")
H1_ALPHA = 0.025
CORE_4B = (2, 14, 17, 18, 34, 35, 39, 46, 48)


def summarize_model(result: dict) -> dict:
    metrics = result["metrics"]
    row: dict = {
        "profile": result["profile"],
        "band": result["band"],
        "passes_gate": result["passes_gate"],
        "status": result["status"],
        "stop_reason": result["stop_reason"],
        "checks": result["checks"],
        "lens_gate": (result.get("lens_validation") or {}).get("gate"),
    }
    wiki_j = metrics.get("wikitext_jspace", {}).get("per_sequence_agreement")
    wiki_r = metrics.get("wikitext_random", {}).get("per_sequence_agreement")
    if wiki_j and wiki_r:
        rng = Random(BOOTSTRAP_SEED)
        deltas = [j - r for j, r in zip(wiki_j, wiki_r)]
        worse = sum(1 for d in deltas if d < 0)
        better = sum(1 for d in deltas if d > 0)
        p_sign = sign_test_p(worse, better)
        si = selectivity_index(
            metrics["wikitext_jspace"]["top1_agreement"],
            metrics["wikitext_random"]["top1_agreement"],
        )
        si_ci = bootstrap_selectivity_index(wiki_j, wiki_r, rng)
        row["h1_anti_selectivity"] = {
            "sequences_j_worse": worse,
            "sequences_j_better": better,
            "sequences_tied": len(deltas) - worse - better,
            "sign_test_p": p_sign,
            "mean_paired_delta": sum(deltas) / len(deltas),
            "mean_paired_delta_ci95": bootstrap_mean(deltas, rng),
            "selectivity_index": si,
            "selectivity_index_ci95": si_ci,
            "confirmed": bool(p_sign < H1_ALPHA and worse > better and si_ci[0] > 1.0),
        }
    else:
        row["h1_anti_selectivity"] = None

    clean = metrics.get("control_clean", {}).get("hits")
    jarm = metrics.get("control_jspace", {}).get("hits")
    rarm = metrics.get("control_random", {}).get("hits")
    if clean and jarm and rarm:
        # Descriptive intervals following the v2 reporting convention (not v3 endpoints).
        rng = Random(BOOTSTRAP_SEED)
        row["descriptive_intervals"] = {
            "j_drop_ci95": bootstrap_paired_difference(clean, jarm, rng),
            "specificity_ci95": bootstrap_paired_difference(rarm, jarm, rng),
        }
    if clean and jarm:
        flips = downflips(clean, jarm)
        # The registered H4 null draws from this model's clean-correct items, so the
        # marked class is the core items PRESENT in that population: core items the
        # clean model answers incorrectly can never be down-flips.
        core_present = len(set(CORE_4B) & {i for i, hit in enumerate(clean) if hit})
        row["h4_flip_core"] = {
            "downflips": flips,
            "core_overlap": len(set(flips) & set(CORE_4B)),
            "clean_correct": sum(clean),
            "core_items_clean_correct": core_present,
            "applicable": bool(metrics.get("positive_control", {}).get("j_drop", 0) > 0 and flips),
            "overlap_tail_p": (
                overlap_tail_p(sum(clean), core_present, len(flips), len(set(flips) & set(CORE_4B)))
                if flips and core_present
                else None
            ),
        }
    else:
        row["h4_flip_core"] = None
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    from jspace.scale_config import REPO_ROOT

    rows = []
    for profile in V3_MODELS:
        path = REPO_ROOT / "extension" / "results" / profile / "scale_gate.json"
        if not path.exists():
            rows.append({"profile": profile, "status": "MISSING"})
            continue
        with path.open() as handle:
            rows.append(summarize_model(json.load(handle)))
    payload = {
        "registration": "extension/PREREGISTRATION_V3.md",
        "h1_alpha_per_model": H1_ALPHA,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "rows": rows,
    }
    output_dir = REPO_ROOT / "extension" / "analysis_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
