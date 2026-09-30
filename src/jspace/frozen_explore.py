"""Exploratory reanalysis of the frozen v2 artifacts.

Every input is an immutable registered artifact: the historical Qwen3-4B calibration
files and the registered 14B scale-gate result. Every output is EXPLORATORY: it was
computed after the registered outcomes were known, it generates hypotheses rather than
confirmatory evidence, and it changes no registered verdict. Confirmatory use of any
quantity produced here requires a fresh preregistration.

The script is deterministic (seeded bootstrap, exact tests) and writes
``extension/exploratory/frozen_reanalysis.json``. It never modifies a frozen file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path
from random import Random

from jspace.scale_analyze import bootstrap_mean, percentile
from jspace.stats import mcnemar_exact_p

BOOTSTRAP_SEED = 20260825
N_BOOTSTRAP = 10_000
RUNG_LABELS = ("14-24:10", "14-31:10", "14-24:5", "14-19:10")


def sign_test_p(negative: int, positive: int) -> float:
    """Exact two-sided sign test on strict paired differences (ties excluded)."""
    n = negative + positive
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(negative, positive) + 1)) / 2**n
    return min(1.0, 2 * tail)


def selectivity_index(agree_intervention: float, agree_control: float) -> float:
    """Ratio of mean disagreement rates: >1 means the targeted arm damages more."""
    control_disagreement = 1.0 - agree_control
    if control_disagreement <= 0:
        raise ValueError("control arm has no disagreement; index undefined")
    return (1.0 - agree_intervention) / control_disagreement


def bootstrap_selectivity_index(
    per_seq_intervention: list[float], per_seq_control: list[float], rng: Random
) -> list[float]:
    """Percentile CI for the paired-sequence selectivity index."""
    if len(per_seq_intervention) != len(per_seq_control):
        raise ValueError("paired per-sequence vectors must have equal length")
    n = len(per_seq_intervention)
    draws = []
    for _ in range(N_BOOTSTRAP):
        top = bottom = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            top += 1.0 - per_seq_intervention[i]
            bottom += 1.0 - per_seq_control[i]
        draws.append(top / bottom if bottom > 0 else float("inf"))
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def min_detectable_drop(n_items: int, alpha: float = 0.05) -> dict:
    """Smallest all-one-way discordant count with exact two-sided McNemar p below alpha."""
    discordant = 1
    while min(1.0, 2 * (1 / 2**discordant)) >= alpha:
        discordant += 1
    examples = []
    for down, up in ((discordant - 1, 0), (discordant, 0), (discordant + 2, 1)):
        n = down + up
        tail = sum(comb(n, i) for i in range(min(down, up) + 1)) / 2**n
        examples.append({"down": down, "up": up, "p": min(1.0, 2 * tail)})
    return {
        "n_items": n_items,
        "alpha": alpha,
        "min_discordant_all_one_way": discordant,
        "min_detectable_drop": discordant / n_items,
        "examples": examples,
    }


def downflips(clean: list[bool], arm: list[bool]) -> list[int]:
    return [i for i, (c, a) in enumerate(zip(clean, arm)) if c and not a]


def hypergeometric_overlap(universe: int, size_a: int, size_b: int) -> dict:
    """Exact null distribution of |A ∩ B| for fixed-size subsets of one universe."""
    lo, hi = max(0, size_a + size_b - universe), min(size_a, size_b)
    pmf = {
        k: comb(size_a, k) * comb(universe - size_a, size_b - k) / comb(universe, size_b)
        for k in range(lo, hi + 1)
    }
    expected_overlap = sum(k * p for k, p in pmf.items())
    expected_jaccard = sum(p * (k / (size_a + size_b - k)) for k, p in pmf.items() if k > 0)
    return {"pmf": pmf, "expected_overlap": expected_overlap, "expected_jaccard": expected_jaccard}


def overlap_tail_p(universe: int, size_a: int, size_b: int, observed: int) -> float:
    pmf = hypergeometric_overlap(universe, size_a, size_b)["pmf"]
    return sum(p for k, p in pmf.items() if k >= observed)


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(repo_root: Path) -> dict:
    baseline = repo_root / "baseline" / "qwen3-4b"
    result_14b = repo_root / "extension" / "results" / "qwen3-14b" / "scale_gate.json"
    calibration_path = baseline / "calibration.json"
    state_path = baseline / "calibration_state.json"
    with result_14b.open() as handle:
        gate_14b = json.load(handle)
    with calibration_path.open() as handle:
        calibration = json.load(handle)
    with state_path.open() as handle:
        state = json.load(handle)

    rng = Random(BOOTSTRAP_SEED)
    metrics = gate_14b["metrics"]
    wiki_j = metrics["wikitext_jspace"]["per_sequence_agreement"]
    wiki_r = metrics["wikitext_random"]["per_sequence_agreement"]
    deltas = [j - r for j, r in zip(wiki_j, wiki_r)]
    worse = sum(1 for d in deltas if d < 0)
    better = sum(1 for d in deltas if d > 0)
    ties = len(deltas) - worse - better
    mean_delta = sum(deltas) / len(deltas)
    delta_ci = bootstrap_mean(deltas, rng)
    si_14b = selectivity_index(
        metrics["wikitext_jspace"]["top1_agreement"], metrics["wikitext_random"]["top1_agreement"]
    )
    si_ci = bootstrap_selectivity_index(wiki_j, wiki_r, rng)

    si_4b_rungs = [
        {
            "band": rung["band"],
            "k": rung["k"],
            "selectivity_index": selectivity_index(
                rung["wikitext_top1_jspace"], rung["wikitext_top1_random"]
            ),
            "note": "point estimate; per-sequence agreements were not retained at 4B",
        }
        for rung in calibration["rungs"]
    ]

    disp_j = metrics["wikitext_jspace"]["mean_layer_relative_displacement"]
    disp_r = metrics["wikitext_random"]["mean_layer_relative_displacement"]
    layers = sorted(disp_j, key=int)
    dose_j = sum(disp_j[layer] for layer in layers)
    dose_r = sum(disp_r[layer] for layer in layers)
    damage_j = 1.0 - metrics["wikitext_jspace"]["top1_agreement"]
    damage_r = 1.0 - metrics["wikitext_random"]["top1_agreement"]
    displacement = {
        "layers": layers,
        "per_layer_jspace": disp_j,
        "per_layer_random": disp_r,
        "entry_layer": layers[0],
        "entry_ratio_j_over_r": disp_j[layers[0]] / disp_r[layers[0]],
        "attenuation_last_over_first_jspace": disp_j[layers[-1]] / disp_j[layers[0]],
        "attenuation_last_over_first_random": disp_r[layers[-1]] / disp_r[layers[0]],
        "cumulative_dose_jspace": dose_j,
        "cumulative_dose_random": dose_r,
        "damage_jspace": damage_j,
        "damage_random": damage_r,
        "damage_per_unit_dose_jspace": damage_j / dose_j,
        "damage_per_unit_dose_random": damage_r / dose_r,
        "damage_per_dose_ratio": (damage_j / dose_j) / (damage_r / dose_r),
        "dose_definition": "sum over band layers of the recorded mean per-layer "
        "relative displacement; a coarse severity proxy, not a causal quantity",
    }

    clean_4b = state["clean_orderops"]
    universe_4b = sum(clean_4b)
    rung_flips = {
        label: downflips(clean_4b, state[f"rung:{label}:jspace"]["hits"]) for label in RUNG_LABELS
    }
    core = set.intersection(*(set(flips) for flips in rung_flips.values()))
    pair_stats = {}
    labels = list(RUNG_LABELS)
    for i, first in enumerate(labels):
        for second in labels[i + 1 :]:
            a, b = set(rung_flips[first]), set(rung_flips[second])
            null = hypergeometric_overlap(universe_4b, len(a), len(b))
            pair_stats[f"{first} vs {second}"] = {
                "jaccard": jaccard(a, b),
                "expected_jaccard_null": null["expected_jaccard"],
                "overlap_tail_p": overlap_tail_p(universe_4b, len(a), len(b), len(a & b)),
            }
    for label in RUNG_LABELS:
        expected_p = next(
            rung["mcnemar_p_drop"]
            for rung in calibration["rungs"]
            if f"{rung['band'][0]}-{rung['band'][1]}:{rung['k']}" == label
        )
        recomputed = mcnemar_exact_p(clean_4b, state[f"rung:{label}:jspace"]["hits"])
        if abs(recomputed - expected_p) > 1e-12:
            raise RuntimeError(f"consistency check failed for rung {label}")

    random_flips = {
        label: downflips(clean_4b, state[f"rung:{label}:random"]["hits"]) for label in RUNG_LABELS
    }
    random_events = sum(len(flips) for flips in random_flips.values())
    random_events_in_core = sum(len(core & set(flips)) for flips in random_flips.values())

    clean_14b = metrics["control_clean"]["hits"]
    flips_14b = downflips(clean_14b, metrics["control_jspace"]["hits"])
    universe_14b = sum(clean_14b)
    null_cross = hypergeometric_overlap(universe_14b, len(core), len(flips_14b))
    flip_anatomy = {
        "clean_correct_4b": universe_4b,
        "rung_downflips": rung_flips,
        "core_items": sorted(core),
        "pairwise": pair_stats,
        "rung_downflips_random": random_flips,
        "random_downflip_events": random_events,
        "random_downflip_events_in_core": random_events_in_core,
        "core_fraction_null": len(core) / universe_4b,
        "downflips_14b": flips_14b,
        "clean_correct_14b": universe_14b,
        "core_overlap_14b": len(core & set(flips_14b)),
        "expected_overlap_14b_null": null_cross["expected_overlap"],
        "alignment_note": "cross-model item indices assume both studies stage the pinned "
        "order-of-operations file in file order; within-4B comparisons do not need this",
    }

    return {
        "status": "exploratory",
        "provenance_note": "computed after registered outcomes were known; "
        "changes no registered verdict",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "inputs": {
            "extension/results/qwen3-14b/scale_gate.json": file_sha256(result_14b),
            "baseline/qwen3-4b/calibration.json": file_sha256(calibration_path),
            "baseline/qwen3-4b/calibration_state.json": file_sha256(state_path),
        },
        "paired_selectivity_14b": {
            "n_sequences": len(deltas),
            "sequences_j_worse": worse,
            "sequences_j_better": better,
            "sequences_tied": ties,
            "sign_test_p": sign_test_p(worse, better),
            "mean_paired_delta": mean_delta,
            "mean_paired_delta_ci95": delta_ci,
            "selectivity_index": si_14b,
            "selectivity_index_ci95": si_ci,
        },
        "selectivity_index_4b_rungs": si_4b_rungs,
        "displacement_14b_wikitext": displacement,
        "flip_anatomy": flip_anatomy,
        "mcnemar_design_limits": min_detectable_drop(len(clean_14b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    from jspace.scale_config import REPO_ROOT

    payload = analyze(REPO_ROOT)
    output_dir = REPO_ROOT / "extension" / "exploratory"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "frozen_reanalysis.json").open("w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
