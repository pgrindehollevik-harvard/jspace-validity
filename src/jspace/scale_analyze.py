"""Locked post-processing for the v2 scale gate.

Requires both preregistered 8B and 14B result files. It combines them with the already
observed 4B light-band rung, computes paired/sequence bootstrap intervals, and writes
the single figure and table reserved for the five-page workshop paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random

from jspace.scale_config import REPO_ROOT, load_scale_config
from jspace.stats import mcnemar_exact_p

BOOTSTRAP_SEED = 20260825
N_BOOTSTRAP = 10_000
OUTPUT_DIR = REPO_ROOT / "extension" / "analysis"


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_mean(values: list[float], rng: random.Random) -> list[float]:
    n = len(values)
    draws = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOTSTRAP)]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def bootstrap_paired_difference(
    first: list[bool], second: list[bool], rng: random.Random
) -> list[float]:
    if len(first) != len(second):
        raise ValueError("paired vectors must have equal length")
    per_item = [float(a) - float(b) for a, b in zip(first, second)]
    return bootstrap_mean(per_item, rng)


def historical_row() -> tuple[dict, dict[str, list[bool]]]:
    baseline = REPO_ROOT / "baseline" / "qwen3-4b"
    with (baseline / "calibration.json").open() as handle:
        calibration = json.load(handle)
    with (baseline / "calibration_state.json").open() as handle:
        state = json.load(handle)
    with (baseline / "lens_validation.json").open() as handle:
        lens_validation = json.load(handle)
    rung = next(
        rung for rung in calibration["rungs"] if rung["band"] == [14, 19] and rung["k"] == 10
    )
    hits = {
        "clean": state["clean_orderops"],
        "jspace": state["rung:14-19:10:jspace"]["hits"],
        "random": state["rung:14-19:10:random"]["hits"],
    }
    p_random_j = mcnemar_exact_p(hits["random"], hits["jspace"])
    checks = {
        "lens_validation": lens_validation["gate"] == "PASS",
        "clean_control_floor": calibration["clean_orderops"] >= 0.60,
        "positive_j_drop": rung["j_drop"] > 0 and rung["mcnemar_p_drop"] < 0.05,
        "j_below_random": (rung["control_random"] > rung["control_jspace"] and p_random_j < 0.05),
        "j_drop_twice_random": rung["j_drop"] >= 2 * max(rung["r_drop"], 0),
        "wikitext_selectivity": rung["wikitext_top1_jspace"] > 0.80,
        "coherence": rung["degenerate_rate"] < 0.10,
    }
    row = {
        "profile": "qwen3-4b",
        "parameters_b": 4,
        "role": "historical",
        "control_clean": calibration["clean_orderops"],
        "control_jspace": rung["control_jspace"],
        "control_random": rung["control_random"],
        "j_drop": rung["j_drop"],
        "random_drop": rung["r_drop"],
        "specificity_random_minus_j": (rung["control_random"] - rung["control_jspace"]),
        "mcnemar_p_clean_vs_j": rung["mcnemar_p_drop"],
        "mcnemar_p_random_vs_j": p_random_j,
        "wikitext_jspace": rung["wikitext_top1_jspace"],
        "wikitext_random": rung["wikitext_top1_random"],
        "degenerate_rate": rung["degenerate_rate"],
        "passes_v2_equivalent_gate": all(checks.values()),
        "checks": checks,
    }
    return row, hits


def extension_row(profile: str) -> tuple[dict, dict[str, list[bool]], list[float] | None]:
    path = REPO_ROOT / "extension" / "results" / profile / "scale_gate.json"
    if not path.exists():
        raise FileNotFoundError(f"missing preregistered result {path}")
    with path.open() as handle:
        result = json.load(handle)
    metrics = result["metrics"]
    control = {
        arm: metrics[f"control_{arm}"]["hits"]
        for arm in ("clean", "jspace", "random")
        if f"control_{arm}" in metrics
    }
    wiki_sequence = metrics.get("wikitext_jspace", {}).get("per_sequence_agreement")
    positive = metrics.get("positive_control", {})
    row = {
        "profile": profile,
        "parameters_b": 8 if profile == "qwen3-8b" else 14,
        "role": result["role"],
        "control_clean": metrics.get("control_clean", {}).get("accuracy"),
        "control_jspace": metrics.get("control_jspace", {}).get("accuracy"),
        "control_random": metrics.get("control_random", {}).get("accuracy"),
        "j_drop": positive.get("j_drop"),
        "random_drop": positive.get("random_drop"),
        "specificity_random_minus_j": positive.get("j_specificity_random_minus_j"),
        "mcnemar_p_clean_vs_j": positive.get("mcnemar_p_clean_vs_j"),
        "mcnemar_p_random_vs_j": positive.get("mcnemar_p_random_vs_j"),
        "wikitext_jspace": metrics.get("wikitext_jspace", {}).get("top1_agreement"),
        "wikitext_random": metrics.get("wikitext_random", {}).get("top1_agreement"),
        "degenerate_rate": (metrics.get("coherence", {}).get("degenerate_rate")),
        "passes_v2_equivalent_gate": result["passes_gate"],
        "checks": result["checks"],
        "status": result["status"],
        "stop_reason": result["stop_reason"],
        "gpu": result["environment"].get("gpu"),
        "peak_cuda_gb": result.get("peak_cuda_gb"),
    }
    return row, control, wiki_sequence


def add_intervals(
    row: dict, hits: dict[str, list[bool]], wiki_sequence: list[float] | None, rng: random.Random
) -> None:
    row["j_drop_ci95"] = (
        bootstrap_paired_difference(hits["clean"], hits["jspace"], rng)
        if "clean" in hits and "jspace" in hits
        else None
    )
    row["specificity_ci95"] = (
        bootstrap_paired_difference(hits["random"], hits["jspace"], rng)
        if "random" in hits and "jspace" in hits
        else None
    )
    row["wikitext_jspace_ci95"] = bootstrap_mean(wiki_sequence, rng) if wiki_sequence else None


def write_table(rows: list[dict]) -> None:
    fields = [
        "profile",
        "role",
        "control_clean",
        "control_jspace",
        "control_random",
        "j_drop",
        "j_drop_ci95",
        "specificity_random_minus_j",
        "specificity_ci95",
        "mcnemar_p_clean_vs_j",
        "mcnemar_p_random_vs_j",
        "wikitext_jspace",
        "wikitext_jspace_ci95",
        "wikitext_random",
        "degenerate_rate",
        "passes_v2_equivalent_gate",
        "status",
        "stop_reason",
        "gpu",
        "peak_cuda_gb",
    ]
    with (OUTPUT_DIR / "scale_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_figure(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    labels = [row["profile"].replace("qwen3-", "") for row in rows]
    x = list(range(len(rows)))
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))

    values = [
        row["specificity_random_minus_j"]
        if row["specificity_random_minus_j"] is not None
        else float("nan")
        for row in rows
    ]
    lower = [
        value - row["specificity_ci95"][0] if row["specificity_ci95"] is not None else 0.0
        for value, row in zip(values, rows)
    ]
    upper = [
        row["specificity_ci95"][1] - value if row["specificity_ci95"] is not None else 0.0
        for value, row in zip(values, rows)
    ]
    axes[0].bar(x, values, color=["#9aa0a6", "#6baed6", "#2171b5"])
    axes[0].errorbar(x, values, yerr=[lower, upper], fmt="none", color="black", capsize=3)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Accuracy(random) − accuracy(J)")
    axes[0].set_title("J-specific causal effect")

    wiki_j = [
        row["wikitext_jspace"] if row["wikitext_jspace"] is not None else float("nan")
        for row in rows
    ]
    wiki_random = [
        row["wikitext_random"] if row["wikitext_random"] is not None else float("nan")
        for row in rows
    ]
    axes[1].plot(x, wiki_j, "o-", label="J-space")
    axes[1].plot(x, wiki_random, "s--", label="random")
    axes[1].axhline(0.80, color="#b2182b", linestyle=":", label="registered 80% gate")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.45, 1.01)
    axes[1].set_ylabel("Top-1 agreement with clean")
    axes[1].set_title("Ordinary-text selectivity")
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xlabel("Qwen3 size")
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "scale_gate.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "scale_gate.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    config = load_scale_config()
    expected = set(config["models"]["primary"] + config["models"]["secondary"])
    if expected != {"qwen3-8b", "qwen3-14b"}:
        raise RuntimeError("analysis is locked to the preregistered 8B/14B design")
    rng = random.Random(BOOTSTRAP_SEED)
    historical, historical_hits = historical_row()
    add_intervals(historical, historical_hits, None, rng)
    rows = [historical]
    for profile in ("qwen3-8b", "qwen3-14b"):
        row, hits, wiki = extension_row(profile)
        add_intervals(row, hits, wiki, rng)
        rows.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "rows": rows,
    }
    with (OUTPUT_DIR / "summary.json").open("w") as handle:
        json.dump(payload, handle, indent=1)
    write_table(rows)
    write_figure(rows)
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
