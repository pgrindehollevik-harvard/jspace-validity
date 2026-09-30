"""Pure aggregation and registered gate checks for checkpointed experiment state."""

from __future__ import annotations

from jspace.stats import mcnemar_exact_p


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def mean_norms(records: list[dict]) -> dict[str, float]:
    by_layer: dict[str, list[float]] = {}
    for record in records:
        for layer, value in record.get("norms", {}).items():
            by_layer.setdefault(str(layer), []).append(value)
    return {layer: mean(values) for layer, values in sorted(by_layer.items())}


def metric_report(state: dict, thresholds: dict) -> tuple[dict, dict]:
    stages = state["stages"]
    metrics: dict = {}
    checks = {
        "lens_validation": stages.get("lens_validation", {}).get("report", {}).get("gate") == "PASS"
    }

    control = stages.get("control", {})
    for arm, records in control.items():
        metrics[f"control_{arm}"] = {
            "n": len(records),
            "accuracy": mean([float(record["hit"]) for record in records]),
            "hits": [record["hit"] for record in records],
            "mean_layer_relative_displacement": mean_norms(records),
            "seconds": sum(record["seconds"] for record in records),
        }
    if "clean" in control:
        checks["clean_control_floor"] = (
            metrics["control_clean"]["accuracy"] >= thresholds["clean_control_accuracy_min"]
        )
    if all(arm in control for arm in ("clean", "jspace", "random")):
        clean_hits = [record["hit"] for record in control["clean"]]
        j_hits = [record["hit"] for record in control["jspace"]]
        random_hits = [record["hit"] for record in control["random"]]
        clean_accuracy = metrics["control_clean"]["accuracy"]
        j_accuracy = metrics["control_jspace"]["accuracy"]
        random_accuracy = metrics["control_random"]["accuracy"]
        j_drop = clean_accuracy - j_accuracy
        random_drop = clean_accuracy - random_accuracy
        p_clean_j = mcnemar_exact_p(clean_hits, j_hits)
        p_random_j = mcnemar_exact_p(random_hits, j_hits)
        metrics["positive_control"] = {
            "j_drop": j_drop,
            "random_drop": random_drop,
            "j_specificity_random_minus_j": random_accuracy - j_accuracy,
            "mcnemar_p_clean_vs_j": p_clean_j,
            "mcnemar_p_random_vs_j": p_random_j,
        }
        alpha = thresholds["mcnemar_alpha"]
        checks["positive_j_drop"] = j_drop > 0 and p_clean_j < alpha
        checks["j_below_random"] = random_accuracy > j_accuracy and p_random_j < alpha
        checks["j_drop_twice_random"] = j_drop >= thresholds["j_drop_vs_random_multiplier"] * max(
            random_drop, 0.0
        )

    wikitext = stages.get("wikitext", {})
    for arm, records in wikitext.items():
        matches = sum(record["matches"] for record in records)
        total = sum(record["total"] for record in records)
        metrics[f"wikitext_{arm}"] = {
            "n_sequences": len(records),
            "matches": matches,
            "positions": total,
            "top1_agreement": matches / total if total else float("nan"),
            "per_sequence_agreement": [record["agreement"] for record in records],
            "mean_layer_relative_displacement": mean_norms(records),
            "seconds": sum(record["seconds"] for record in records),
        }
    if "jspace" in wikitext:
        checks["wikitext_selectivity"] = (
            metrics["wikitext_jspace"]["top1_agreement"]
            > thresholds["wikitext_top1_jspace_min_exclusive"]
        )

    coherence = stages.get("coherence", [])
    summaries = [record["summary"] for record in coherence if "summary" in record]
    if summaries:
        rate = mean([float(summary["degenerate"]) for summary in summaries])
        metrics["coherence"] = {
            "n": len(summaries),
            "degenerate_rate": rate,
            "flags": [summary["degenerate"] for summary in summaries],
            "tokens": sum(summary["tokens"] for summary in summaries),
            "seconds": sum(record["seconds"] for record in coherence if "summary" in record),
            "mean_layer_relative_displacement": mean_norms(summaries),
        }
        checks["coherence"] = rate < thresholds["degenerate_rate_max_exclusive"]
    return metrics, checks
