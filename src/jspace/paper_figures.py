"""Deterministic manuscript figures from the frozen and registered artifacts.

Reads only committed result/analysis files and writes paper/figures/*.{pdf,png}.
The locked v2 figure (extension/analysis/scale_gate.*) is untouched. Colors are the
repository's existing two-series convention (validated colorblind-safe); series
identity is never color-alone (marker shapes and direct labels differ).
"""

from __future__ import annotations

import json

from jspace.scale_config import REPO_ROOT

BLUE = "#1f77b4"  # J-space arm
ORANGE = "#ff7f0e"  # norm-matched random arm
GRAY = "#6b7280"
OUT = REPO_ROOT / "paper" / "figures"


def load(path: str) -> dict:
    with (REPO_ROOT / path).open() as handle:
        return json.load(handle)


def style(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)
    axis.xaxis.label.set_size(9)
    axis.yaxis.label.set_size(9)
    axis.title.set_size(9.5)


def figure_si_scale(plt) -> None:
    explore = load("extension/exploratory/frozen_reanalysis.json")
    v3 = {row["profile"]: row for row in load("extension/analysis_v3/summary.json")["rows"]}
    si_4b = next(
        row
        for row in explore["selectivity_index_4b_rungs"]
        if row["band"] == [14, 19] and row["k"] == 10
    )
    points = [
        ("1.7B", 1.7, v3["qwen3-1.7b"]["h1_anti_selectivity"], True),
        (
            "4B",
            4.0,
            {"selectivity_index": si_4b["selectivity_index"], "selectivity_index_ci95": None},
            False,
        ),
        ("14B", 14.0, explore["paired_selectivity_14b"], False),
        ("32B", 32.0, v3["qwen3-32b"]["h1_anti_selectivity"], True),
    ]

    figure, axis = plt.subplots(figsize=(5.5, 2.7))
    xs, ys = [], []
    for label, x, row, registered in points:
        si = row["selectivity_index"]
        ci = row.get("selectivity_index_ci95")
        xs.append(x)
        ys.append(si)
        axis.plot(
            x,
            si,
            marker="o",
            markersize=6.0,
            zorder=4,
            markerfacecolor=BLUE if registered else "white",
            markeredgecolor=BLUE,
            markeredgewidth=1.6,
        )
        if ci:
            axis.errorbar(
                x,
                si,
                yerr=[[si - ci[0]], [ci[1] - si]],
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=4.5,
                capthick=1.0,
                zorder=5,
            )
        if label == "1.7B":
            offset, ha = (13, 1), "left"
        elif label == "4B":
            offset, ha = (0, -15), "center"
        else:
            offset, ha = (0, 9), "center"
        axis.annotate(
            f"{si:.2f}",
            (x, si),
            textcoords="offset points",
            xytext=offset,
            ha=ha,
            fontsize=8,
            color="#374151",
        )
    axis.plot(xs, ys, color=BLUE, linewidth=1.4, alpha=0.55, zorder=2)
    axis.axhline(1.0, color=GRAY, linewidth=1.0, linestyle=":")
    axis.annotate(
        "parity with norm-matched random damage",
        (1.55, 1.0),
        textcoords="offset points",
        xytext=(0, 4),
        fontsize=7.5,
        color=GRAY,
    )
    axis.annotate(
        "8B: no downstream data\n(v2 lens stop)",
        (8.0, 1.42),
        ha="center",
        fontsize=7,
        color=GRAY,
        style="italic",
    )
    axis.set_xscale("log")
    axis.set_xticks([1.7, 4, 8, 14, 32])
    axis.set_xticklabels(["1.7B", "4B", "8B", "14B", "32B"])
    axis.minorticks_off()
    axis.set_xlabel("Qwen3 parameters")
    axis.set_ylabel("Selectivity index (J / random damage)")
    axis.set_ylim(0.88, 3.85)
    style(axis)
    figure.tight_layout()
    figure.savefig(OUT / "si_scale.pdf", bbox_inches="tight")
    figure.savefig(OUT / "si_scale.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def figure_paired_scatter(plt) -> None:
    panels = [
        ("qwen3-1.7b", "1.7B"),
        ("qwen3-14b", "14B"),
        ("qwen3-32b", "32B"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(5.5, 2.05), sharex=True, sharey=True)
    for axis, (profile, label) in zip(axes, panels):
        result = load(f"extension/results/{profile}/scale_gate.json")["metrics"]
        j = result["wikitext_jspace"]["per_sequence_agreement"]
        r = result["wikitext_random"]["per_sequence_agreement"]
        worse = sum(1 for a, b in zip(j, r) if a < b)
        lo, hi = 0.52, 1.0
        axis.plot([lo, hi], [lo, hi], color=GRAY, linewidth=1.0, linestyle="--", zorder=1)
        axis.scatter(r, j, s=13, color=BLUE, alpha=0.75, linewidths=0, zorder=3)
        axis.set_title(f"{label}: {worse}/50 below")
        axis.set_xlim(lo, hi)
        axis.set_ylim(lo, hi)
        axis.set_aspect("equal")
        axis.set_xlabel("random-arm agreement")
        style(axis)
    axes[0].set_ylabel("J-arm agreement")
    figure.tight_layout()
    figure.savefig(OUT / "paired_scatter.pdf", bbox_inches="tight")
    figure.savefig(OUT / "paired_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def figure_displacement(plt) -> None:
    panels = [("qwen3-14b", "14B"), ("qwen3-32b", "32B")]
    figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.25))
    for axis, (profile, label) in zip(axes, panels):
        result = load(f"extension/results/{profile}/scale_gate.json")["metrics"]
        disp_j = result["wikitext_jspace"]["mean_layer_relative_displacement"]
        disp_r = result["wikitext_random"]["mean_layer_relative_displacement"]
        layers = sorted(disp_j, key=int)
        x = [int(layer) for layer in layers]
        axis.plot(
            x,
            [disp_j[layer] for layer in layers],
            color=BLUE,
            linewidth=1.6,
            marker="o",
            markersize=4.5,
            label="J-space",
        )
        axis.plot(
            x,
            [disp_r[layer] for layer in layers],
            color=ORANGE,
            linewidth=1.6,
            marker="s",
            markersize=4.5,
            label="random",
        )
        axis.annotate(
            "matched at entry",
            (x[0], disp_j[layers[0]]),
            textcoords="offset points",
            xytext=(6, 7),
            fontsize=7,
            color="#374151",
        )
        axis.set_title(label)
        axis.set_xlabel("layer")
        axis.set_ylim(0.0, 0.135)
        style(axis)
    axes[0].set_ylabel("mean relative displacement")
    axes[1].legend(frameon=False, fontsize=8, loc="center right")
    figure.tight_layout()
    figure.savefig(OUT / "displacement.pdf", bbox_inches="tight")
    figure.savefig(OUT / "displacement.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    figure_si_scale(plt)
    figure_paired_scatter(plt)
    figure_displacement(plt)
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
