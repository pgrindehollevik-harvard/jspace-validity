"""Checkpointed, preregistered 8B/14B scale-validity experiment.

The scientific choices live in extension/config.json and extension/PREREGISTRATION.md.
This runner refuses excluded profiles, dirty tracked code, and changed configuration or
data digests. It writes only namespaced version-2 state and result artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from jspace import core
from jspace.ablation import JSpaceAblator
from jspace.generate import generate, generate_resumable
from jspace.lens_validation import KS, item_fractions, passes_gate
from jspace.metrics import mean, metric_report
from jspace.profiles import LENS_REVISION, get_profile
from jspace.quality import max_ngram_repetition
from jspace.scale_config import (
    REPO_ROOT,
    config_sha256,
    confirmatory_profiles,
    load_scale_config,
    transferred_band,
)
from jspace.scale_data import load_staged_data
from jspace.scale_runtime import load_scale_setup, verify_companion_checkout


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=1)
    temporary.replace(path)


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def require_clean_tracked_tree() -> None:
    result = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError("scientific jobs require committed, clean tracked files")


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def environment_report() -> dict:
    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    # Hostnames and scheduler identifiers are operational metadata, not scientific
    # provenance. Keep public result artifacts portable by recording them only when a
    # local operator explicitly opts in.
    if os.environ.get("JSPACE_RECORD_EXECUTION_IDS") == "1":
        report["host"] = platform.node()
        report["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
    return report


def update_peak(state: dict) -> None:
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        state["peak_cuda_gb"] = max(state.get("peak_cuda_gb", 0.0), peak)


def lens_override_report() -> dict:
    """Record true lens provenance when a registered local artifact replaces the stock file."""
    from jspace.scale_runtime import lens_override_path

    override = lens_override_path()
    if override is None:
        return {}
    return {
        "lens_override_file": str(override.relative_to(REPO_ROOT))
        if override.is_relative_to(REPO_ROOT)
        else str(override),
        "lens_override_sha256": hashlib.sha256(override.read_bytes()).hexdigest(),
    }


def load_items(config: dict, name: str) -> list[dict]:
    path = REPO_ROOT / config["data"][name]["path"]
    with path.open() as handle:
        items = json.load(handle)["items"]
    expected = config["data"][name]["n"]
    if len(items) != expected:
        raise RuntimeError(f"{name} has {len(items)} items, expected {expected}")
    return items


def run_lens_validation(setup, config: dict, state: dict, save) -> dict:
    stage = state["stages"].setdefault("lens_validation", {"datasets": {}})
    specs = (("order_ops", True), ("multihop", False))
    for dataset_name, expand in specs:
        items = load_items(config, dataset_name)
        dataset = stage["datasets"].setdefault(dataset_name, {})
        for arm, use_jacobian in (("jlens", True), ("logit_lens", False)):
            records = dataset.setdefault(arm, [])
            for index in range(len(records), len(items)):
                started = time.perf_counter()
                fractions = item_fractions(setup, items[index], use_jacobian, expand)
                records.append(
                    {
                        "index": index,
                        "fractions": fractions,
                        "seconds": time.perf_counter() - started,
                    }
                )
                update_peak(state)
                save()
                print(
                    f"lens[{dataset_name}:{arm}] {index + 1}/{len(items)}",
                    flush=True,
                )

    report = {}
    for dataset_name, _ in specs:
        dataset = stage["datasets"][dataset_name]
        report[dataset_name] = {"n": len(dataset["jlens"])}
        for arm in ("jlens", "logit_lens"):
            report[dataset_name][arm] = {
                str(k): mean([r["fractions"][str(k)] for r in dataset[arm]]) for k in KS
            }

    report["gate"] = "PASS" if passes_gate(report) else "FAIL"
    stage["report"] = report
    save()
    return report


def run_control_arm(
    setup,
    items: list[dict],
    profile_key: str,
    band: tuple[int, int],
    k: int,
    arm: str,
    state: dict,
    save,
) -> list[dict]:
    stage = state["stages"].setdefault("control", {})
    records = stage.setdefault(arm, [])
    ablator = None
    if arm != "clean":
        ablator = JSpaceAblator(setup, band, k=k, mode=arm).install()
    try:
        for index in range(len(records), len(items)):
            item = items[index]
            if ablator is not None and arm == "random":
                ablator.reseed(stable_seed(profile_key, "control", str(index)))
            started = time.perf_counter()
            result = generate(
                setup,
                item["prompt"],
                max_new_tokens=8,
                seed=0,
                ablator=ablator,
                greedy=True,
            )
            target = item["target"].strip().lower()
            records.append(
                {
                    "index": index,
                    "hit": target in result.text.lower(),
                    "text": result.text,
                    "tokens": result.n_new,
                    "seconds": time.perf_counter() - started,
                    "norms": {str(layer): value for layer, value in result.norm_log.items()},
                }
            )
            update_peak(state)
            save()
            print(f"control[{arm}] {index + 1}/{len(items)}", flush=True)
    finally:
        if ablator is not None:
            ablator.remove()
    return records


def run_wikitext_arm(
    setup,
    texts: list[str],
    profile_key: str,
    band: tuple[int, int],
    k: int,
    arm: str,
    max_tokens: int,
    state: dict,
    save,
) -> list[dict]:
    stage = state["stages"].setdefault("wikitext", {})
    records = stage.setdefault(arm, [])
    ablator = JSpaceAblator(setup, band, k=k, mode=arm).install()
    try:
        for index in range(len(records), len(texts)):
            if arm == "random":
                ablator.reseed(stable_seed(profile_key, "wikitext", str(index)))
            ids = setup.tok(
                texts[index],
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
            ).input_ids.to(setup.device)
            started = time.perf_counter()
            with torch.no_grad():
                ablator.enabled = False
                clean_logits = setup.hf(ids).logits[0]
                clean_argmax = clean_logits.argmax(-1)
                ablator.exempt_ids = clean_logits.topk(core.K_CLEAN_EXEMPT, dim=-1).indices
                del clean_logits
                ablator.enabled = True
                ablated_argmax = setup.hf(ids).logits[0].argmax(-1)
                ablator.enabled = False
            matched = int((clean_argmax == ablated_argmax).sum().item())
            total = clean_argmax.numel()
            records.append(
                {
                    "index": index,
                    "text_sha256": hashlib.sha256(texts[index].encode()).hexdigest(),
                    "matches": matched,
                    "total": total,
                    "agreement": matched / total,
                    "seconds": time.perf_counter() - started,
                    "norms": {
                        str(layer): value for layer, value in ablator.pop_norm_summary().items()
                    },
                }
            )
            update_peak(state)
            save()
            print(f"wikitext[{arm}] {index + 1}/{len(texts)}", flush=True)
    finally:
        ablator.enabled = False
        ablator.remove()
    return records


def run_coherence(
    setup,
    problems: list[dict],
    config: dict,
    profile_key: str,
    band: tuple[int, int],
    k: int,
    state: dict,
    save,
) -> list[dict]:
    specification = config["data"]["coherence"]
    records = state["stages"].setdefault("coherence", [])
    ablator = JSpaceAblator(setup, band, k=k, mode="jspace").install()
    try:
        for index in range(len(records), len(problems)):
            records.append(
                {
                    "index": index,
                    "id": problems[index]["id"],
                    "generation": {},
                    "seconds": 0.0,
                }
            )
            save()
        for index, item in enumerate(problems):
            record = records[index]
            if "summary" in record:
                continue
            prompt = core.chat_prompt(setup, item["problem"], "cot")
            started = time.perf_counter()
            result = generate_resumable(
                setup,
                prompt,
                max_new_tokens=specification["max_new_tokens"],
                seed=specification["generation_seed"],
                ablator=ablator,
                state=record["generation"],
                save=save,
            )
            record["seconds"] += time.perf_counter() - started
            degenerate = max_ngram_repetition(result.token_ids) > 0.5 or result.hit_cap
            record["summary"] = {
                "degenerate": bool(degenerate),
                "tokens": len(result.token_ids),
                "hit_cap": result.hit_cap,
                "stopped": result.stopped,
                "text": result.text,
                "norms": {str(layer): value for layer, value in result.norm_log.items()},
            }
            update_peak(state)
            save()
            print(f"coherence[jspace] {index + 1}/{len(problems)}", flush=True)
    finally:
        ablator.remove()
    return records


def final_report(
    profile_key: str, config: dict, data: dict, state: dict, stop_reason: str | None
) -> dict:
    profile = get_profile(profile_key)
    metrics, checks = metric_report(state, config["thresholds"])
    required = (
        "lens_validation",
        "clean_control_floor",
        "positive_j_drop",
        "j_below_random",
        "j_drop_twice_random",
        "wikitext_selectivity",
        "coherence",
    )
    passes = all(checks.get(name) is True for name in required)
    coherence_records = state["stages"].get("coherence", [])
    outputs = [
        {
            "id": record["id"],
            "seconds": record["seconds"],
            **record["summary"],
        }
        for record in coherence_records
        if "summary" in record
    ]
    return {
        "schema_version": 2,
        "title": config["title"],
        "profile": profile_key,
        "role": "primary" if profile_key in config["models"]["primary"] else "secondary",
        "model_id": profile.model_id,
        "model_revision": profile.model_revision,
        "lens_file": profile.lens_file,
        "lens_revision": LENS_REVISION,
        **lens_override_report(),
        "companion": state["companion"],
        "code_commit": state["code_commit"],
        "config_sha256": state["config_sha256"],
        "data_sha256": data["payload_sha256"],
        "band": transferred_band(profile.n_layers, config),
        "k": config["intervention"]["k"],
        "started_at": state["started_at"],
        "finished_at": utc_now(),
        "environment": state["environment"],
        "peak_cuda_gb": state.get("peak_cuda_gb"),
        "lens_validation": state["stages"].get("lens_validation", {}).get("report"),
        "metrics": metrics,
        "checks": checks,
        "required_checks": required,
        "passes_gate": passes,
        "status": "PASS" if passes else "STOP",
        "stop_reason": None if passes else stop_reason,
        "coherence_outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=confirmatory_profiles(), required=True)
    args = parser.parse_args()
    require_clean_tracked_tree()

    config = load_scale_config()
    data = load_staged_data()
    companion = verify_companion_checkout()
    profile = get_profile(args.model)
    band = transferred_band(profile.n_layers, config)
    state_path = REPO_ROOT / "extension" / "state" / f"{args.model}.json"
    result_path = REPO_ROOT / "extension" / "results" / args.model / "scale_gate.json"
    commit = git_commit()
    if state_path.exists():
        with state_path.open() as handle:
            state = json.load(handle)
        expected = (
            2,
            args.model,
            commit,
            config_sha256(),
            data["payload_sha256"],
            companion,
        )
        observed = (
            state.get("schema_version"),
            state.get("profile"),
            state.get("code_commit"),
            state.get("config_sha256"),
            state.get("data_sha256"),
            state.get("companion"),
        )
        if observed != expected:
            raise RuntimeError(
                f"checkpoint provenance mismatch: observed={observed}, expected={expected}"
            )
    else:
        state = {
            "schema_version": 2,
            "profile": args.model,
            "code_commit": commit,
            "config_sha256": config_sha256(),
            "data_sha256": data["payload_sha256"],
            "companion": companion,
            "started_at": utc_now(),
            "environment": environment_report(),
            "stages": {},
        }

    def save() -> None:
        atomic_dump(state_path, state)

    save()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print(
        json.dumps(
            {
                "profile": args.model,
                "role": "primary" if args.model in config["models"]["primary"] else "secondary",
                "band": band,
                "config_sha256": state["config_sha256"],
                "data_sha256": state["data_sha256"],
                "code_commit": commit,
            },
            indent=2,
        ),
        flush=True,
    )

    # Version-3 registrations set this flag to measure every non-coherence stage even
    # when an earlier prerequisite fails; the gate verdict itself stays conjunctive.
    # Absent (the frozen v2 configuration), the ordered early stops below are unchanged.
    unconditional = bool(config.get("measure_noncoherence_unconditionally"))

    setup = load_scale_setup(args.model, device="cuda")
    lens_report = run_lens_validation(setup, config, state, save)
    if lens_report["gate"] != "PASS" and not unconditional:
        report = final_report(args.model, config, data, state, "lens_validation_failed")
        atomic_dump(result_path, report)
        print(json.dumps(report, indent=2), flush=True)
        return

    control_items = load_items(config, "order_ops")
    clean = run_control_arm(
        setup,
        control_items,
        args.model,
        band,
        config["intervention"]["k"],
        "clean",
        state,
        save,
    )
    clean_accuracy = mean([float(record["hit"]) for record in clean])
    if clean_accuracy < config["thresholds"]["clean_control_accuracy_min"] and not unconditional:
        report = final_report(args.model, config, data, state, "clean_control_below_floor")
        atomic_dump(result_path, report)
        print(json.dumps(report, indent=2), flush=True)
        return

    for arm in ("jspace", "random"):
        run_control_arm(
            setup,
            control_items,
            args.model,
            band,
            config["intervention"]["k"],
            arm,
            state,
            save,
        )
    for arm in ("jspace", "random"):
        run_wikitext_arm(
            setup,
            data["wikitext"],
            args.model,
            band,
            config["intervention"]["k"],
            arm,
            config["data"]["wikitext"]["max_tokens"],
            state,
            save,
        )

    _, checks = metric_report(state, config["thresholds"])
    # In the frozen v2 flow the first two names are true by construction here, so this
    # expanded conjunction leaves v2 stop reasons byte-identical; under the v3
    # unconditional flag it keeps coherence gated on every earlier condition.
    noncoherence = (
        "lens_validation",
        "clean_control_floor",
        "positive_j_drop",
        "j_below_random",
        "j_drop_twice_random",
        "wikitext_selectivity",
    )
    if not all(checks.get(name) is True for name in noncoherence):
        failed = [name for name in noncoherence if checks.get(name) is not True]
        report = final_report(
            args.model,
            config,
            data,
            state,
            "noncoherence_gate_failed:" + ",".join(failed),
        )
        atomic_dump(result_path, report)
        print(json.dumps(report, indent=2), flush=True)
        return

    run_coherence(
        setup,
        data["coherence"],
        config,
        args.model,
        band,
        config["intervention"]["k"],
        state,
        save,
    )
    report = final_report(args.model, config, data, state, "coherence_gate_failed")
    atomic_dump(result_path, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
