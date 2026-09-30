"""Registered comparison between the stock Jacobian lens and the logit lens."""

from __future__ import annotations

import torch

KS = (1, 5, 10)
WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}
OPS = {
    "multiplication": ["multiplication", "multiply", "multiplied", "times", "product", "*", "×"],
    "addition": ["addition", "add", "added", "plus", "sum", "+"],
    "subtraction": ["subtraction", "subtract", "subtracted", "minus", "difference", "-", "−"],
    "division": ["division", "divide", "divided", "quotient", "/", "÷"],
}


def synonyms(intermediate: str) -> list[str]:
    if intermediate in OPS:
        return OPS[intermediate]
    try:
        number = int(intermediate)
    except ValueError:
        return [intermediate]
    return [intermediate] + ([WORDS[number]] if number in WORDS else [])


def single_token_ids(tokenizer, words: list[str]) -> list[int]:
    ids = set()
    for word in words:
        for variant in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
    return sorted(ids)


def best_rank(logits: torch.Tensor, target_ids: list[int]) -> int:
    order = logits.argsort(descending=True)
    lookup = torch.empty_like(order)
    lookup[order] = torch.arange(len(order), device=order.device)
    return int(min(lookup[target].item() for target in target_ids))


def item_fractions(
    setup, item: dict, use_jacobian: bool, expand_synonyms: bool
) -> dict[str, float]:
    layers = sorted(setup.lens.jacobians)
    lens_logits, _, _ = setup.lens.apply(
        setup.model,
        item["prompt"],
        positions=[-1],
        layers=layers,
        use_jacobian=use_jacobian,
    )
    ranks = []
    for intermediate in item["intermediates"]:
        text = str(intermediate)
        forms = synonyms(text) if expand_synonyms else [text]
        target_ids = single_token_ids(setup.tok, forms)
        if target_ids:
            ranks.append(min(best_rank(lens_logits[layer][0], target_ids) for layer in layers))
    return {str(k): sum(rank < k for rank in ranks) / max(len(ranks), 1) for k in KS}


def passes_gate(report: dict) -> bool:
    def wins(result: dict) -> bool:
        jacobian = result["jlens"]
        baseline = result["logit_lens"]
        return (
            jacobian["1"] >= baseline["1"]
            and jacobian["5"] >= baseline["5"]
            and (jacobian["1"] > baseline["1"] or jacobian["5"] > baseline["5"])
        )

    return any(wins(report[name]) for name in ("order_ops", "multihop"))
