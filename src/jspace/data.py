"""Deterministic loaders for the two public datasets used by the scale gate."""

import random

from datasets import load_dataset


def gsm8k_answer(solution: str) -> str:
    return solution.split("####")[-1].strip().replace(",", "")


def load_gsm8k(n: int, seed: int, split: str, revision: str) -> list[dict]:
    dataset = load_dataset("openai/gsm8k", "main", split=split, revision=revision)
    indices = random.Random(seed).sample(range(len(dataset)), n)
    return [
        {
            "id": f"gsm8k-{split}-{index}",
            "dataset": "gsm8k",
            "level": 0,
            "problem": dataset[index]["question"],
            "answer": gsm8k_answer(dataset[index]["answer"]),
        }
        for index in indices
    ]


def load_wikitext_heldout(
    n: int,
    min_chars: int,
    skip: int,
    max_chars: int,
    revision: str,
) -> list[str]:
    """Select the registered held-out Wikitext rows without materializing the corpus."""
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        streaming=True,
        revision=revision,
    )
    selected: list[str] = []
    eligible = 0
    for row in dataset:
        text = row["text"].strip()
        if len(text) < min_chars:
            continue
        eligible += 1
        if eligible <= skip:
            continue
        selected.append(text[:max_chars])
        if len(selected) == n:
            return selected
    raise RuntimeError("not enough eligible Wikitext rows")
