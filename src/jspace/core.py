"""Backend-neutral model setup and generation constants for the registered study."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import transformers

if TYPE_CHECKING:
    import jlens


K_ABLATE = 10
K_CLEAN_EXEMPT = 10
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}

COT_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
DIRECT_INSTRUCTION = "Give only the final answer within \\boxed{}. Do not show any working."


@dataclass
class Setup:
    hf: transformers.PreTrainedModel
    tok: transformers.PreTrainedTokenizerBase
    model: "jlens.HFLensModel"
    lens: "jlens.JacobianLens"
    device: str

    @property
    def blocks(self):
        return self.model.layers

    @property
    def final_norm(self):
        return self.hf.model.norm

    @property
    def lm_head(self):
        return self.hf.lm_head


def resolve_device(device: str | None = None) -> str:
    """Resolve an explicit backend, otherwise prefer CUDA and then MPS."""
    requested = device or os.environ.get("JSPACE_DEVICE")
    if requested:
        requested = requested.lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("JSPACE_DEVICE requests CUDA, but CUDA is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("JSPACE_DEVICE requests MPS, but MPS is unavailable")
        if requested not in {"cpu", "mps"} and not requested.startswith("cuda"):
            raise ValueError(f"unsupported JSPACE_DEVICE={requested!r}")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def chat_prompt(setup: Setup, problem: str, mode: str) -> str:
    if mode not in {"cot", "direct"}:
        raise ValueError(f"unknown prompting mode {mode!r}")
    instruction = COT_INSTRUCTION if mode == "cot" else DIRECT_INSTRUCTION
    messages = [{"role": "user", "content": f"{problem}\n\n{instruction}"}]
    return setup.tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
