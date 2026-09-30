"""J-space ablation as forward hooks on decoder blocks.

Protocol (source paper section 3.5.2 and extension/PREREGISTRATION.md):
at each (token position, band layer), rank J-lens readout logits computed as
lm_head(final_rmsnorm(J_l @ h)), take the top K_ABLATE token ids, drop any id in the clean
model's top-10 next-token predictions at that position, form directions
v_t = J_l^T (gamma * W_U[t]), and jointly project the residual stream out of their span.

mode="random" is the per-slot norm-matched control: it computes the displacement the J-space
ablation WOULD apply at each slot, then removes an equal-norm component along random
directions at that same slot instead (paper A.23 protocol).

The caller must set `exempt_ids` (from a clean forward pass over the same token stream)
BEFORE each hooked forward. Hooks fire only when `enabled` is True, so the same installed
ablator serves both the clean and ablated passes of the dual-stream generation loop.

The hot path stays on-device: projection uses a batched 10x10 regularized solve, and norm
bookkeeping synchronizes only once per generation through pop_norm_stats().
"""

import torch

from jspace.core import K_ABLATE, Setup

_RAND_BANK_SIZE = 4096


class JSpaceAblator:
    def __init__(
        self,
        setup: Setup,
        band: tuple[int, int],
        k: int = K_ABLATE,
        mode: str = "jspace",
        rng_seed: int = 0,
    ):
        if mode not in {"jspace", "random"}:
            raise ValueError(f"unknown ablation mode {mode!r}")
        self.setup = setup
        self.band = list(range(band[0], band[1] + 1))
        self.k = k
        self.mode = mode
        self.enabled = False
        self.exempt_ids: torch.Tensor | None = None  # [seq, K_CLEAN_EXEMPT], on device
        self._handles = []
        self._rng = torch.Generator(device="cpu").manual_seed(rng_seed)
        dtype = setup.hf.dtype
        # Move only the registered band Jacobians onto the accelerator.
        self._J = {
            layer: setup.lens.jacobians[layer].to(setup.device, dtype) for layer in self.band
        }
        self._gamma = setup.final_norm.weight.detach()
        if mode == "random":
            self._rand_bank = torch.randn(
                _RAND_BANK_SIZE, setup.model.d_model, generator=self._rng
            ).to(setup.device, torch.float32)
        # On-device norm accumulators, synced once per generation.
        self._norm_sum = {layer: torch.zeros((), device=setup.device) for layer in self.band}
        self._norm_cnt = {layer: 0 for layer in self.band}

    # -- hook management -------------------------------------------------------------

    def install(self):
        for layer in self.band:
            self._handles.append(
                self.setup.blocks[layer].register_forward_hook(self._make_hook(layer))
            )
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()

    def reseed(self, seed: int):
        """Pin the per-generation random-direction index stream so every random-arm
        generation is reproducible from (problem_id, seed_idx) alone, independent of
        execution order and resume points. The direction bank itself is fixed at
        construction and unaffected."""
        self._rng.manual_seed(seed)

    def pop_norm_stats(self) -> tuple[dict[int, float], dict[int, int]]:
        """Return per-layer means and slot counts since the last call, then reset."""
        counts = dict(self._norm_cnt)
        out = {layer: (self._norm_sum[layer] / max(counts[layer], 1)).item() for layer in self.band}
        for layer in self.band:
            self._norm_sum[layer] = torch.zeros((), device=self.setup.device)
            self._norm_cnt[layer] = 0
        return out, counts

    def pop_norm_summary(self) -> dict[int, float]:
        """Mean ||delta_h||/||h|| per layer since the last call. Syncs once."""
        return self.pop_norm_stats()[0]

    # -- the intervention ------------------------------------------------------------

    def _make_hook(self, layer: int):
        def hook(module, args, output):
            if not self.enabled:
                return None
            h = output[0] if isinstance(output, tuple) else output  # [B, S, d]
            h_new = self._ablate(h, layer)
            if isinstance(output, tuple):
                return (h_new,) + tuple(output[1:])
            return h_new

        return hook

    def _ablate(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        if self.exempt_ids is None:
            raise RuntimeError("set exempt_ids before a hooked forward")
        B, S, d = h.shape
        if B != 1:
            raise ValueError("batch size 1 required by the paired clean stream")
        J = self._J[layer]
        # Readout logits at every position: lm_head(norm(J @ h)). Never materialize
        # W_U @ J (vocab x d_model per layer); the transported vector is only [S, d].
        transported = h[0] @ J.T  # [S, d]
        logits = self.setup.lm_head(self.setup.final_norm(transported))  # [S, vocab]
        top_ids = logits.topk(self.k, dim=-1).indices  # [S, k]
        exempt = self.exempt_ids.to(h.device)  # [S, 10]
        # Exemption mask without isin (MPS-native broadcast compare).
        exempt_mask = (top_ids.unsqueeze(2) == exempt.unsqueeze(1)).any(2)  # [S, k]

        # Directions for ALL top-k, exempt ones zeroed out of the projection basis:
        # a zero row contributes nothing to span/solve (with regularization), so the
        # whole batch of positions runs as one padded einsum + batched 10x10 solve.
        # bf16 matmul against the resident J (a per-call J.float() would materialize a
        # fresh 26 MB fp32 matrix per band layer per forward); upcast only the small
        # [S, k, d] result for the fp32 solve.
        w = self.setup.lm_head.weight[top_ids] * self._gamma  # [S, k, d]
        V = (w @ J).float()  # [S, k, d]
        V = V * (~exempt_mask).unsqueeze(-1)
        hs = h[0].float()  # [S, d]
        c = torch.einsum("skd,sd->sk", V, hs)  # [S, k]
        G = torch.einsum("skd,smd->skm", V, V)  # [S, k, k]
        eye = torch.eye(self.k, device=h.device)
        # Regularization scaled to G's magnitude keeps zeroed (exempt) rows inert and
        # near-collinear directions numerically safe.
        reg = 1e-4 * G.diagonal(dim1=1, dim2=2).amax(1).clamp(min=1e-8)  # [S]
        coeffs = torch.linalg.solve(G + reg[:, None, None] * eye, c.unsqueeze(-1))
        delta = torch.einsum("skd,sk->sd", V, coeffs.squeeze(-1))  # [S, d]

        if self.mode == "random":
            delta = self._random_matched(delta, hs, S, d)

        h_out = h.clone()
        h_out[0] = (hs - delta).to(h.dtype)
        dn = delta.norm(dim=-1) / (hs.norm(dim=-1) + 1e-8)  # [S]
        self._norm_sum[layer] = self._norm_sum[layer] + dn.sum()
        self._norm_cnt[layer] += S
        return h_out

    def _random_matched(
        self, j_delta: torch.Tensor, hs: torch.Tensor, S: int, d: int
    ) -> torch.Tensor:
        """Equal-norm removal along k random directions at each slot (A.23).

        Directions come from a fixed pre-generated bank; indices are drawn on CPU
        (seeded, no device sync) per slot.
        """
        idx = torch.randint(0, _RAND_BANK_SIZE, (S, self.k), generator=self._rng)
        R = self._rand_bank[idx.to(hs.device)]  # [S, k, d]
        c = torch.einsum("skd,sd->sk", R, hs)
        G = torch.einsum("skd,smd->skm", R, R)
        eye = torch.eye(self.k, device=hs.device)
        reg = 1e-4 * G.diagonal(dim1=1, dim2=2).amax(1).clamp(min=1e-8)
        coeffs = torch.linalg.solve(G + reg[:, None, None] * eye, c.unsqueeze(-1))
        r = torch.einsum("skd,sk->sd", R, coeffs.squeeze(-1))  # [S, d]
        target = j_delta.norm(dim=-1, keepdim=True)
        return r * target / (r.norm(dim=-1, keepdim=True) + 1e-8)
