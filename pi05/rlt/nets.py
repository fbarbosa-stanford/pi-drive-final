"""Actor / critic networks for the RLT learner core.

All nets are small MLPs over the flat state ``x = [z_rl, s_p]`` and (for the
actor) the reference chunk ``a_ref``. The actor is deliberately self-contained
(raw tensors in, action out, no buffer/optimiser coupling) so it TorchScript- /
ONNX-exports cleanly to run later under the friend's PyTorch/TensorRT runtime.

Design choices fixed by the spec:
  - Gaussian policy with a *fixed* per-dim std (no log-std head, no entropy term).
  - The mean is tanh-squashed into the native action bounds (accel, curvature),
    so the actor always proposes a kinematically valid edit regardless of a_ref
    scale; the downstream safety layer still clamps as a hard backstop.
  - Critic is an ensemble of >= 2 Q-nets with Polyak target copies; min-of-N is
    used for both the TD target and the (pessimistic) policy objective.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def build_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """pi_theta(a | z_rl, s_p, a_ref) = N(mu_theta, sigma^2 I), fixed sigma.

    Inputs are flattened and concatenated: [z_rl, s_p, flatten(a_ref)]. Output
    mean is tanh-squashed per action dim into [low, high]. ``forward`` returns
    the deterministic mean (what inference uses); use ``rsample`` for training.
    """

    def __init__(
        self,
        state_dim: int,
        chunk_len: int,
        action_dim: int,
        hidden: Sequence[int],
        action_low: Sequence[float],
        action_high: Sequence[float],
        std: Sequence[float],
    ) -> None:
        super().__init__()
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.action_flat = chunk_len * action_dim
        in_dim = state_dim + self.action_flat
        self.trunk = build_mlp(in_dim, hidden, self.action_flat)

        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("a_center", (low + high) / 2.0)
        self.register_buffer("a_half", (high - low) / 2.0)
        self.register_buffer("a_std", torch.as_tensor(std, dtype=torch.float32))

    def _flat_inputs(self, z_rl: Tensor, s_p: Tensor, a_ref: Tensor) -> Tensor:
        a_ref_flat = a_ref.reshape(a_ref.shape[0], -1)
        return torch.cat([z_rl, s_p, a_ref_flat], dim=-1)

    def mean(self, z_rl: Tensor, s_p: Tensor, a_ref: Tensor) -> Tensor:
        raw = self.trunk(self._flat_inputs(z_rl, s_p, a_ref))
        raw = raw.view(-1, self.chunk_len, self.action_dim)
        return self.a_center + self.a_half * torch.tanh(raw)

    def forward(self, z_rl: Tensor, s_p: Tensor, a_ref: Tensor) -> Tensor:
        # Deterministic mean -- the inference/export path (always sees a_ref).
        return self.mean(z_rl, s_p, a_ref)

    def rsample(self, z_rl: Tensor, s_p: Tensor, a_ref: Tensor) -> Tensor:
        """Reparameterized sample: mu + sigma * eps. Grad flows through mu."""
        mu = self.mean(z_rl, s_p, a_ref)
        eps = torch.randn_like(mu)
        return mu + self.a_std * eps


class QEnsemble(nn.Module):
    """Ensemble of N critics Q(z_rl, s_p, a) -> scalar."""

    def __init__(
        self,
        state_dim: int,
        action_flat_dim: int,
        hidden: Sequence[int],
        n_critics: int,
    ) -> None:
        super().__init__()
        in_dim = state_dim + action_flat_dim
        self.nets = nn.ModuleList(
            build_mlp(in_dim, hidden, 1) for _ in range(n_critics)
        )

    def forward(self, z_rl: Tensor, s_p: Tensor, a: Tensor) -> Tensor:
        """Return per-critic Q values, shape (n_critics, B)."""
        x = torch.cat([z_rl, s_p, a.reshape(a.shape[0], -1)], dim=-1)
        return torch.stack([net(x).squeeze(-1) for net in self.nets], dim=0)

    def min_q(self, z_rl: Tensor, s_p: Tensor, a: Tensor) -> Tensor:
        """min over the ensemble, shape (B,)."""
        return self.forward(z_rl, s_p, a).min(dim=0).values


@torch.no_grad()
def polyak_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    """target <- tau * online + (1 - tau) * target, in place."""
    for p_t, p_o in zip(target.parameters(), online.parameters(), strict=True):
        p_t.mul_(1.0 - tau).add_(p_o, alpha=tau)
    for b_t, b_o in zip(target.buffers(), online.buffers(), strict=True):
        b_t.copy_(b_o)
