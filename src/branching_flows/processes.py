"""Base flow processes: bridge (training) and step (generation).

Minimal port of the Flowfusion.jl base processes required by BranchingFlows.

Each process provides two operations:

* ``bridge(x_src, x_tgt, t_cur, t_eval)`` -- sample the state at ``t_eval``
  from the conditional path starting at ``x_src`` (time ``t_cur``) and
  terminating at ``x_tgt`` (time 1).  Used during *training* to construct
  conditional bridge states along forest branches.  Operates on individual
  elements (unbatched tensors).

* ``step(xt, x1_pred, s1, s2)`` -- advance the state from ``s1`` to ``s2``
  using the model's endpoint prediction ``x1_pred``.  Used during *generation*
  (forward-time Euler integration).  Operates on batched tensors.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseProcess(ABC):
    """Abstract base for flow matching processes."""

    @abstractmethod
    def bridge(
        self,
        x_src: torch.Tensor,
        x_tgt: torch.Tensor,
        t_cur: float,
        t_eval: float,
    ) -> torch.Tensor:
        """Sample from the conditional path at *t_eval* (unbatched)."""

    @abstractmethod
    def step(
        self,
        xt: torch.Tensor,
        x1_pred: torch.Tensor,
        s1: float,
        s2: float,
    ) -> torch.Tensor:
        """Euler step from *s1* to *s2* using model prediction (batched)."""


# ---------------------------------------------------------------------------
# Continuous processes
# ---------------------------------------------------------------------------


class BrownianBridge(BaseProcess):
    """Brownian bridge process with constant diffusion *sigma*.

    Bridge distribution at ``t_eval`` given source at ``t_cur`` targeting
    ``x_tgt`` at t=1::

        alpha = (t_eval - t_cur) / (1 - t_cur)
        mean  = x_src + alpha * (x_tgt - x_src)
        var   = sigma^2 * (t_eval - t_cur) * (1 - t_eval) / (1 - t_cur)
        x     ~ N(mean, var * I)

    Step uses the OT velocity ``v = (x1_pred - xt) / (1 - s1)`` plus noise.
    """

    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma

    def bridge(
        self,
        x_src: torch.Tensor,
        x_tgt: torch.Tensor,
        t_cur: float,
        t_eval: float,
    ) -> torch.Tensor:
        denom = 1.0 - t_cur
        if denom <= 0:
            return x_tgt.clone()
        alpha = (t_eval - t_cur) / denom
        mean = x_src + alpha * (x_tgt - x_src)
        var = self.sigma**2 * (t_eval - t_cur) * (1.0 - t_eval) / denom
        if var <= 0:
            return mean
        return mean + math.sqrt(var) * torch.randn_like(mean)

    def step(
        self,
        xt: torch.Tensor,
        x1_pred: torch.Tensor,
        s1: float,
        s2: float,
    ) -> torch.Tensor:
        dt = s2 - s1
        denom = 1.0 - s1
        if denom <= 0:
            return x1_pred
        velocity = (x1_pred - xt) / denom
        noise = self.sigma * math.sqrt(dt) * torch.randn_like(xt)
        return xt + velocity * dt + noise


class OUFlow(BaseProcess):
    """Ornstein-Uhlenbeck process with time-dependent diffusion.

    SDE: ``dX_t = theta * (x1 - X_t) dt + sqrt(v_t) dW_t``

    where ``v_t`` interpolates linearly from ``var_0`` at t=0 to ``var_1``
    at t=1 (typically ``var_1`` is near 0 for a pinching bridge).

    The bridge conditional distribution remains Gaussian (see paper Appendix D).
    """

    def __init__(
        self,
        theta: float = 5.0,
        var_0: float = 10.0,
        var_1: float = 0.001,
    ):
        self.theta = theta
        self.var_0 = var_0
        self.var_1 = var_1

    def _var_t(self, t: float) -> float:
        return self.var_0 + (self.var_1 - self.var_0) * t

    def bridge(
        self,
        x_src: torch.Tensor,
        x_tgt: torch.Tensor,
        t_cur: float,
        t_eval: float,
    ) -> torch.Tensor:
        dt = t_eval - t_cur
        if dt <= 0 or t_eval >= 1.0:
            return x_tgt.clone()

        decay = math.exp(-self.theta * dt)
        mean = x_tgt + (x_src - x_tgt) * decay

        remaining = 1.0 - t_eval
        v_mid = self._var_t((t_cur + t_eval) / 2.0)
        var = v_mid * (1.0 - decay**2) / (2.0 * self.theta)
        # Scale down variance as we approach t=1
        var *= min(remaining / max(1.0 - t_cur, 1e-8), 1.0)

        if var <= 0:
            return mean
        return mean + math.sqrt(var) * torch.randn_like(mean)

    def step(
        self,
        xt: torch.Tensor,
        x1_pred: torch.Tensor,
        s1: float,
        s2: float,
    ) -> torch.Tensor:
        dt = s2 - s1
        drift = self.theta * (x1_pred - xt) * dt
        v_t = self._var_t(s1)
        diffusion = math.sqrt(v_t * dt) * torch.randn_like(xt)
        return xt + drift + diffusion


# ---------------------------------------------------------------------------
# Discrete process
# ---------------------------------------------------------------------------


class DiscreteInterpolatingFlow(BaseProcess):
    """Noisy interpolating discrete flow matching (DFM).

    At time *t*, the conditional distribution is a mixture:
    - ``kappa_1(t)`` probability of being the target token
    - ``kappa_2(t)`` probability of being a uniform random token
    - ``kappa_3(t)`` probability of being the source token

    Uses CDF schedulers ``F1``, ``F2`` (defaulting to Beta CDFs) and a
    uniform noise weight ``omega_u``.

    *K* is the vocabulary size including the mask/dummy token at index ``K-1``.
    """

    def __init__(
        self,
        K: int,
        omega_u: float = 0.2,
        beta_a1: float = 2.0,
        beta_b1: float = 2.0,
        beta_a2: float = 2.0,
        beta_b2: float = 2.0,
    ):
        from scipy.stats import beta as beta_dist

        self.K = K
        self.omega_u = omega_u
        self._F1 = beta_dist(beta_a1, beta_b1)
        self._F2 = beta_dist(beta_a2, beta_b2)

    def _kappas(self, t: float) -> tuple[float, float, float]:
        k1 = self._F1.cdf(t)
        k2 = self.omega_u * (1.0 - k1) * self._F2.cdf(t)
        k3 = 1.0 - k1 - k2
        return k1, k2, k3

    def bridge(
        self,
        x_src: torch.Tensor | int,
        x_tgt: torch.Tensor | int,
        t_cur: float,
        t_eval: float,
    ) -> torch.Tensor | int:
        """Sample discrete token at *t_eval* on the bridge from *x_src* to *x_tgt*.

        Conditional on being at *x_src* at *t_cur* and needing to reach
        *x_tgt* by t=1, the probability of each outcome at *t_eval* is
        derived from the DFM interpolant on ``[t_cur, 1]``.
        """
        _, _, k3_cur = self._kappas(t_cur)
        k1_eval, k2_eval, k3_eval = self._kappas(t_eval)

        if k3_cur <= 0:
            return x_tgt if isinstance(x_tgt, int) else x_tgt.clone()

        ratio = k3_eval / k3_cur
        p_target = k1_eval - (self._F1.cdf(t_cur)) * ratio
        p_target = max(p_target, 0.0)
        p_uniform = (
            k2_eval
            - (self.omega_u * (1.0 - self._F1.cdf(t_cur)) * self._F2.cdf(t_cur)) * ratio
        )
        p_uniform = max(p_uniform, 0.0)
        p_source = max(ratio, 0.0)

        total = p_target + p_uniform + p_source
        if total <= 0:
            return x_tgt if isinstance(x_tgt, int) else x_tgt.clone()
        p_target /= total
        p_uniform /= total

        u = torch.rand(1).item()
        if u < p_target:
            return x_tgt if isinstance(x_tgt, int) else x_tgt.clone()
        if u < p_target + p_uniform:
            val = torch.randint(0, self.K, (1,)).item()
            return (
                val if isinstance(x_tgt, int) else torch.tensor(val, dtype=x_tgt.dtype)
            )
        return x_src if isinstance(x_src, int) else x_src.clone()

    def step(
        self,
        xt: torch.Tensor,
        x1_pred: torch.Tensor,
        s1: float,
        s2: float,
    ) -> torch.Tensor:
        """Euler step for discrete tokens.

        *x1_pred* should be logits of shape ``(batch, length, K)`` or
        category indices of shape ``(batch, length)``.

        Returns updated category indices ``(batch, length)``.
        """
        dt = s2 - s1
        k1, k2, k3 = self._kappas(s1)
        if k3 <= 0:
            if x1_pred.dim() > xt.dim():
                return x1_pred.argmax(dim=-1)
            return x1_pred.clone()

        # Transition rate: probability of jumping away from current token
        dk1 = self._F1.pdf(s1)
        dk3 = -(
            dk1
            + self.omega_u
            * (-(dk1) * self._F2.cdf(s1) + (1.0 - self._F1.cdf(s1)) * self._F2.pdf(s1))
        )
        rate = max(-(dk3 / k3), 0.0)
        p_jump = 1.0 - math.exp(-rate * dt)

        jump_mask = torch.rand_like(xt, dtype=torch.float32) < p_jump

        if x1_pred.dim() > xt.dim():
            new_tokens = x1_pred.argmax(dim=-1)
        else:
            new_tokens = x1_pred

        result = torch.where(jump_mask, new_tokens, xt)
        return result


# ---------------------------------------------------------------------------
# Multi-process helpers
# ---------------------------------------------------------------------------


def bridge_multi(
    processes: BaseProcess | tuple[BaseProcess, ...] | list[BaseProcess],
    x_srcs: Any,
    x_tgts: Any,
    t_cur: float,
    t_eval: float,
) -> Any:
    """Bridge each component of a (possibly multimodal) state independently."""
    if isinstance(processes, (list, tuple)):
        return tuple(
            p.bridge(xs, xt, t_cur, t_eval)
            for p, xs, xt in zip(processes, x_srcs, x_tgts)
        )
    return processes.bridge(x_srcs, x_tgts, t_cur, t_eval)


def step_multi(
    processes: BaseProcess | tuple[BaseProcess, ...] | list[BaseProcess],
    xts: Any,
    x1_preds: Any,
    s1: float,
    s2: float,
) -> Any:
    """Step each component of a (possibly multimodal) state independently."""
    if isinstance(processes, (list, tuple)):
        return tuple(
            p.step(xt, x1p, s1, s2) for p, xt, x1p in zip(processes, xts, x1_preds)
        )
    return processes.step(xts, x1_preds, s1, s2)
