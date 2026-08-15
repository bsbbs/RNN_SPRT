"""
Unified runner for the six noisy observers.

Examples
--------
From Python:

    from noise_model_runner import run_model

    result, df = run_model(noise="mixed", horizon="infinite")
    result, df = run_model(noise="representation", horizon="finite")
    result, df = run_model(noise="execution", horizon="finite")

From the command line:

    python normative/noise_model_runner.py --noise mixed --horizon infinite
    python normative/noise_model_runner.py --noise representation --horizon finite --n-trials 10000
    python normative/noise_model_runner.py --noise execution --horizon finite --save-dir normative/figures
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

try:
    from typing import Literal
except ImportError:  # pragma: no cover - for older Python environments.
    from typing_extensions import Literal

tmp_cache = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(tmp_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(tmp_cache / "xdg-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

plt.rcParams.update({
    'font.family': 'Arial',
    'axes.labelsize': 9,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.75,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.loc': 'upper right'
})


NoiseKind = Literal["representation", "execution", "mixed"]
HorizonKind = Literal["finite", "infinite"]
DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "figures"


@dataclass
class ModelConfig:
    c: float = 0.03
    sigma_repr: float = 0.25
    tau_exec: float = 0.025
    r_correct: float = 1.0
    r_wrong: float = 0.0
    n_trials: int = 5 * 10**4
    max_timestep: int = 10
    deadline: int = 11
    seed: int = 1
    mu: float = 2.2
    sigma: float = 5.0
    l_min: float = 3.5 #-13
    l_max: float = 3.5 # 13
    n_l: int = 2001
    n_gh: int = 401
    tol: float = 1e-9
    max_iter: int = 5000


@dataclass
class StimulusSet:
    evidence_log10: np.ndarray
    p_a: np.ndarray
    p_b: np.ndarray


@dataclass
class ModelResult:
    noise: NoiseKind
    horizon: HorizonKind
    config: ModelConfig
    stimuli: StimulusSet
    l_grid: np.ndarray
    b_grid: np.ndarray
    q0: np.ndarray
    q1: np.ndarray
    v_choose: np.ndarray
    v: np.ndarray
    q_wait: np.ndarray
    lower_l: Union[np.ndarray, float]
    upper_l: Union[np.ndarray, float]
    lower_b: Union[np.ndarray, float]
    upper_b: Union[np.ndarray, float]
    p_sample: np.ndarray
    p_choose0: np.ndarray
    p_choose1: np.ndarray
    gh_x: np.ndarray
    gh_w: np.ndarray

    @property
    def has_representation_noise(self) -> bool:
        return self.noise in {"representation", "mixed"}

    @property
    def has_execution_noise(self) -> bool:
        return self.noise in {"execution", "mixed"}


def normalize_noise(noise: str) -> NoiseKind:
    aliases = {
        "noisy": "representation",
        "noisy_repr": "representation",
        "noisy_repre": "representation",
        "representation": "representation",
        "repr": "representation",
        "stochastic": "execution",
        "stochastic_execution": "execution",
        "execution": "execution",
        "mixed": "mixed",
    }
    key = noise.lower().replace("-", "_")
    if key not in aliases:
        raise ValueError(f"Unknown noise={noise!r}; use representation, execution, or mixed.")
    return aliases[key]  # type: ignore[return-value]


def normalize_horizon(horizon: str) -> HorizonKind:
    key = horizon.lower().replace("-", "_")
    if key not in {"finite", "infinite"}:
        raise ValueError(f"Unknown horizon={horizon!r}; use finite or infinite.")
    return key  # type: ignore[return-value]


def logistic_log10(l_value: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    return 1.0 / (1.0 + 10.0 ** (-l_value))


def log10_odds(b_value: Union[np.ndarray, float], eps: float = 1e-12) -> Union[np.ndarray, float]:
    b_clipped = np.clip(b_value, eps, 1.0 - eps)
    return np.log10(b_clipped / (1.0 - b_clipped))

def build_stimuli(config: ModelConfig) -> StimulusSet:
    """Closed-form stimulus set for the equal-variance Gaussian generative model.

    Each hypothesis emits a scalar cue ``x`` from a Gaussian:

        H1 (target B):  x ~ N(+mu, sigma^2)
        H0 (target A):  x ~ N(-mu, sigma^2)

    Because the two Gaussians share the same sigma, the quadratic terms in the
    log-likelihood ratio cancel and the per-sample log10 LR is *linear* in x:

        log10[ N(x; mu, s) / N(x; -mu, s) ]
            = ( -(x - mu)^2 + (x + mu)^2 ) / (2 s^2 ln 10)
            = ( 4 mu x ) / (2 s^2 ln 10)
            = ( 2 mu / (s^2 ln 10) ) * x .

    The x that produces a target log10 LR ``ell`` therefore has the exact
    inverse (no grid / interpolation needed):

        x(ell) = ell * s^2 ln 10 / (2 mu).

    The sampling frequency of that cue under each hypothesis is the Gaussian
    density at x(ell), normalized across the stimulus set:

        p_b(ell) ∝ exp( -(x(ell) - mu)^2 / (2 s^2) )
        p_a(ell) ∝ exp( -(x(ell) + mu)^2 / (2 s^2) )

    This reproduces the target log10 LR analytically.
    """
    mu = config.mu
    sigma = config.sigma
    ln10 = np.log(10.0)

    evidence_log10 = np.array(
        [
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
        ],
        dtype=float,
    )

    # Closed-form inverse of the linear log10 likelihood-ratio map.
    x_targets = evidence_log10 * (sigma ** 2 * ln10) / (2.0 * mu)

    # Gaussian sampling frequencies at those cue values (analytic densities).
    p_b_raw = norm.pdf(x_targets, mu, sigma)
    p_a_raw = norm.pdf(x_targets, -mu, sigma)

    return StimulusSet(
        evidence_log10=evidence_log10,
        p_a=p_a_raw / p_a_raw.sum(),
        p_b=p_b_raw / p_b_raw.sum(),
    )


def softmax_3actions(q0: np.ndarray, q1: np.ndarray, q_wait: np.ndarray, tau: float) -> np.ndarray:
    q_stack = np.vstack([q0, q1, q_wait]) / tau
    q_stack = q_stack - np.max(q_stack, axis=0, keepdims=True)
    exp_q = np.exp(q_stack)
    return exp_q / exp_q.sum(axis=0, keepdims=True)


def interp_on_l_grid(l_query: np.ndarray, l_grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    l_query = np.clip(l_query, l_grid[0], l_grid[-1])
    return np.interp(l_query, l_grid, values)


def wait_value(
    next_v: np.ndarray,
    l_grid: np.ndarray,
    b_grid: np.ndarray,
    stimuli: StimulusSet,
    cost: float,
    sigma_repr: float,
    gh_x: np.ndarray,
    gh_w: np.ndarray,
) -> np.ndarray:
    q_wait = np.zeros_like(l_grid)

    if sigma_repr > 0:
        noise_nodes = np.sqrt(2.0) * sigma_repr * gh_x
        noise_weights = gh_w / np.sqrt(np.pi)
    else:
        noise_nodes = np.array([0.0])
        noise_weights = np.array([1.0])

    for i, ell_i in enumerate(stimuli.evidence_log10):
        pred_i = b_grid * stimuli.p_b[i] + (1.0 - b_grid) * stimuli.p_a[i]
        expected_over_noise = np.zeros_like(l_grid)

        for node, weight in zip(noise_nodes, noise_weights):
            l_next = l_grid + ell_i + node
            expected_over_noise += weight * interp_on_l_grid(l_next, l_grid, next_v)

        q_wait += pred_i * expected_over_noise

    return -cost + q_wait


def boundaries_from_continue_region(
    continue_region: np.ndarray,
    l_grid: np.ndarray,
) -> tuple[float, float, float, float]:
    if np.any(continue_region):
        lower_l = float(l_grid[continue_region][0])
        upper_l = float(l_grid[continue_region][-1])
    else:
        lower_l = 0.0
        upper_l = 0.0

    lower_b = float(logistic_log10(lower_l))
    upper_b = float(logistic_log10(upper_l))
    return lower_l, upper_l, lower_b, upper_b


def compute_model(
    noise: str = "mixed",
    horizon: str = "infinite",
    config: Optional[ModelConfig] = None,
    verbose: bool = True,
    noisy: Optional[str] = None,
    horzion: Optional[str] = None,
) -> ModelResult:
    if noisy is not None:
        noise = noisy
    if horzion is not None:
        horizon = horzion

    noise_kind = normalize_noise(noise)
    horizon_kind = normalize_horizon(horizon)
    cfg = config or ModelConfig()

    n_gh = cfg.n_gh
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_gh) # numerical approximation of the Gaussian kernel
    stimuli = build_stimuli(cfg)

    l_grid = np.linspace(cfg.l_min, cfg.l_max, cfg.n_l)
    b_grid = logistic_log10(l_grid)
    q0 = (1.0 - b_grid) * cfg.r_correct + b_grid * cfg.r_wrong
    q1 = b_grid * cfg.r_correct + (1.0 - b_grid) * cfg.r_wrong
    v_choose = np.maximum(q0, q1)
    sigma_for_value = cfg.sigma_repr if noise_kind in {"representation", "mixed"} else 0.0

    if horizon_kind == "finite": # backward induction of V_star based on Bellman operator for finite horizon
        v = np.zeros((cfg.deadline + 1, cfg.n_l))
        q_wait = np.zeros((cfg.deadline + 1, cfg.n_l))
        p_sample = np.zeros((cfg.deadline + 1, cfg.n_l))
        p_choose0 = np.zeros((cfg.deadline + 1, cfg.n_l))
        p_choose1 = np.zeros((cfg.deadline + 1, cfg.n_l))
        lower_l = np.full(cfg.deadline + 1, np.nan)
        upper_l = np.full(cfg.deadline + 1, np.nan)
        lower_b = np.full(cfg.deadline + 1, np.nan)
        upper_b = np.full(cfg.deadline + 1, np.nan)

        v[cfg.deadline, :] = 0.0
        for t in range(cfg.deadline - 1, 0, -1):
            q_wait[t, :] = wait_value(
                next_v=v[t + 1, :],
                l_grid=l_grid,
                b_grid=b_grid,
                stimuli=stimuli,
                cost=cfg.c,
                sigma_repr=sigma_for_value,
                gh_x=gh_x,
                gh_w=gh_w,
            )
            v[t, :] = np.maximum(v_choose, q_wait[t, :])
            continue_region = q_wait[t, :] > v_choose
            lower_l[t], upper_l[t], lower_b[t], upper_b[t] = boundaries_from_continue_region(
                continue_region,
                l_grid,
            )

            if noise_kind in {"execution", "mixed"}:
                probs = softmax_3actions(q0, q1, q_wait[t, :], cfg.tau_exec)
                p_choose0[t, :] = probs[0, :]
                p_choose1[t, :] = probs[1, :]
                p_sample[t, :] = probs[2, :]
            else:
                p_choose0[t, :] = l_grid <= lower_l[t]
                p_choose1[t, :] = l_grid >= upper_l[t]
                p_sample[t, :] = continue_region.astype(float)

        if verbose:
            print_model_summary(noise_kind, horizon_kind, cfg, lower_l, upper_l, lower_b, upper_b)

    else: # induction of V_star using value iteration strategy for infinite horizon
        v = v_choose.copy()
        q_wait = np.zeros_like(l_grid)

        for iteration in range(cfg.max_iter):
            v_old = v.copy()
            q_wait = wait_value(
                next_v=v_old,
                l_grid=l_grid,
                b_grid=b_grid,
                stimuli=stimuli,
                cost=cfg.c,
                sigma_repr=sigma_for_value,
                gh_x=gh_x,
                gh_w=gh_w,
            )
            v = np.maximum(v_choose, q_wait)
            diff = np.max(np.abs(v - v_old))
            if diff < cfg.tol:
                if verbose:
                    print(f"Value iteration converged after {iteration + 1} iterations.")
                break
        else:
            if verbose:
                print("Warning: value iteration did not converge.")

        continue_region = q_wait > v_choose
        lower_l, upper_l, lower_b, upper_b = boundaries_from_continue_region(continue_region, l_grid)

        if noise_kind in {"execution", "mixed"}:
            probs = softmax_3actions(q0, q1, q_wait, cfg.tau_exec)
            p_choose0 = probs[0, :]
            p_choose1 = probs[1, :]
            p_sample = probs[2, :]
        else:
            p_choose0 = (l_grid <= lower_l).astype(float)
            p_choose1 = (l_grid >= upper_l).astype(float)
            p_sample = continue_region.astype(float)

        if verbose:
            print_model_summary(noise_kind, horizon_kind, cfg, lower_l, upper_l, lower_b, upper_b)

    return ModelResult(
        noise=noise_kind,
        horizon=horizon_kind,
        config=cfg,
        stimuli=stimuli,
        l_grid=l_grid,
        b_grid=b_grid,
        q0=q0,
        q1=q1,
        v_choose=v_choose,
        v=v,
        q_wait=q_wait,
        lower_l=lower_l,
        upper_l=upper_l,
        lower_b=lower_b,
        upper_b=upper_b,
        p_sample=p_sample,
        p_choose0=p_choose0,
        p_choose1=p_choose1,
        gh_x=gh_x,
        gh_w=gh_w,
    )


def print_model_summary(
    noise: NoiseKind,
    horizon: HorizonKind,
    config: ModelConfig,
    lower_l: Union[np.ndarray, float],
    upper_l: Union[np.ndarray, float],
    lower_b: Union[np.ndarray, float],
    upper_b: Union[np.ndarray, float],
) -> None:
    print(f"{horizon}-horizon {noise} observer")
    print(f"c={config.c}, sigma_repr={config.sigma_repr}, tau_exec={config.tau_exec}")

    if horizon == "finite":
        for t in range(1, config.max_timestep + 1):
            print(
                f"t={t:2d}: "
                f"lower_L={lower_l[t]: .3f}, upper_L={upper_l[t]: .3f}, "
                f"lower_b={lower_b[t]: .3f}, upper_b={upper_b[t]: .3f}"
            )
    else:
        print(f"lower_L={lower_l: .3f}, upper_L={upper_l: .3f}")
        print(f"lower_b={lower_b: .3f}, upper_b={upper_b: .3f}")


def value_arrays_for_time(result: ModelResult, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if result.horizon == "finite":
        return (
            result.q_wait[t, :],
            result.p_choose0[t, :],
            result.p_choose1[t, :],
            result.p_sample[t, :],
        )
    return result.q_wait, result.p_choose0, result.p_choose1, result.p_sample


def boundaries_for_time(result: ModelResult, t: int) -> tuple[float, float, float, float]:
    if result.horizon == "finite":
        return (
            float(result.lower_l[t]),
            float(result.upper_l[t]),
            float(result.lower_b[t]),
            float(result.upper_b[t]),
        )
    return (
        float(result.lower_l),
        float(result.upper_l),
        float(result.lower_b),
        float(result.upper_b),
    )


def simulate_trials(
    result: ModelResult,
    n_trials: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    cfg = result.config
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    n_trials = cfg.n_trials if n_trials is None else n_trials
    rows = []

    for trial in range(n_trials):
        true_h = int(rng.integers(0, 2))
        ground_l = 0.0
        represented_l = 0.0

        for t in range(1, cfg.max_timestep + 1):
            if true_h == 0:
                stim_idx = int(rng.choice(len(result.stimuli.p_a), p=result.stimuli.p_a))
            else:
                stim_idx = int(rng.choice(len(result.stimuli.p_b), p=result.stimuli.p_b))

            ell = float(result.stimuli.evidence_log10[stim_idx])
            ground_l += ell
            represented_l += ell

            if result.has_representation_noise:
                represented_l += float(rng.normal(0.0, cfg.sigma_repr))

            ground_b = float(logistic_log10(ground_l))

            q_wait_t, _, _, _ = value_arrays_for_time(result, t)
            q0_t = float(np.interp(represented_l, result.l_grid, result.q0))
            q1_t = float(np.interp(represented_l, result.l_grid, result.q1))
            qw_t = float(np.interp(represented_l, result.l_grid, q_wait_t))

            if result.has_execution_noise:
                probs_t = softmax_3actions(
                    np.array([q0_t]),
                    np.array([q1_t]),
                    np.array([qw_t]),
                    cfg.tau_exec,
                ).flatten()
                action = int(rng.choice([0, 1, 2], p=probs_t))
            else:
                lower_l, upper_l, _, _ = boundaries_for_time(result, t)
                if represented_l <= lower_l:
                    action = 0
                    probs_t = np.array([1.0, 0.0, 0.0])
                elif represented_l >= upper_l:
                    action = 1
                    probs_t = np.array([0.0, 1.0, 0.0])
                else:
                    action = 2
                    probs_t = np.array([0.0, 0.0, 1.0])

            rows.append(
                {
                    "trial": trial,
                    "true_H": true_h,
                    "time_step": t,
                    "stim_idx": stim_idx,
                    "ground_L": ground_l,
                    "ground_b": ground_b,
                    "evidence_sum": ground_l,
                    "belief": ground_b,
                    "action": action,
                    "p_choose0": probs_t[0],
                    "p_choose1": probs_t[1],
                    "p_sample": probs_t[2],
                    "p_sample_model": float(p_sample_ground_view(result, np.array([ground_l]), t)[0]),
                }
            )

            if action in {0, 1}:
                break

    df = pd.DataFrame(rows)
    if verbose:
        print(df.head())
        print(df["action"].value_counts())
    return df


def empirical_policy_by_bins(
    df: pd.DataFrame,
    y_col: str,
    y_min: float,
    y_max: float,
    max_timestep: int,
    n_bins: int = 60,
    min_count: int = 10,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    bins = np.linspace(y_min, y_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    out = {}

    for t in range(1, max_timestep + 1):
        dft = df[df["time_step"] == t]
        y = dft[y_col].to_numpy()
        is_sample = (dft["action"] == 2).astype(float).to_numpy()
        p_emp = np.full_like(centers, np.nan, dtype=float)

        for k in range(n_bins):
            mask = (y >= bins[k]) & (y < bins[k + 1])
            if mask.sum() >= min_count:
                p_emp[k] = is_sample[mask].mean()

        out[t] = p_emp

    return centers, out


def p_sample_internal(result: ModelResult, y_grid: np.ndarray, t: int) -> np.ndarray:
    _, _, _, p_sample_t = value_arrays_for_time(result, t)
    return np.interp(
        y_grid,
        result.l_grid,
        p_sample_t,
        left=float(p_sample_t[0]),
        right=float(p_sample_t[-1]),
    )


def p_sample_ground_view(result: ModelResult, y_grid: np.ndarray, t: int) -> np.ndarray:
    if not result.has_representation_noise:
        return p_sample_internal(result, y_grid, t)

    s_t = result.config.sigma_repr * np.sqrt(t)
    if s_t == 0:
        return p_sample_internal(result, y_grid, t)

    if result.noise == "representation":
        lower_l, upper_l, _, _ = boundaries_for_time(result, t)
        return norm.cdf((upper_l - y_grid) / s_t) - norm.cdf((lower_l - y_grid) / s_t)

    p_out = np.zeros_like(y_grid)
    nodes_t = np.sqrt(2.0) * s_t * result.gh_x
    weights_t = result.gh_w / np.sqrt(np.pi)
    _, _, _, p_sample_t = value_arrays_for_time(result, t)

    for node, weight in zip(nodes_t, weights_t):
        p_out += weight * np.interp(
            y_grid + node,
            result.l_grid,
            p_sample_t,
            left=float(p_sample_t[0]),
            right=float(p_sample_t[-1]),
        )

    return p_out


def plot_value_summary(result: ModelResult, show: bool = True) -> plt.Figure:
    cfg = result.config
    time_colors = plt.cm.tab10(np.arange(cfg.max_timestep))

    if result.horizon == "finite":
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
        ax_value, ax_boundary, ax_policy = axes

        ax_value.plot(result.b_grid, result.v_choose, color="black", linewidth=2, label=r"$V_{choose}$")
        for t in range(1, cfg.max_timestep + 1):
            ax_value.plot(
                result.b_grid,
                result.q_wait[t, :],
                color=time_colors[t - 1],
                linewidth=1.2,
                label=f"t={t}",
            )
        ax_value.set_xlabel(r"Internal belief $\tilde b=P(H_1)$")
        ax_value.set_ylabel("Value")
        ax_value.set_title("Wait value")
        ax_value.legend(fontsize=6, ncol=2, frameon=True)

        t_grid = np.arange(1, cfg.max_timestep + 1)
        ax_boundary.plot(t_grid, result.lower_l[1 : cfg.max_timestep + 1], marker="o", color="blue")
        ax_boundary.plot(
            t_grid,
            result.upper_l[1 : cfg.max_timestep + 1],
            marker="s",
            linestyle="--",
            color="red",
        )
        ax_boundary.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax_boundary.set_xlabel("Time step")
        ax_boundary.set_ylabel(r"Internal log10 odds $\tilde L$")
        ax_boundary.set_xticks(t_grid)
        ax_boundary.set_title("Boundaries")

        for t in range(1, cfg.max_timestep + 1):
            ax_policy.plot(
                result.b_grid,
                result.p_sample[t, :],
                color=time_colors[t - 1],
                linewidth=1.5,
                label=f"t={t}",
            )
        ax_policy.set_xlabel(r"Internal belief $\tilde b=P(H_1)$")
        ax_policy.set_ylabel(r"$p(sample)$")
        ax_policy.set_ylim(-0.05, 1.05)
        ax_policy.set_title("Sampling policy")
        ax_policy.legend(fontsize=6, ncol=2, frameon=True)

    else:
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
        ax_value, ax_policy = axes

        ax_value.plot(result.b_grid, result.v, linewidth=2, label=r"$V^*$")
        ax_value.plot(result.b_grid, result.v_choose, "--", linewidth=2, label=r"$V_{choose}$")
        ax_value.plot(result.b_grid, result.q_wait, ":", linewidth=2, label=r"$Q_W$")
        ax_value.axvline(float(result.lower_b), color="black", linestyle=":", linewidth=1)
        ax_value.axvline(float(result.upper_b), color="black", linestyle=":", linewidth=1)
        ax_value.set_xlabel(r"Internal belief $\tilde b=P(H_1)$")
        ax_value.set_ylabel("Value")
        ax_value.set_title("Value functions")
        ax_value.legend(fontsize=8)

        ax_policy.plot(result.b_grid, result.p_choose0, color="blue", linewidth=2, label=r"Choose $H_0$")
        ax_policy.plot(result.b_grid, result.p_choose1, color="red", linewidth=2, label=r"Choose $H_1$")
        ax_policy.plot(result.b_grid, result.p_sample, color="green", linewidth=2, label="Sample")
        ax_policy.axvline(float(result.lower_b), color="black", linestyle=":", linewidth=1)
        ax_policy.axvline(float(result.upper_b), color="black", linestyle=":", linewidth=1)
        ax_policy.set_xlabel(r"Internal belief $\tilde b=P(H_1)$")
        ax_policy.set_ylabel("Action probability")
        ax_policy.set_ylim(-0.05, 1.05)
        ax_policy.set_title("Policy")
        ax_policy.legend(fontsize=8)

    fig.suptitle(model_title(result), fontsize=10)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _compute_evidence_distribution(
    ev,
    distribution,
    y_min,
    y_max,
    kernel_sigma=1.0,
    kernel_bin_width=0.05,
):
    ev = np.asarray(ev, dtype=float)
    if ev.size == 0:
        return np.array([]), np.array([])

    if distribution == "discrete":
        return np.unique(np.round(ev, 10), return_counts=True)

    if distribution != "kernel":
        raise ValueError(
            f"distribution must be 'discrete' or 'kernel', got {distribution!r}"
        )

    edges = np.arange(y_min, y_max + kernel_bin_width, kernel_bin_width)
    counts, edges = np.histogram(ev, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    if kernel_sigma > 0:
        counts = gaussian_filter(
            counts.astype(float),
            sigma=kernel_sigma,
            mode="nearest",
        )
    return centers, counts


def plot_distribution_and_policy(
    result: ModelResult,
    df: pd.DataFrame,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    show_empirical_policy: bool = False,
    distribution: Literal["discrete", "kernel"] = "discrete",
    kernel_sigma: float = 1.0,
    kernel_bin_width: float = 0.05,
    show: bool = True,
) -> tuple[plt.Figure, plt.Figure]:
    cfg = result.config
    y_col = "ground_L"
    y_label =  r"Cumulative evidence"
    y_min = -2.5 if y_min is None and result.horizon == "infinite" else (-2.0 if y_min is None else y_min)
    y_max = 2.5 if y_max is None and result.horizon == "infinite" else (2.0 if y_max is None else y_max)
    y_grid = np.linspace(y_min, y_max, 500)
    time_colors = plt.cm.plasma(np.linspace(0.1, 0.95, cfg.max_timestep))

    colors = {0: "blue", 1: "red", 2: "green"}
    labels = {0: r"Chosen $H_0$", 1: r"Chosen $H_1$", 2: "Sampling"}

    fig_dist, dist_axes = plt.subplots(
        1,
        cfg.max_timestep,
        figsize=(4.85, 2.05),
        sharey=True,
    )
    if cfg.max_timestep == 1:
        dist_axes = np.array([dist_axes])

    for t in range(1, cfg.max_timestep + 1):
        ax = dist_axes[t - 1]
        timestep_data = df[df["time_step"] == t]
        max_count = 0.0

        for action in [0, 1, 2]:
            action_data = timestep_data[timestep_data["action"] == action]
            if action_data.empty:
                continue

            y = action_data[y_col].to_numpy()
            y_vals, counts = _compute_evidence_distribution(
                y,
                distribution=distribution,
                y_min=y_min,
                y_max=y_max,
                kernel_sigma=kernel_sigma,
                kernel_bin_width=kernel_bin_width,
            )
            if y_vals.size == 0:
                continue

            max_count = max(max_count, float(counts.max()))

            fill_kwargs = {
                "color": colors[action],
                "alpha": 0.3,
                "label": labels[action] if t == 1 else None,
                "linewidth": 0,
            }
            if distribution == "discrete":
                fill_kwargs["step"] = "mid"

            ax.fill_betweenx(
                y_vals,
                0,
                counts,
                **fill_kwargs,
            )

            if action in {0, 1}:
                ax.axhline(np.mean(y), color=colors[action], linewidth=1.4)

        lower_l, upper_l, _, _ = boundaries_for_time(result, t)
        ax.axhline(
            lower_l,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        ax.axhline(
            upper_l,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )

        if t == 1:
            ax.legend(loc="upper left", fontsize=7, frameon=True)

        ax.set_ylim(y_min, y_max)
        if max_count > 0:
            ax.set_xlim(0, max_count * 1.05)
        else:
            ax.set_xlim(0, 1)

        ax.set_xticks([])
        ax.set_xlabel(str(t), fontsize=8)
        ax.label_outer()
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    fig_dist.text(0.54, 0.03, "Time step", ha="center", fontsize=8)
    fig_dist.text(0.04, 0.5, y_label, va="center", rotation="vertical", fontsize=9)
    dist_axes[0].legend(loc="best", fontsize=7)
    fig_dist.subplots_adjust(left=0.10, right=0.995, bottom=0.20, top=0.98, wspace=0.00)

    centers, empirical_policy = empirical_policy_by_bins(
        df,
        y_col=y_col,
        y_min=y_min,
        y_max=y_max,
        max_timestep=cfg.max_timestep,
        n_bins=60,
    )

    fig_policy, policy_ax = plt.subplots(1, 1, figsize=(2.05, 2.55))

    for t in range(1, cfg.max_timestep + 1):
        p_sample_y = p_sample_ground_view(result, y_grid, t)
        color_t = time_colors[t - 1]
        policy_ax.plot(p_sample_y, y_grid, color=color_t, linewidth=1.6, label=f"t={t}")

        if show_empirical_policy:
            policy_ax.plot(
                empirical_policy[t],
                centers,
                color=color_t,
                linestyle="--",
                linewidth=1.1,
                alpha=0.85,
            )

    policy_ax.set_xlim(-0.05, 1.05)
    policy_ax.set_xticks([0, 0.5, 1.0])
    policy_ax.set_ylim(y_min, y_max)
    policy_ax.set_xlabel(r"$p(\mathrm{sample})$", fontsize=10)
    policy_ax.set_ylabel(y_label, fontsize=10)
    policy_ax.tick_params(axis="both", labelsize=9)

    sm = plt.cm.ScalarMappable(
        cmap="plasma",
        norm=plt.Normalize(vmin=1, vmax=cfg.max_timestep),
    )
    sm.set_array([])
    cbar = fig_policy.colorbar(sm, ax=policy_ax, fraction=0.08, pad=0.08)
    cbar.set_label("Time step", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig_policy.tight_layout()
    if show:
        plt.show()
    return fig_dist, fig_policy


def model_title(result: ModelResult) -> str:
    labels = {
        "representation": "Noisy representation",
        "execution": "Stochastic execution",
        "mixed": "Mixed noise",
    }
    return f"{result.horizon.capitalize()}-horizon {labels[result.noise]}"


def run_model(
    noise: str = "mixed",
    horizon: str = "infinite",
    config: Optional[ModelConfig] = None,
    plot: bool = True,
    show: bool = True,
    save_dir: Optional[Union[str, Path]] = DEFAULT_SAVE_DIR,
    show_empirical_policy: bool = False,
    verbose: bool = True,
    noisy: Optional[str] = None,
    horzion: Optional[str] = None,
) -> tuple[ModelResult, pd.DataFrame]:
    if noisy is not None:
        noise = noisy
    if horzion is not None:
        horizon = horzion

    result = compute_model(noise=noise, horizon=horizon, config=config, verbose=verbose)
    df = simulate_trials(result, verbose=verbose)

    if plot:
        figures = [("value_summary", plot_value_summary(result, show=show))]
        fig_dist, fig_policy = plot_distribution_and_policy(
            result,
            df,
            distribution="kernel",
            kernel_sigma=1.0,
            kernel_bin_width=0.1,
            show_empirical_policy=show_empirical_policy,
            show=show,
        )
        figures.extend(
            [
                ("ground_distribution", fig_dist),
                ("ground_policy", fig_policy),
            ]
        )

        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            for suffix, fig in figures:
                fig.savefig(
                    save_path / f"{result.noise}_{result.horizon}_{suffix}.svg",
                    bbox_inches="tight",
                )
            if verbose:
                print(f"Saved figures to: {save_path}")

    return result, df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run unified noisy-observer models.")
    parser.add_argument("--noise", "--noisy", default="mixed", help="representation, execution, or mixed")
    parser.add_argument("--horizon", "--horzion", default="infinite", help="finite or infinite")
    parser.add_argument("--n-trials", type=int, default=5 * 10**4)
    parser.add_argument("--max-timestep", type=int, default=10)
    parser.add_argument("--deadline", type=int, default=11)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sigma-repr", type=float, default=0.25)
    parser.add_argument("--tau-exec", type=float, default=0.025)
    parser.add_argument("--cost", type=float, default=0.03)
    parser.add_argument("--n-l", type=int, default=2001)
    parser.add_argument("--n-gh", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--show-empirical-policy", action="store_true")
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = ModelConfig(
        c=args.cost,
        sigma_repr=args.sigma_repr,
        tau_exec=args.tau_exec,
        n_trials=args.n_trials,
        max_timestep=args.max_timestep,
        deadline=args.deadline,
        seed=args.seed,
        n_l=args.n_l,
        n_gh=args.n_gh,
        tol=args.tol,
        max_iter=args.max_iter,
    )
    run_model(
        noise=args.noise,
        horizon=args.horizon,
        config=config,
        plot=not args.skip_plots,
        show=not args.no_show,
        save_dir=args.save_dir,
        show_empirical_policy=args.show_empirical_policy,
    )


if __name__ == "__main__":
    main()
