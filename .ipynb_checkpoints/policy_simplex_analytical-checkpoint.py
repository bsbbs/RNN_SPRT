#!/usr/bin/env python3
"""
Analytical simplex + ground-truth-belief visualizations of the three-action
policy  (p_A = choose H0, p_B = choose H1, p_S = sample).

Everything here is analytical: empirical figures (Points 1-2) read the Bellman
solution from ``noise_model_runner.compute_model`` and evaluate the closed-form
ground-view policy across time steps; theoretical figures (Points 3-4) drop the
Bellman solve entirely and impose a fixed decision mid-line.

Coloring conventions (shared)
-----------------------------
* Simplex curves: colored by ground-truth belief b with a Red-White-Blue
  colormap broken at b = 0.5  (b=0 -> red, b=0.5 -> white, b=1 -> blue).
* Belief-space curves: three action families
      p(choose H0) -> reds,  p(choose H1) -> blues,  p(sample) -> greens
  graded dark -> light along whatever the "sequence" variable is
  (time step for Points 1-2; noise level for Points 3-4).
* Overlay order: the earliest item in the sequence is drawn LAST (on top).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(tmp_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(tmp_root / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon
from scipy.stats import norm

from noise_model_runner import ModelConfig, compute_model, logistic_log10

# ----------------------------------------------------------------------------
# Global knobs
# ----------------------------------------------------------------------------
TRIANGLE_H = np.sqrt(3.0) / 2.0
MAX_T = 10
DEADLINE = 11
COST = 0.03
TAU_EXEC = 0.025
SIGMA_REPR = 0.25
MAX_STEP_EVIDENCE = 0.8          # max |log10 LR| per cue -> reachable |L| <= 0.8 t
N_CURVE = 401
EPS = 1e-9

B_CMAP = "RdBu"                  # b=0 red, b=1 blue
B_NORM = TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)

# dark -> light endpoints for the three action families
_FAMILY = {
    "H0": ((0.50, 0.00, 0.00), (1.00, 0.72, 0.72)),   # red
    "H1": ((0.00, 0.00, 0.50), (0.72, 0.78, 1.00)),   # blue
    "S":  ((0.00, 0.35, 0.00), (0.62, 0.90, 0.62)),   # green
}


def family_color(key: str, frac: float) -> tuple[float, float, float]:
    """frac in [0,1]: 0 -> dark, 1 -> light."""
    dark, light = _FAMILY[key]
    frac = float(np.clip(frac, 0.0, 1.0))
    return tuple(d + frac * (l - d) for d, l in zip(dark, light))


# ----------------------------------------------------------------------------
# Simplex geometry
# ----------------------------------------------------------------------------
def simplex_xy(p_a: np.ndarray, p_b: np.ndarray, p_s: np.ndarray) -> np.ndarray:
    """(p_A left, p_B right, p_S top) -> 2D coordinates."""
    x = p_b + 0.5 * p_s
    y = TRIANGLE_H * p_s
    return np.stack([x, y], axis=-1)


def draw_simplex(ax) -> None:
    ax.add_patch(
        Polygon([[0, 0], [0.5, TRIANGLE_H], [1, 0]], closed=True,
                facecolor="1", edgecolor="none", zorder=-2)
    )
    ax.plot([0, 0.5, 1, 0], [0, TRIANGLE_H, 0, 0], color="black", lw=1.0, zorder=4)
    ax.text(0.5, TRIANGLE_H + 0.04, r"$p_S$", ha="center", va="bottom")
    ax.text(-0.02, -0.05, r"$p_A$", ha="left", va="top")
    ax.text(1.02, -0.05, r"$p_B$", ha="right", va="top")
    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(-0.12, TRIANGLE_H + 0.12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def simplex_curve_by_b(ax, p_a, p_b, p_s, b, *, lw=2.5, zorder=3, alpha=1.0):
    """Draw a policy curve on the simplex, colored by belief b (RdBu, mid 0.5)."""
    xy = simplex_xy(np.asarray(p_a), np.asarray(p_b), np.asarray(p_s))
    segs = np.stack([xy[:-1], xy[1:]], axis=1)
    seg_b = 0.5 * (np.asarray(b)[:-1] + np.asarray(b)[1:])
    lc = LineCollection(segs, cmap=plt.get_cmap(B_CMAP), norm=B_NORM,
                        linewidth=lw, alpha=alpha, zorder=zorder)
    lc.set_array(seg_b)
    ax.add_collection(lc)
    return lc


def simplex_point(ax, p_a, p_b, p_s, color, *, zorder=6, size=70):
    """Filled marker with white halo for an extremal prediction (b=0 or b=1)."""
    xy = simplex_xy(np.array([p_a]), np.array([p_b]), np.array([p_s]))[0]
    ax.scatter([xy[0]], [xy[1]], s=size + 55, facecolors="white",
               edgecolors="white", zorder=zorder, linewidths=0)
    ax.scatter([xy[0]], [xy[1]], s=size, facecolors=[color],
               edgecolors="white", linewidths=1.4, zorder=zorder + 1)


# ----------------------------------------------------------------------------
# Closed-form policies
# ----------------------------------------------------------------------------
def softmax3(q0, q1, qw, tau):
    """Softmax policy over (choose H0, choose H1, sample). tau -> inf gives 1/3 each."""
    if not np.isfinite(tau) or tau == np.inf:
        shape = np.broadcast(q0, q1, qw).shape
        third = np.full(shape, 1.0 / 3.0)
        return third, third.copy(), third.copy()
    stack = np.vstack([np.broadcast_to(q0, np.shape(qw)).astype(float),
                       np.broadcast_to(q1, np.shape(qw)).astype(float),
                       np.asarray(qw, dtype=float)]) / tau
    stack -= stack.max(axis=0, keepdims=True)
    ex = np.exp(stack)
    ex /= ex.sum(axis=0, keepdims=True)
    return ex[0], ex[1], ex[2]


def rep_ground_view(L, lower_l, upper_l, sigma_eff, gain=1.0):
    """Hard internal boundary marginalized over L_tilde | L ~ N(gain*L, sigma_eff^2)."""
    if sigma_eff == np.inf:
        # non-extreme evidence collapses to p_A = p_B = 0.5, p_S = 0
        pa = np.full_like(np.asarray(L, dtype=float), 0.5)
        pb = np.full_like(pa, 0.5)
        ps = np.zeros_like(pa)
        return pa, pb, ps
    if sigma_eff <= 0.0:
        m = gain * np.asarray(L, dtype=float)
        pa = (m <= lower_l).astype(float)
        pb = (m >= upper_l).astype(float)
        return pa, pb, 1.0 - pa - pb
    m = gain * np.asarray(L, dtype=float)
    pa = norm.cdf((lower_l - m) / sigma_eff)
    pb = norm.cdf((m - upper_l) / sigma_eff)
    ps = norm.cdf((upper_l - m) / sigma_eff) - norm.cdf((lower_l - m) / sigma_eff)
    return pa, pb, ps


def reachable_L(t: int, n: int = N_CURVE) -> np.ndarray:
    Lmax = MAX_STEP_EVIDENCE * t
    return np.linspace(-Lmax, Lmax, n)


# ----------------------------------------------------------------------------
# Empirical policy extraction (Points 1-2), per condition + time step
# ----------------------------------------------------------------------------
def exec_policy_at_t(result, t: int):
    """Execution-only ground view at step t over the reachable L range."""
    L = reachable_L(t)
    if result.horizon == "finite":
        p0, p1, ps = result.p_choose0[t], result.p_choose1[t], result.p_sample[t]
    else:
        p0, p1, ps = result.p_choose0, result.p_choose1, result.p_sample
    pa = np.interp(L, result.l_grid, p0)
    pb = np.interp(L, result.l_grid, p1)
    psamp = np.interp(L, result.l_grid, ps)
    b = logistic_log10(L)
    return L, b, pa, pb, psamp


def rep_policy_at_t(result, t: int):
    """Representation-only (perfect integration) ground view at step t."""
    L = reachable_L(t)
    if result.horizon == "finite":
        lo, hi = float(result.lower_l[t]), float(result.upper_l[t])
    else:
        lo, hi = float(result.lower_l), float(result.upper_l)
    s_t = SIGMA_REPR * np.sqrt(t)
    pa, pb, ps = rep_ground_view(L, lo, hi, s_t)
    b = logistic_log10(L)
    return L, b, pa, pb, ps


# ----------------------------------------------------------------------------
# Figure: 2x2 empirical (Points 1 and 2)
# ----------------------------------------------------------------------------
def empirical_2x2(kind: str, out: Path):
    """kind in {'execution','representation'}."""
    if kind == "execution":
        res_unc = compute_model("execution", "infinite",
                                 ModelConfig(c=COST, tau_exec=TAU_EXEC, sigma_repr=0.0),
                                 verbose=False)
        res_con = compute_model("execution", "finite",
                                ModelConfig(c=COST, tau_exec=TAU_EXEC, sigma_repr=0.0,
                                            deadline=DEADLINE, max_timestep=MAX_T),
                                verbose=False)
        policy_at_t = exec_policy_at_t
        title = rf"Execution noise only ($\tau={TAU_EXEC:g}$)"
    else:
        res_unc = compute_model("representation", "infinite",
                                ModelConfig(c=COST, sigma_repr=SIGMA_REPR, tau_exec=0.0),
                                verbose=False)
        res_con = compute_model("representation", "finite",
                                ModelConfig(c=COST, sigma_repr=SIGMA_REPR, tau_exec=0.0,
                                            deadline=DEADLINE, max_timestep=MAX_T),
                                verbose=False)
        policy_at_t = rep_policy_at_t
        title = rf"Representation noise only ($\sigma_r={SIGMA_REPR:g}$, perfect integration)"

    rows = [("Time unconstrained", res_unc), ("Time constrained", res_con)]
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 7.4), constrained_layout=True)

    Lrange_info = {}
    for r, (row_label, res) in enumerate(rows):
        ax_simplex, ax_curve = axes[r]
        draw_simplex(ax_simplex)
        lc = None
        # earliest time step on top: draw t = MAX_T .. 1
        for t in range(MAX_T, 0, -1):
            L, b, pa, pb, ps = policy_at_t(res, t)
            z = 3 + (MAX_T - t)               # earlier t -> higher zorder
            lc = simplex_curve_by_b(ax_simplex, pa, pb, ps, b,
                                    lw=2.4, zorder=z, alpha=0.95)
            frac = (t - 1) / (MAX_T - 1)      # t=1 dark, t=MAX_T light
            ax_curve.plot(b, pa, color=family_color("H0", frac), lw=1.6, zorder=z)
            ax_curve.plot(b, pb, color=family_color("H1", frac), lw=1.6, zorder=z)
            ax_curve.plot(b, ps, color=family_color("S", frac), lw=1.6, zorder=z)
        Lrange_info[row_label] = (-MAX_STEP_EVIDENCE * MAX_T, MAX_STEP_EVIDENCE * MAX_T)

        ax_curve.set_xlim(0, 1)
        ax_curve.set_ylim(-0.03, 1.03)
        ax_curve.set_xlabel(r"Ground-truth belief $b=P(H_1\mid \mathrm{evidence})$")
        ax_curve.set_ylabel("Action probability")
        for sp in ("top", "right"):
            ax_curve.spines[sp].set_visible(False)
        ax_simplex.set_ylabel(row_label, fontsize=11)

    # shared belief colorbar on the far left (avoids overlapping curve panels)
    cb = fig.colorbar(lc, ax=axes[:, 0], location="left", fraction=0.05,
                      pad=0.02, aspect=34, shrink=0.85)
    cb.set_label(r"belief $b$")

    # figure-level legend for the three action families + grading note
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=family_color("H0", 0.0), lw=2.4, label=r"$p(\mathrm{choose}\ H_0)$"),
        Line2D([0], [0], color=family_color("H1", 0.0), lw=2.4, label=r"$p(\mathrm{choose}\ H_1)$"),
        Line2D([0], [0], color=family_color("S", 0.0), lw=2.4, label=r"$p(\mathrm{sample})$"),
    ]
    fig.legend(handles=handles, ncol=3, fontsize=8, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, 0.015, "(dark = t1, light = t10)", ha="center",
             fontsize=7.5, color="0.4")

    fig.suptitle(title, fontsize=12)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return Lrange_info


# ----------------------------------------------------------------------------
# Figure: theoretical execution noise (Point 3), 1x2
# ----------------------------------------------------------------------------
def theoretical_Q(beta_lo=0.15, beta_hi=0.85, peak=0.90):
    """Fixed, reasonable decision mid-line via a concave wait value.
    Q0=1-b, Q1=b, Qw concave with crossings at beta_lo/beta_hi."""
    # symmetric: peak - B*(b-0.5)^2 crosses Q1=b at beta_hi
    B = (peak - beta_hi) / (beta_hi - 0.5) ** 2

    def Qfun(b):
        q0 = 1.0 - b
        q1 = b
        qw = peak - B * (b - 0.5) ** 2
        return q0, q1, qw

    return Qfun


def theoretical_execution(out: Path, tau_levels=(0.05, 0.10, 0.20, 0.50)):
    Qfun = theoretical_Q()
    b = np.linspace(EPS, 1 - EPS, N_CURVE)
    q0, q1, qw = Qfun(b)
    levels = list(tau_levels) + [np.inf]
    n_lev = len(levels)

    fig, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(7.4, 3.7))
    draw_simplex(ax_s)
    lc = None
    for j, tau in enumerate(reversed(levels)):           # earliest (smallest) on top
        idx = n_lev - 1 - j
        pa, pb, ps = softmax3(q0, q1, qw, tau)
        z = 3 + j
        lc = simplex_curve_by_b(ax_s, pa, pb, ps, b, lw=2.6, zorder=z,
                                alpha=0.6 if not np.isfinite(tau) else 0.95)
        frac = idx / (n_lev - 1)                          # level 0 dark, last light
        ax_c.plot(b, pa, color=family_color("H0", frac), lw=1.8, zorder=z)
        ax_c.plot(b, pb, color=family_color("H1", frac), lw=1.8, zorder=z)
        ax_c.plot(b, ps, color=family_color("S", frac), lw=1.8, zorder=z)
        # extremal predictions b=0, b=1
        pa0, pb0, ps0 = softmax3(*Qfun(np.array([0.0])), tau)
        pa1, pb1, ps1 = softmax3(*Qfun(np.array([1.0])), tau)
        simplex_point(ax_s, pa0[0], pb0[0], ps0[0], family_color("H0", frac))
        simplex_point(ax_s, pa1[0], pb1[0], ps1[0], family_color("H1", frac))

    cb = fig.colorbar(lc, ax=ax_s, fraction=0.05, pad=0.04, aspect=26)
    cb.set_label(r"belief $b$")
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(-0.03, 1.03)
    ax_c.set_xlabel(r"Ground-truth belief $b$")
    ax_c.set_ylabel("Action probability")
    for sp in ("top", "right"):
        ax_c.spines[sp].set_visible(False)
    ax_c.text(0.99, 0.98, "dark = low noise\nlight = high noise",
              fontsize=7, ha="right", va="top", transform=ax_c.transAxes, color="0.4")
    fig.suptitle(r"Theoretical execution noise (fixed mid-line); $\tau\to\infty$ collapses to centroid",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure: theoretical representation noise (Point 4), 1x2, two versions
# ----------------------------------------------------------------------------
def _rep_corner_points(ax_s, color, lower_l, upper_l, gain):
    """Extremal evidence L = -inf -> p_A=1 ; L=+inf -> p_B=1 (corners)."""
    simplex_point(ax_s, 1.0, 0.0, 0.0, color)   # b=0
    simplex_point(ax_s, 0.0, 1.0, 0.0, color)   # b=1


def theoretical_representation(out: Path, mode: str,
                               beta_lo=0.15, beta_hi=0.85):
    """mode='sigma'  : perfect integration, vary inference-noise sigma.
       mode='kappa'  : no inference noise, vary forgetting kappa<0.5
                       via leaky accumulator L_tilde_t = 2*kappa*L_tilde_{t-1}+ell_t.
    """
    lower_l = np.log10(beta_lo / (1 - beta_lo))
    upper_l = np.log10(beta_hi / (1 - beta_hi))

    # finite-L grid for curves; corners handled separately as b=0,1
    L = np.linspace(-6.0, 6.0, N_CURVE)
    b = logistic_log10(L)

    if mode == "sigma":
        levels = [0.25, 0.5, 1.0, 2.0, np.inf]
        label = r"inference noise $\sigma$"
        gains = [1.0] * len(levels)
        sigmas = levels
        sub = r"perfect integration, varying inference noise $\sigma$"
    else:
        # Leaky log-odds accumulator (no inference noise):
        #     L_tilde_t = 2*kappa * L_tilde_{t-1} + ell_t,   r = 2*kappa in (0,1]
        # kappa = 0.5 -> r = 1 -> exact Bayesian summation (hard boundary).
        # kappa < 0.5 -> recency: for a fixed ground-truth cumulative L the
        # internal estimate spreads, softening the ground-view policy.
        # We summarize that order-dependent spread by an EFFECTIVE std
        # sigma_eff(kappa) (gain held at 1 so extreme evidence still reaches
        # the corners), with sigma_eff -> 0 at kappa=0.5 and -> inf at kappa->0.
        kappas = [0.5, 0.4, 0.3, 0.2]
        t_ref = MAX_T
        ev = compute_model("representation", "infinite",
                           ModelConfig(c=COST, sigma_repr=0.0, tau_exec=0.0),
                           verbose=False).stimuli
        v_ell = float(np.mean(ev.evidence_log10 ** 2))     # per-cue log-odds variance
        gains, sigmas = [], []
        for kap in kappas:
            r = 2.0 * kap
            w = r ** np.arange(t_ref)[::-1]                 # leaky weights on ell_1..ell_t
            sigma_eff = np.sqrt(np.sum((w - 1.0) ** 2) * v_ell)
            gains.append(1.0)
            sigmas.append(float(sigma_eff))
        # kappa -> 0 limit behaves like infinite representation noise
        kappas = kappas + [0.0]
        sigmas = sigmas + [np.inf]
        gains = gains + [1.0]
        levels = kappas
        label = r"forgetting"
        sub = r"no inference noise, varying forgetting $\kappa\leq 0.5$ (leaky accumulator)"

    n_lev = len(levels)
    fig, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(7.4, 3.7))
    draw_simplex(ax_s)
    lc = None
    for j in range(n_lev - 1, -1, -1):                       # earliest (idx0) on top
        sig = sigmas[j]
        gain = gains[j]
        z = 3 + (n_lev - 1 - j)
        pa, pb, ps = rep_ground_view(L, lower_l, upper_l, sig, gain=gain)
        lc = simplex_curve_by_b(ax_s, pa, pb, ps, b, lw=2.6, zorder=z,
                                alpha=0.6 if sig == np.inf else 0.95)
        frac = j / (n_lev - 1)
        ax_c.plot(b, pa, color=family_color("H0", frac), lw=1.8, zorder=z)
        ax_c.plot(b, pb, color=family_color("H1", frac), lw=1.8, zorder=z)
        ax_c.plot(b, ps, color=family_color("S", frac), lw=1.8, zorder=z)

    # extremal corners: evidence beyond the noise -> exact b=0 / b=1
    simplex_point(ax_s, 1.0, 0.0, 0.0, (0.35, 0, 0))
    simplex_point(ax_s, 0.0, 1.0, 0.0, (0, 0, 0.35))
    ax_c.scatter([0, 1], [1, 1], s=55, facecolors=[(0.35, 0, 0), (0, 0, 0.35)],
                 edgecolors="white", linewidths=1.2, zorder=8)

    cb = fig.colorbar(lc, ax=ax_s, fraction=0.05, pad=0.04, aspect=26)
    cb.set_label(r"belief $b$")
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(-0.03, 1.03)
    ax_c.set_xlabel(r"Ground-truth belief $b$")
    ax_c.set_ylabel("Action probability")
    for sp in ("top", "right"):
        ax_c.spines[sp].set_visible(False)
    ax_c.text(0.99, 0.98, f"dark = low {label}\nlight = high {label}\n(filled dots: b=0,1 extremes)",
              fontsize=6.5, ha="right", va="top", transform=ax_c.transAxes, color="0.4")
    fig.suptitle("Theoretical representation noise: " + sub, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
def main(outdir: str = "/home/claude/figs"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    info = {}
    info["exec_2x2"] = empirical_2x2("execution", out / "point1_execution_2x2.png")
    info["rep_2x2"] = empirical_2x2("representation", out / "point2_representation_2x2.png")
    theoretical_execution(out / "point3_execution_theoretical.png")
    theoretical_representation(out / "point4a_representation_sigma.png", mode="sigma")
    theoretical_representation(out / "point4b_representation_kappa.png", mode="kappa")
    print("done; reachable L ranges:", info)


if __name__ == "__main__":
    main()
