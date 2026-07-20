#!/usr/bin/env python3
"""
Analytical simplex + value + ground-truth-belief visualizations of the
three-action policy (p0 = choose H0, p1 = choose H1, pS = sample).

Each figure now has THREE columns per condition block:
  (1) simplex   (2) value landscape V0/V1/Vw   (3) stacked policy:
      top = p(choose H0)+p(choose H1),  bottom = p(sample)

Coloring
  * simplex curves: colored by ground-truth belief b (RdBu, broken at 0.5)
  * policy/value families: red=H0, blue=H1, green=sample/wait,
    graded dark->light along the sequence variable (time, or noise level)
  * earliest sequence item drawn on top; extrema drawn frontmost.
"""

from __future__ import annotations

import os
from pathlib import Path

tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(tmp_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(tmp_root / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.stats import norm

from noise_model_runner import ModelConfig, compute_model, logistic_log10

# ----------------------------------------------------------------------------
TRIANGLE_H = np.sqrt(3.0) / 2.0
MAX_T = 10
DEADLINE = 11
COST = 0.03
TAU_EXEC_EMP = 0.1          # Point 1: larger tau so simplex curves bow off edge
SIGMA_REPR = 0.25
MAX_STEP_EVIDENCE = 0.8     # reachable |L| <= 0.8 t
N_CURVE = 401
EPS = 1e-9

B_CMAP = "RdBu"
B_NORM = TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)
GRAY_INF = (0.35, 0.35, 0.35)

# theoretical noise levels
TAU_LEVELS = [0.03, 0.09, 0.25, 0.7]          # Point 3 (first two pushed lower)
SIGMA_LEVELS = [0.25, 0.6, 1.2, 2.5]          # Point 4a
KAPPA_LEVELS = [0.95, 0.80, 0.62, 0.45]       # Point 4b retention (Prop 3)

# fixed theoretical decision boundary (belief)
BETA_LO, BETA_HI, QW_PEAK = 0.15, 0.85, 0.90

_FAMILY = {
    "H0": ((0.50, 0.00, 0.00), (1.00, 0.72, 0.72)),
    "H1": ((0.00, 0.00, 0.50), (0.72, 0.78, 1.00)),
    "S":  ((0.00, 0.35, 0.00), (0.62, 0.90, 0.62)),
}


def family_color(key, frac):
    dark, light = _FAMILY[key]
    frac = float(np.clip(frac, 0.0, 1.0))
    return tuple(d + frac * (l - d) for d, l in zip(dark, light))


# ----------------------------------------------------------------------------
# simplex geometry  (p0 left, p1 right, pS top)
# ----------------------------------------------------------------------------
def simplex_xy(p0, p1, ps):
    x = p1 + 0.5 * ps
    y = TRIANGLE_H * ps
    return np.stack([x, y], axis=-1)


def draw_simplex(ax):
    ax.add_patch(Polygon([[0, 0], [0.5, TRIANGLE_H], [1, 0]], closed=True,
                         facecolor="1", edgecolor="none", zorder=-2))
    ax.plot([0, 0.5, 1, 0], [0, TRIANGLE_H, 0, 0], color="black", lw=1.0, zorder=4)
    ax.text(0.5, TRIANGLE_H + 0.04, r"$p_S$", ha="center", va="bottom")
    ax.text(-0.02, -0.05, r"$p_0$", ha="left", va="top")
    ax.text(1.02, -0.05, r"$p_1$", ha="right", va="top")
    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(-0.12, TRIANGLE_H + 0.12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def simplex_curve_by_b(ax, p0, p1, ps, b, *, lw=2.4, zorder=3, alpha=1.0):
    xy = simplex_xy(np.asarray(p0), np.asarray(p1), np.asarray(ps))
    segs = np.stack([xy[:-1], xy[1:]], axis=1)
    seg_b = 0.5 * (np.asarray(b)[:-1] + np.asarray(b)[1:])
    lc = LineCollection(segs, cmap=plt.get_cmap(B_CMAP), norm=B_NORM,
                        linewidth=lw, alpha=alpha, zorder=zorder)
    lc.set_array(seg_b)
    ax.add_collection(lc)
    return lc


def simplex_point(ax, p0, p1, ps, color, *, zorder=20, size=64):
    xy = simplex_xy(np.array([p0]), np.array([p1]), np.array([ps]))[0]
    ax.scatter([xy[0]], [xy[1]], s=size + 60, facecolors="white",
               edgecolors="white", zorder=zorder, linewidths=0)
    ax.scatter([xy[0]], [xy[1]], s=size, facecolors=[color],
               edgecolors="white", linewidths=1.3, zorder=zorder + 1)


# ----------------------------------------------------------------------------
# closed-form policies / values
# ----------------------------------------------------------------------------
def softmax3(q0, q1, qw, tau):
    if not np.isfinite(tau):
        shp = np.shape(qw)
        third = np.full(shp, 1 / 3)
        return third, third.copy(), third.copy()
    s = np.vstack([np.broadcast_to(q0, np.shape(qw)).astype(float),
                   np.broadcast_to(q1, np.shape(qw)).astype(float),
                   np.asarray(qw, float)]) / tau
    s -= s.max(0, keepdims=True)
    e = np.exp(s); e /= e.sum(0, keepdims=True)
    return e[0], e[1], e[2]


def rep_ground_view(L, lo, hi, m_of_L, s_t):
    """Hard internal boundary marginalized over L_tilde|L ~ N(m_of_L(L), s_t^2)."""
    if not np.isfinite(s_t):                       # infinite spread
        m = m_of_L(L)
        pa = np.full_like(np.asarray(L, float), 0.5)
        return pa, pa.copy(), np.zeros_like(pa)
    m = m_of_L(L)
    if s_t <= 0:
        pa = (m <= lo).astype(float); pb = (m >= hi).astype(float)
        return pa, pb, 1 - pa - pb
    pa = norm.cdf((lo - m) / s_t)
    pb = norm.cdf((m - hi) / s_t)
    ps = norm.cdf((hi - m) / s_t) - norm.cdf((lo - m) / s_t)
    return pa, pb, ps


def concave_qwait(b, beta_lo=BETA_LO, beta_hi=BETA_HI, peak=QW_PEAK):
    B = (peak - beta_hi) / (beta_hi - 0.5) ** 2
    return peak - B * (b - 0.5) ** 2


def reachable_L(t, n=N_CURVE):
    Lmax = MAX_STEP_EVIDENCE * t
    return np.linspace(-Lmax, Lmax, n)


# ----------------------------------------------------------------------------
# generic 3-panel block
# ----------------------------------------------------------------------------
def make_block_axes(fig, gs, r0):
    """Return (ax_simplex, ax_value, ax_choose, ax_sample) for a condition row.
    gs is a GridSpec with 2 sub-rows per block and 3 columns."""
    ax_s = fig.add_subplot(gs[r0:r0 + 2, 0])
    ax_v = fig.add_subplot(gs[r0:r0 + 2, 1])
    ax_c = fig.add_subplot(gs[r0, 2])
    ax_p = fig.add_subplot(gs[r0 + 1, 2], sharex=ax_c)
    return ax_s, ax_v, ax_c, ax_p


def style_policy_axes(ax_c, ax_p, betas):
    for ax in (ax_c, ax_p):
        for blo in betas:
            ax.axvline(blo, color="0.75", ls="--", lw=1.0, zorder=0)
        ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.03)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax_c.set_ylabel("Choice prob.")
    ax_p.set_ylabel("Sample prob.")
    ax_p.set_xlabel(r"Ground-truth belief $b$")
    plt.setp(ax_c.get_xticklabels(), visible=False)


def style_value_axes(ax_v, betas):
    for blo in betas:
        ax_v.axvline(blo, color="0.75", ls="--", lw=1.0, zorder=0)
    ax_v.set_xlim(0, 1)
    ax_v.set_xlabel(r"Ground-truth belief $b$")
    ax_v.set_ylabel("Value")
    for sp in ("top", "right"):
        ax_v.spines[sp].set_visible(False)


def draw_value(ax_v, b, q0, q1, qw_layers, fracs, betas):
    """qw_layers: list of wait-value arrays; fracs: green shade per layer.
    q0,q1 drawn once (dark X). """
    style_value_axes(ax_v, betas)
    ax_v.plot(b, q0, color=family_color("H0", 0.0), lw=2.0, zorder=5)
    ax_v.plot(b, q1, color=family_color("H1", 0.0), lw=2.0, zorder=5)
    n = len(qw_layers)
    for i, qw in enumerate(qw_layers):
        fr = fracs[i] if fracs is not None else 0.0
        z = 3 + (n - 1 - i)
        ax_v.plot(b, qw, color=family_color("S", fr), lw=1.8, zorder=z)


# ----------------------------------------------------------------------------
# EMPIRICAL 2x2 (Points 1 and 2)
# ----------------------------------------------------------------------------
def empirical_figure(kind, out):
    if kind == "execution":
        res_unc = compute_model("execution", "infinite",
                                ModelConfig(c=COST, tau_exec=TAU_EXEC_EMP, sigma_repr=0.0),
                                verbose=False)
        res_con = compute_model("execution", "finite",
                                ModelConfig(c=COST, tau_exec=TAU_EXEC_EMP, sigma_repr=0.0,
                                            deadline=DEADLINE, max_timestep=MAX_T),
                                verbose=False)
        title = rf"Execution noise only ($\tau={TAU_EXEC_EMP:g}$)"
    else:
        res_unc = compute_model("representation", "infinite",
                                ModelConfig(c=COST, sigma_repr=SIGMA_REPR, tau_exec=0.0),
                                verbose=False)
        res_con = compute_model("representation", "finite",
                                ModelConfig(c=COST, sigma_repr=SIGMA_REPR, tau_exec=0.0,
                                            deadline=DEADLINE, max_timestep=MAX_T),
                                verbose=False)
        title = rf"Representation noise only ($\sigma_r={SIGMA_REPR:g}$, perfect integration)"

    def policy_at_t(res, t):
        L = reachable_L(t)
        if kind == "execution":
            if res.horizon == "finite":
                p0, p1, ps = res.p_choose0[t], res.p_choose1[t], res.p_sample[t]
            else:
                p0, p1, ps = res.p_choose0, res.p_choose1, res.p_sample
            pa = np.interp(L, res.l_grid, p0)
            pb = np.interp(L, res.l_grid, p1)
            psamp = np.interp(L, res.l_grid, ps)
        else:
            if res.horizon == "finite":
                lo, hi = float(res.lower_l[t]), float(res.upper_l[t])
            else:
                lo, hi = float(res.lower_l), float(res.upper_l)
            s_t = SIGMA_REPR * np.sqrt(t)
            pa, pb, psamp = rep_ground_view(L, lo, hi, lambda x: x, s_t)
        return L, logistic_log10(L), pa, pb, psamp

    fig = plt.figure(figsize=(11.5, 7.6))
    gs = GridSpec(4, 3, figure=fig, width_ratios=[1.05, 1.0, 1.15],
                  height_ratios=[1, 1, 1, 1], hspace=0.35, wspace=0.38)

    rows = [("Time unconstrained", res_unc, 0), ("Time constrained", res_con, 2)]
    top_simplex_lc = None
    for label, res, r0 in rows:
        ax_s, ax_v, ax_c, ax_p = make_block_axes(fig, gs, r0)
        draw_simplex(ax_s)
        ax_s.set_title(label, fontsize=11)
        b_grid = res.b_grid

        # value panel
        q0 = 1 - b_grid; q1 = b_grid
        if res.horizon == "finite":
            qw_layers = [np.interp(b_grid, res.b_grid, res.q_wait[t]) for t in range(1, MAX_T + 1)]
            fracs = [(t - 1) / (MAX_T - 1) for t in range(1, MAX_T + 1)]
            qw_layers = qw_layers[::-1]; fracs = fracs[::-1]   # earliest on top
            betas = [float(res.lower_b[1]), float(res.upper_b[1])]
        else:
            qw_layers = [res.q_wait]
            fracs = [0.0]
            betas = [float(res.lower_b), float(res.upper_b)]
        draw_value(ax_v, b_grid, q0, q1, qw_layers, fracs, betas)

        # policy panels + simplex
        style_policy_axes(ax_c, ax_p, betas)
        for t in range(MAX_T, 0, -1):
            L, b, p0, p1, ps = policy_at_t(res, t)
            z = 3 + (MAX_T - t)
            lc = simplex_curve_by_b(ax_s, p0, p1, ps, b, lw=2.2, zorder=z)
            if r0 == 0:
                top_simplex_lc = lc
            fr = (t - 1) / (MAX_T - 1)
            ax_c.plot(b, p0, color=family_color("H0", fr), lw=1.5, zorder=z)
            ax_c.plot(b, p1, color=family_color("H1", fr), lw=1.5, zorder=z)
            ax_p.plot(b, ps, color=family_color("S", fr), lw=1.5, zorder=z)
        # extrema frontmost: policy at the largest reachable |L| (t=MAX_T edges)
        L, b, p0, p1, ps = policy_at_t(res, MAX_T)
        simplex_point(ax_s, p0[0], p1[0], ps[0], family_color("H0", 0.0))
        simplex_point(ax_s, p0[-1], p1[-1], ps[-1], family_color("H1", 0.0))

    # belief colorbar attached to TOP-LEFT simplex only
    top_ax = fig.axes[0]
    cb = fig.colorbar(top_simplex_lc, ax=top_ax, location="left",
                      fraction=0.05, shrink=0.62, pad=0.10, aspect=18)
    cb.set_label("Ground-truth evidence (b)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # time-step legend (dark->light = t1->t10)
    sm = ScalarMappable(cmap="Greys", norm=plt.Normalize(1, MAX_T)); sm.set_array([])
    cb2 = fig.colorbar(sm, ax=fig.axes[3], fraction=0.045, pad=0.06, aspect=18)
    cb2.set_label("time step (dark=1, light=10)", fontsize=7)
    cb2.ax.tick_params(labelsize=7)

    handles = [Line2D([0], [0], color=family_color("H0", 0.0), lw=2.2, label=r"$p(\mathrm{choose}\ H_0)$"),
               Line2D([0], [0], color=family_color("H1", 0.0), lw=2.2, label=r"$p(\mathrm{choose}\ H_1)$"),
               Line2D([0], [0], color=family_color("S", 0.0), lw=2.2, label=r"$p(\mathrm{sample})$")]
    fig.legend(handles=handles, ncol=3, fontsize=8, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=13)
    fig.savefig(out, dpi=125, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# THEORETICAL single-block figures (Points 3, 4a, 4b)
# ----------------------------------------------------------------------------
def theoretical_figure(out, *, mode, levels, level_label, level_fmt, sub, add_inf=True):
    """mode in {'exec','rep_sigma','rep_kappa'}.
    levels: list of finite levels; an infinite/collapse level is appended if add_inf.
    """
    lo_l = np.log10(BETA_LO / (1 - BETA_LO))
    hi_l = np.log10(BETA_HI / (1 - BETA_HI))

    b = np.linspace(EPS, 1 - EPS, N_CURVE)
    if mode == "exec":
        L = None
    else:
        L = np.linspace(-6.0, 6.0, N_CURVE)
        b = logistic_log10(L)

    all_levels = list(levels) + ([np.inf] if add_inf else [])
    n = len(all_levels)

    fig = plt.figure(figsize=(11.2, 4.2))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.05, 1.0, 1.15],
                  height_ratios=[1, 1], hspace=0.12, wspace=0.40)
    ax_s = fig.add_subplot(gs[:, 0]); draw_simplex(ax_s)
    ax_v = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_p = fig.add_subplot(gs[1, 2], sharex=ax_c)

    betas = [BETA_LO, BETA_HI]
    style_policy_axes(ax_c, ax_p, betas)

    # value landscape (noise-invariant): single X + single concave wait
    q0 = 1 - b; q1 = b; qw = concave_qwait(b)
    draw_value(ax_v, b, q0, q1, [qw], [0.0], betas)

    legend_handles = []
    lc = None
    for j in range(n - 1, -1, -1):                 # earliest (idx0) drawn last/top
        lev = all_levels[j]
        z = 3 + (n - 1 - j)
        frac = j / (n - 1)
        is_inf = not np.isfinite(lev)

        if mode == "exec":
            p0, p1, ps = softmax3(q0, q1, qw, lev)
            xb = b
            # extrema points at b=0,1
            e0 = softmax3(*[np.array([v]) for v in (1.0, 0.0, concave_qwait(np.array([0.0]))[0])], lev)
            e1 = softmax3(*[np.array([v]) for v in (0.0, 1.0, concave_qwait(np.array([1.0]))[0])], lev)
            extrema = [(e0[0][0], e0[1][0], e0[2][0], family_color("H0", frac) if not is_inf else GRAY_INF),
                       (e1[0][0], e1[1][0], e1[2][0], family_color("H1", frac) if not is_inf else GRAY_INF)]
        elif mode == "rep_sigma":
            s_t = np.inf if is_inf else lev
            p0, p1, ps = rep_ground_view(L, lo_l, hi_l, lambda x: x, s_t)
            xb = b
            if is_inf:
                extrema = [(0.5, 0.5, 0.0, GRAY_INF)]      # belief collapses to 0.5
            else:
                extrema = [(1.0, 0.0, 0.0, family_color("H0", frac)),
                           (0.0, 1.0, 0.0, family_color("H1", frac))]
        else:  # rep_kappa  (Proposition 3 exact, constant per-step evidence ell)
            # x-axis L is the per-step ground-truth log-odds (steady evidence).
            # m_t = (1-kappa) sum_s kappa^{t-s} ell = ell (1 - kappa^t)
            # s_t^2 = sigma^2 (1 - kappa^{2t}) / (1 - kappa^2)
            t_ref = MAX_T
            sigma_fixed = 0.25
            kap = lev
            gain = 1.0 - kap ** t_ref
            s_t = sigma_fixed * np.sqrt((1 - kap ** (2 * t_ref)) / (1 - kap ** 2))
            p0, p1, ps = rep_ground_view(L, lo_l, hi_l, lambda x, g=gain: g * x, s_t)
            # extreme evidence beats the noise -> corners
            extrema = [(1.0, 0.0, 0.0, family_color("H0", frac)),
                       (0.0, 1.0, 0.0, family_color("H1", frac))]
            xb = b

        lc = simplex_curve_by_b(ax_s, p0, p1, ps, xb, lw=2.5, zorder=z,
                                alpha=0.55 if is_inf else 0.95)
        ax_c.plot(xb, p0, color=(GRAY_INF if is_inf else family_color("H0", frac)), lw=1.8, zorder=z)
        ax_c.plot(xb, p1, color=(GRAY_INF if is_inf else family_color("H1", frac)), lw=1.8, zorder=z)
        ax_p.plot(xb, ps, color=(GRAY_INF if is_inf else family_color("S", frac)), lw=1.8, zorder=z)

        lab = (r"$\infty$" if is_inf else level_fmt.format(lev))
        legend_handles.append(Line2D([0], [0],
                              color=(GRAY_INF if is_inf else family_color("S", frac)),
                              lw=2.4, label=lab))
        # stash extrema, draw after loop so they are frontmost
        if j == n - 1:
            extrema_store = []
        extrema_store.extend(extrema)

    for (p0e, p1e, pse, col) in extrema_store:
        simplex_point(ax_s, p0e, p1e, pse, col)

    # legend on simplex listing every level value
    legend_handles = legend_handles[::-1]
    ax_s.legend(handles=legend_handles, title=level_label, fontsize=7,
                title_fontsize=8, frameon=False, loc="upper left",
                bbox_to_anchor=(-0.30, 1.02))

    cb = fig.colorbar(lc, ax=ax_s, location="right", fraction=0.05,
                      shrink=0.7, pad=0.04, aspect=20)
    cb.set_label("Ground-truth evidence (b)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle(sub, fontsize=11)
    fig.savefig(out, dpi=125, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
def main(outdir="/home/claude/figs3"):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    empirical_figure("execution", out / "point1_execution.png")
    empirical_figure("representation", out / "point2_representation.png")
    theoretical_figure(out / "point3_execution.png", mode="exec",
                       levels=TAU_LEVELS, level_label=r"$\tau$",
                       level_fmt=r"$\tau={:g}$",
                       sub=r"Theoretical execution noise (fixed mid-line); $\tau\to\infty$ -> centroid")
    theoretical_figure(out / "point4a_rep_sigma.png", mode="rep_sigma",
                       levels=SIGMA_LEVELS, level_label=r"$\sigma$",
                       level_fmt=r"$\sigma={:g}$",
                       sub=r"Theoretical representation noise: perfect integration, varying inference noise $\sigma$")
    theoretical_figure(out / "point4b_rep_kappa.png", mode="rep_kappa",
                       levels=KAPPA_LEVELS, level_label=r"$\kappa$",
                       level_fmt=r"$\kappa={:g}$", add_inf=False,
                       sub=r"Theoretical representation noise: varying retention $\kappa$ (Prop. 3; per-step evidence)")
    print("done")


if __name__ == "__main__":
    main()
