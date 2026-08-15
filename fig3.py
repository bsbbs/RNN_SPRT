#!/usr/bin/env python3
"""
Figure 3 -- Representation noise makes decision boundaries *look* soft even
though the observer's internal policy stays a hard threshold.

Key message
-----------
A noisy internal representation L~ = L + xi (xi ~ N(0, s^2), s = sigma*sqrt(t))
keeps the OPTIMAL policy a hard staircase in the observer's own belief b~
(panel A), but when the experimenter refers the same behaviour to the
ground-truth belief b, marginalizing over the noise convolves the step with a
Gaussian kernel and the policy appears soft (panel B).  Panels C/D contrast the
same 50,000 trials in internal (b~) vs ground-truth (b) coordinates.  Panel E
(simplex) shows that as the representation noise grows the policy converges to
the (0.5, 0.5, 0) point on the p0-p1 edge -- qualitatively DIFFERENT from
execution noise (fig2.py), whose limit is the centroid (1/3, 1/3, 1/3).
Panel F shows p(sample) collapsing toward 0 as noise grows, and, below it, the
corresponding p(choose H0) / p(choose H1).

axis labels : fontsize 12
tick labels : fontsize 10
sampling cost c = 0.04 in every simulation.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde, norm

import noise_model_runner as nmr

# --------------------------------------------------------------------------
# Output directory
# --------------------------------------------------------------------------
fallback_outdir = str(Path.home())
try:
    outdir = os.path.dirname(os.path.abspath(__file__))
except (NameError, TypeError):
    outdir = fallback_outdir


# --------------------------------------------------------------------------
# Global style  (identical block in fig2.py)
# --------------------------------------------------------------------------
plt.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.family": ["Arial", "Sans"],
    "mathtext.fontset": "stix",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

LBL_FS = 12       # axis-label fontsize      (project convention)
TICK_FS = 10      # tick-label fontsize      (project convention)
LEG_FS = 10       # legend entry fontsize
TITLE_FS = 12     # legend-title fontsize
ANN_FS = 10       # in-panel annotations
EQ_FS = 14        # in-panel equations
PANEL_FS = 16     # panel letters

# Colour convention shared with fig2.py:
#   H0 = blue family, H1 = red family, S = green family (sample / wait);
#   inside each family the level index k runs from fully gray (k = 0, the most
#   stochastic level) to fully saturated (k = 5, the deterministic policy).
GRAY_C = (0.62, 0.62, 0.62)
DARK = {
    "S": (0.0, 0.32, 0.0),
    "H0": (0.0, 0.0, 0.55),
    "H1": (0.55, 0.0, 0.0),
}
GRAYFRAC = [1.0, 0.80, 0.60, 0.40, 0.20, 0.0]
B_CMAP = "RdBu_r"              # b=0 (H0) -> blue, b=1 (H1) -> red
B_NORM = TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)
TRI_H = np.sqrt(3.0) / 2.0
DOT_S = 44

# Pale variants used only by the trial-density columns of panels C / D.
PALE_H0 = (0.55, 0.62, 0.92)     # mean log-odds of the chosen-H0 trials
PALE_H1 = (0.92, 0.55, 0.55)     # mean log-odds of the chosen-H1 trials
PASTEL = {"H0": "blue", "H1": "red", "S": "green"}

# ---------------------------- model settings ------------------------------ #
COST = 0.04
SIGMA_REPR = 0.25          # per-step representational noise SD (in log10 odds)
NTRIAL = 50_000
MAX_TIMESTEP = 10
SEED = 1
N_GH = 121
B_REP_T = 4                # representative time step of panel B (s=sigma*sqrt(t))
BW = 0.1                   # histogram bin width of the density columns
COL_W = 0.85               # width of one density column, in time-step units
YMIN, YMAX = -1.9, 1.9

# Panels E/F sweep the effective noise s = sigma_repr*sqrt(t), from the
# deterministic policy (s = 0) to the uninformative one (s -> inf), exactly as
# panels A/B of fig2.py sweep the softmax temperature tau.
LEVEL_S = [np.inf, 3.2, 1.7, 0.95, 0.55, 0.0]

LEVEL_LABELS = [
    r"$\infty$",
    r"$3.2$",
    r"$1.7$",
    r"$0.95$",
    r"$0.55$",
    r"$0$",
]


def fam_level(family, k):
    """Colour of family `family` at stochasticity level k (fig2.py helper)."""
    gray_weight = GRAYFRAC[k]
    dark = DARK[family]
    return tuple(
        (1.0 - gray_weight) * dark_i + gray_weight * gray_i
        for dark_i, gray_i in zip(dark, GRAY_C)
    )


def level_color(k):
    return fam_level("S", k)


def belief(log10_odds):
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(log10_odds, dtype=float)))


def log10_odds(b, eps=1e-12):
    b = np.clip(np.asarray(b, dtype=float), eps, 1.0 - eps)
    return np.log10(b / (1.0 - b))


# ==========================================================================
# Model
# ==========================================================================
def solve():
    """Representation-noise model (hard internal boundary) plus the ideal
    (sigma=0) boundaries used as the fixed reference of the E/F sweep."""
    cfg_r = nmr.ModelConfig(c=COST, sigma_repr=SIGMA_REPR, tau_exec=0.0,
                            n_trials=NTRIAL, max_timestep=MAX_TIMESTEP,
                            n_gh=N_GH, seed=SEED)
    rr = nmr.compute_model(noise="representation", horizon="infinite",
                           config=cfg_r, verbose=False)

    cfg_0 = nmr.ModelConfig(c=COST, sigma_repr=0.0, tau_exec=0.0, n_gh=N_GH)
    r0 = nmr.compute_model(noise="representation", horizon="infinite",
                           config=cfg_0, verbose=False)
    return rr, r0


def simulate_internal_and_ground(res, n_trials=NTRIAL, seed=SEED):
    """Vectorized rep-noise simulation that records BOTH the internal (noisy)
    running log-odds L~ and the ground-truth running log-odds L, so panels C
    and D can show the identical trials in the two coordinate systems.
    Decisions use the hard internal threshold on L~."""
    rng = np.random.default_rng(seed)
    lo, hi = res.lower_l, res.upper_l
    ev = res.stimuli.evidence_log10
    p_a, p_b = res.stimuli.p_a, res.stimuli.p_b

    true_h = rng.integers(0, 2, size=n_trials)
    ground_L = np.zeros(n_trials)
    repr_L = np.zeros(n_trials)
    active = np.ones(n_trials, dtype=bool)
    rec = []                                    # per time step

    for t in range(1, MAX_TIMESTEP + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            rec.append(dict(gL=np.array([]), rL=np.array([]),
                            a=np.array([], dtype=int)))
            continue
        h = true_h[idx]
        cue = np.empty(idx.size, dtype=int)
        m0 = h == 0
        if m0.any():
            cue[m0] = rng.choice(len(ev), size=int(m0.sum()), p=p_a)
        if (~m0).any():
            cue[~m0] = rng.choice(len(ev), size=int((~m0).sum()), p=p_b)

        ell = ev[cue]
        ground_L[idx] += ell
        repr_L[idx] += ell + rng.normal(0.0, SIGMA_REPR, size=idx.size)

        rL = repr_L[idx]
        action = np.where(rL <= lo, 0, np.where(rL >= hi, 1, 2))
        rec.append(dict(gL=ground_L[idx].copy(), rL=rL.copy(), a=action))
        active[idx[action != 2]] = False

    return rec


# ==========================================================================
# Ground-view policy helpers
# ==========================================================================
def ps_ground(l_values, t, lower_l, upper_l, sigma=SIGMA_REPR):
    """p(sample | ground-truth L) after marginalizing the representation noise
    at time step t, i.e. at noise level s = sigma*sqrt(t)."""
    s = sigma * np.sqrt(t)
    return norm.cdf((upper_l - l_values) / s) - norm.cdf((lower_l - l_values) / s)


def policy_at_level(model, k, l_values=None):
    """(p0, p1, ps) as a function of the ground-truth evidence L at noise level
    LEVEL_S[k], the internal boundary being held at (lower_l, upper_l).
    k = 0                : the fully stochastic limit  (s -> inf)
    k = len(LEVEL_S) - 1 : the deterministic policy    (s = 0)
    """
    l_values = model.l_grid if l_values is None else np.asarray(l_values, float) # np.linspace(-13.2, 13.2, 2001) # 
    lower_l, upper_l = model.lower_l, model.upper_l
    s = LEVEL_S[k]

    if k == 0:                                   # s -> inf : (1/2, 1/2, 0)
        half = np.full_like(l_values, 0.5)
        return half, half, np.zeros_like(l_values)
    if s == 0.0:                                 # hard internal threshold
        p0 = (l_values <= lower_l).astype(float)
        p1 = (l_values >= upper_l).astype(float)
        return p0, p1, 1.0 - p0 - p1

    p0 = norm.cdf((lower_l - l_values) / s)
    p1 = norm.cdf((l_values - upper_l) / s)
    ps = norm.cdf((upper_l - l_values) / s) - norm.cdf((lower_l - l_values) / s)
    return p0, p1, ps


# ==========================================================================
# Panels A, transform, B -- hard internal policy -> soft apparent policy
# ==========================================================================
def panel_A(ax, res):
    """Hard internal policy: p(sample) vs internal belief b~."""
    b = res.b_grid
    ax.plot(b, res.p_sample, color=fam_level("S", 5), lw=2.2)
    for bb in (res.lower_b, res.upper_b):
        ax.axvline(bb, color="0.6", ls="--", lw=1.0, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.10)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel(r"$p(\mathrm{sample})$", fontsize=LBL_FS)
    ax.set_xlabel(r"Internal belief $\tilde b=P(H_1\mid \tilde L)$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)
    """
    ax.text(0.5, 0.55, "wait", ha="center", va="center", color=fam_level("S", 5),
            fontsize=ANN_FS, fontweight="bold")
    ax.text(0.09, 0.20, r"choose $H_0$", ha="center", color=fam_level("H0", 5),
            fontsize=ANN_FS)
    ax.text(0.91, 0.20, r"choose $H_1$", ha="center", color=fam_level("H1", 5),
            fontsize=ANN_FS)
            """


def panel_transform(ax, res):
    """Link between panels A and B: referring the hard internal policy to the
    ground-truth belief adds representation noise."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.annotate(r"$\tilde L\sim\mathcal{N}(L,\,s^{2})$", #r"$\tilde L = L + \xi,\ \ \xi\sim\mathcal{N}(0,\,s^2)$",
                xy=(0.5, 0.80), ha="center", va="center", fontsize=LBL_FS)
    ax.annotate("", xy=(0.5, 0.06), xytext=(0.5, 0.56),
                arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.0))


def panel_B(ax, res):
    """Soft apparent policy: p(sample) vs ground-truth Bayesian posterior b at a
    representative time step."""
    l_values = np.linspace(-13, 13, 1201)
    b = belief(l_values)
    ps = ps_ground(l_values, B_REP_T, res.lower_l, res.upper_l)
    ax.plot(b, ps, color=fam_level("S", 5), lw=2.4)

    # 50% crossings of the smeared transition zones
    crossings = l_values[np.where(np.diff(np.sign(ps - 0.5)) != 0)[0]]
    for l_cross in crossings:
        ax.axvline(float(belief(l_cross)), color="0.3", ls="--", lw=1.0, zorder=1)
    for bb in (res.lower_b, res.upper_b):
        ax.axvline(bb, color="0.75", ls=":", lw=0.9, zorder=0)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.10)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel(r"$p(\mathrm{sample})$", fontsize=LBL_FS)
    ax.set_xlabel(r"Ground-truth coordinate $b=P(H_1\mid L)$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)


# ==========================================================================
# Panels C, D -- 50k-trial simulation in internal vs ground-truth coordinates
# ==========================================================================
def _mean_segments(ax, y_all, a_all, t, col_w=COL_W):
    """Horizontal segment at the AVERAGE log odds of the committed trials of
    each type, spanning the width of that time step's density column."""
    for a, color in ((0, PALE_H0), (1, PALE_H1)):
        y = y_all[a_all == a]
        if y.size > 5:
            ax.plot([t, t + col_w], [y.mean(), y.mean()], color=color, lw=1.5)


def _violin_column(ax, rec, key, col_w=COL_W):
    y_grid = np.linspace(YMIN, YMAX, 320)
    for t, r in enumerate(rec, start=1):
        y_all = r[key]
        a_all = r["a"]
        dens = {}
        max_d = 0.0
        for a in (0, 1, 2):
            y = y_all[a_all == a]
            if y.size > 15 and np.std(y) > 1e-3:
                d = gaussian_kde(y, bw_method=0.32)(y_grid) * y.size
            else:
                d = np.zeros_like(y_grid)
            dens[a] = d
            max_d = max(max_d, float(d.max()))
        if max_d <= 0:
            continue
        for a, color in ((2, PASTEL["S"]), (0, PASTEL["H0"]), (1, PASTEL["H1"])):
            ax.fill_betweenx(y_grid, t, t + dens[a] / max_d * col_w,
                             color=color, alpha=0.3, lw=0)
        _mean_segments(ax, y_all, a_all, t, col_w)


def _hist_column(ax, rec, key, col_w=COL_W, bw=BW):
    """Histogram-style density columns: counts into bins CENTERED on the 0.1
    evidence lattice, drawn as horizontal step bars.  Because the optimal
    boundary is a non-lattice real, it falls in a between-bin valley -> clean
    separation of the chosen from the waiting trials."""
    edges = np.arange(YMIN - bw / 2.0, YMAX + bw / 2.0 + 1e-9, bw)
    centers = edges[:-1] + bw / 2.0
    for t, r in enumerate(rec, start=1):
        y_all = r[key]
        a_all = r["a"]
        hist = {}
        max_count = 0.0
        for a in (0, 1, 2):
            y = y_all[a_all == a]
            counts, _ = np.histogram(y, bins=edges)
            hist[a] = counts.astype(float)
            max_count = max(max_count, float(counts.max()) if counts.size else 0.0)
        if max_count <= 0:
            continue
        for a, color in ((2, PASTEL["S"]), (0, PASTEL["H0"]), (1, PASTEL["H1"])):
            ax.fill_betweenx(centers, t, t + hist[a] / max_count * col_w,
                             step="mid", color=color, alpha=0.3, lw=0)
        _mean_segments(ax, y_all, a_all, t, col_w)


def _trial_type_legend(ax):
    handles = [
        Patch(facecolor=PASTEL["H0"], edgecolor="none", alpha=0.3, label=r"chosen $H_0$"),
        Patch(facecolor=PASTEL["S"], edgecolor="none", alpha=0.3, label=r"sampling"),
        Patch(facecolor=PASTEL["H1"], edgecolor="none", alpha=0.3, label=r"chosen $H_1$"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=LEG_FS, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0)


def panel_C(ax, rec, res, show_legend=True):
    """Same trials in INTERNAL coordinates: the hard boundary cleanly separates
    the chosen from the waiting trials."""
    _hist_column(ax, rec, "rL")
    for l_bound in (res.lower_l, res.upper_l):
        ax.plot([0.6, MAX_TIMESTEP + COL_W + 0.2], [l_bound, l_bound],
                color="0.15", ls="--", lw=1.3)
    ax.axhline(0, color="0.9", lw=0.6, zorder=0)
    ax.set_xlim(0.6, MAX_TIMESTEP + COL_W + 0.3)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(range(1, MAX_TIMESTEP + 1))
    ax.set_xlabel("Time step", fontsize=LBL_FS)
    ax.set_ylabel(r"Internal evidence $\tilde L$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)
    if show_legend:
        _trial_type_legend(ax)


def panel_D(ax, rec, res, show_legend=False):
    """Same trials in GROUND-TRUTH coordinates: chosen and waiting distributions
    overlap; dashed lines are the per-step 50% p(sample) crossings."""
    _violin_column(ax, rec, "gL")
    l_values = np.linspace(YMIN, YMAX, 2001)
    lower_cr, upper_cr, ts_ok = [], [], []
    for t in range(1, MAX_TIMESTEP + 1):
        ps = ps_ground(l_values, t, res.lower_l, res.upper_l)
        crossings = l_values[np.where(np.diff(np.sign(ps - 0.5)) != 0)[0]]
        if crossings.size >= 2:
            lower_cr.append(crossings.min())
            upper_cr.append(crossings.max())
            ts_ok.append(t)
    ax.plot(ts_ok, lower_cr, color="0.15", ls="--", lw=1.3, marker="o", ms=2.6)
    ax.plot(ts_ok, upper_cr, color="0.15", ls="--", lw=1.3, marker="o", ms=2.6)
    ax.axhline(0, color="0.9", lw=0.6, zorder=0)
    ax.set_xlim(0.6, MAX_TIMESTEP + COL_W + 0.3)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(range(1, MAX_TIMESTEP + 1))
    ax.set_xlabel("Time step", fontsize=LBL_FS)
    ax.set_ylabel(r"Cumulative evidence $L$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)
    if show_legend:
        _trial_type_legend(ax)


# ==========================================================================
# Panel E -- simplex
# ==========================================================================
def simplex_xy(p0, p1, ps):
    return np.stack([p1 + 0.5 * ps, TRI_H * ps], axis=-1)


def draw_simplex(ax):
    ax.add_patch(Polygon(
        [[0, 0], [0.5, TRI_H], [1, 0]], closed=True,
        facecolor="1", edgecolor="none", zorder=-2,
    ))
    ax.plot([0, 0.5, 1, 0], [0, TRI_H, 0, 0], color="k", lw=1.0, zorder=4)
    ax.text(0.5, TRI_H + 0.045, r"$p_S$", ha="center", va="bottom", fontsize=LBL_FS)
    ax.text(-0.02, -0.05, r"$p_0$", ha="left", va="top", fontsize=LBL_FS)
    ax.text(1.02, -0.05, r"$p_1$", ha="right", va="top", fontsize=LBL_FS)
    ax.set_xlim(-0.16, 1.16)
    ax.set_ylim(-0.12, TRI_H + 0.12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def simplex_curve(ax, p0, p1, ps, b, lw=2.6, z=3):
    xy = simplex_xy(np.asarray(p0), np.asarray(p1), np.asarray(ps))
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    la = LineCollection(
        segments, cmap=plt.get_cmap(B_CMAP), norm=B_NORM, lw=lw, zorder=z,
    )
    la.set_array(0.5 * (b[:-1] + b[1:]))
    ax.add_collection(la)
    return la


def simplex_pt(ax, p0, p1, ps, color, size=DOT_S):
    xy = simplex_xy(np.array([p0]), np.array([p1]), np.array([ps]))[0]
    ax.scatter([xy[0]], [xy[1]], s=size + 55, facecolors="white",
               edgecolors="white", zorder=20, lw=0)
    ax.scatter([xy[0]], [xy[1]], s=size, facecolors=[color],
               edgecolors="white", lw=0.8, zorder=21)


def panel_E(ax, r0):
    """Simplex trajectories across representation-noise levels.  As s grows the
    whole family migrates to the p0-p1 EDGE and piles up at (0.5, 0.5, 0) --
    unlike execution noise (fig2 panel A), whose limit is the centroid.
    Trajectories are coloured by the ground-truth coordinate b (blue=H0 .. red=H1).
    """
    draw_simplex(ax)
    b = r0.b_grid
    sub = slice(0, len(b), 6)
    la = None
    for k in range(len(LEVEL_S) - 1, -1, -1):
        p0, p1, ps = policy_at_level(r0, k)
        if k == 0:
            simplex_pt(ax, 0.5, 0.5, 0.0, level_color(0))
            continue
        la = simplex_curve(ax, p0[sub], p1[sub], ps[sub], b[sub], lw=2.6, z=3 + k)
        simplex_pt(ax, p0[0], p1[0], ps[0], fam_level("H0", k))
        simplex_pt(ax, p0[-1], p1[-1], ps[-1], fam_level("H1", k))

    # The two qualitatively different limits.
    centroid = simplex_xy(np.array([1 / 3]), np.array([1 / 3]), np.array([1 / 3]))[0]
    ax.scatter([centroid[0]], [centroid[1]], s=DOT_S + 26, facecolors="none",
               edgecolors="0.35", lw=1.4, zorder=25)
    ax.annotate("Random policy\n" + r"$(\frac{1}{3},\frac{1}{3},\frac{1}{3})$",
                xy=(centroid[0], centroid[1]), xytext=(0.15, 0.52),
                textcoords="data", fontsize=LBL_FS, color="0.15",
                ha="right", va="center", zorder=1,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.75),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.35"))
    ax.text(0.5, -0.025, r"$s\!\to\!\infty:\ (\frac{1}{2},\frac{1}{2},0)$",
            ha="center", va="top", fontsize=LBL_FS, color="0.15", zorder=27)
    
    return la


# ==========================================================================
# Panel F -- noise sweep 
# ==========================================================================
def panel_F(ax_top, ax_bot, r0):
    """Two stacked sub-panels sharing the belief axis.
    Top: p(sample) vs b for the LEVEL_S noise levels -- the plateau collapses
    toward 0 (never toward 1/3, which is the execution-noise limit of fig2).
    Bottom: the corresponding p(choose H0) and p(choose H1), which converge to
    1/2 rather than to 1/3."""
    b = r0.b_grid
    for k in range(len(LEVEL_S) - 1, -1, -1):
        p0, p1, ps = policy_at_level(r0, k)
        if k == 0:                       # s -> inf : p_sample = 0, p_choose = 1/2
            ax_top.axhline(0.0, color=GRAY_C, lw=1.4)
            ax_bot.axhline(0.5, color=GRAY_C, lw=1.4)
            continue
        ax_top.plot(b, ps, color=fam_level("S", k), lw=1.4, zorder=3 + k)
        ax_bot.plot(b, p0, color=fam_level("H0", k), lw=1.4, zorder=3 + k)
        ax_bot.plot(b, p1, color=fam_level("H1", k), lw=1.4, zorder=3 + k)

    handles = [
        Line2D([0], [0], color=level_color(k), lw=1.4, label=LEVEL_LABELS[k])
        for k in [5, 4, 3, 2, 1, 0]
    ]
    ax_top.legend(
        handles=handles, title=r"Noise ($s=\sigma\sqrt{t}$)",
        fontsize=LEG_FS, title_fontsize=TITLE_FS, frameon=True,
        facecolor="white", edgecolor="none", framealpha=0.86,
        loc="upper left", bbox_to_anchor=(1.03, 1.0), ncol=1, handlelength=1.1,
        columnspacing=0.75, labelspacing=0.28, borderaxespad=0.0)
    ax_top.set_ylabel(r"$p(\mathrm{sample})$", fontsize=LBL_FS)

    leg = [Line2D([0], [0], color=fam_level("S", 5), lw=2, label=r"$p(\mathrm{sample})$"),
           Line2D([0], [0], color=fam_level("H0", 5), lw=2, label=r"$p(\mathrm{choose}\ H_0)$"),
           Line2D([0], [0], color=fam_level("H1", 5), lw=2, label=r"$p(\mathrm{choose}\ H_1)$")]
    ax_bot.legend(handles=leg, frameon=False, fontsize=LEG_FS, loc="upper center",
                  ncol=1, handlelength=1.0, handletextpad=0.3,
                  columnspacing=0.8, borderaxespad=0.2,
                  bbox_to_anchor=(0.5, 1.1))
    ax_bot.set_ylabel(r"$p(\mathrm{choose})$", fontsize=LBL_FS)
    ax_bot.set_xlabel(r"Ground truth ($b$)", fontsize=LBL_FS)
    for ax in (ax_top, ax_bot):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.04, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.axvline(0.5, color="0.85", ls=":", lw=0.7)
        ax.tick_params(labelsize=TICK_FS)
    plt.setp(ax_top.get_xticklabels(), visible=False)


# ==========================================================================
# Layout
# ==========================================================================
def add_panel_label(ax, label, x_offset=-30, y_offset=6):
    ax.annotate(
        label, xy=(0, 1), xycoords="axes fraction",
        xytext=(x_offset, y_offset), textcoords="offset points",
        fontsize=PANEL_FS, fontweight="bold", va="bottom",
        annotation_clip=False,
    )


def main():
    rr, r0 = solve()
    rec = simulate_internal_and_ground(rr)

    fig = plt.figure(figsize=(14.2, 8.4))
    # A 2-band x 3-column master grid.
    #   band 1 (row 0) : A (top) | C | E (simplex)
    #   band 2 (row 1) : B (bot) | D | F (noise sweep)
    # The left column (A / equation+arrow / B) spans both bands.
    outer = GridSpec(16, 3, figure=fig,
                     height_ratios=[1.0] * 16,
                     width_ratios=[2.0, 2.9, 2.5],
                     hspace=0.42, wspace=0.36,
                     left=0.052, right=0.885, top=0.90, bottom=0.075)

    # ---- left column : A / equation+arrow / B  (spans both bands)
    gab = outer[0:16, 0].subgridspec(3, 1, height_ratios=[1.0, 0.2, 1.0],
                                    hspace=0.5)
    ax_a = fig.add_subplot(gab[0])
    ax_t = fig.add_subplot(gab[1])
    ax_b = fig.add_subplot(gab[2])
    panel_A(ax_a, rr)
    panel_transform(ax_t, rr)
    panel_B(ax_b, rr)

    # ---- middle column : C (top) over D (bottom)
    ax_c = fig.add_subplot(outer[0:7, 1])
    ax_d = fig.add_subplot(outer[9:16, 1])
    panel_C(ax_c, rec, rr, show_legend=True)
    panel_D(ax_d, rec, rr, show_legend=False)

    # ---- right column : E = simplex (top), F = noise sweep (bottom)
    # Same construction as the A-over-B column of fig2.py: the simplex keeps a
    # full grid cell and carries its colorbar as an inset, and the two stacked
    # sub-panels of F are inset by one margin unit on each side.
    ax_e = fig.add_subplot(outer[0:8, 2])
    la = panel_E(ax_e, r0)

    grid_f_width = outer[8:16, 2].subgridspec(1, 3, width_ratios=[2, 20, 1], wspace=0)
    grid_f = grid_f_width[0, 1].subgridspec(2, 1, hspace=0.14)
    ax_ft = fig.add_subplot(grid_f[0])
    ax_fb = fig.add_subplot(grid_f[1], sharex=ax_ft)
    panel_F(ax_ft, ax_fb, r0)

    # Belief colorbar to the right of panel E, without shrinking its square cell.
    if la is not None:
        cax = inset_axes(
            ax_e, width="3.0%", height="60%", loc="center right",
            bbox_to_anchor=(0, 0.0, 1.0, 1.0),
            bbox_transform=ax_e.transAxes, borderpad=0,
        )
        colorbar = fig.colorbar(la, cax=cax)
        colorbar.set_label(r"Ground truth ($b$)", fontsize=LBL_FS, labelpad=4)
        colorbar.ax.yaxis.set_label_position("right")
        colorbar.ax.yaxis.set_ticks_position("right")
        colorbar.ax.tick_params(labelsize=TICK_FS)
        colorbar.set_ticks([0, 0.5, 1])
        colorbar.outline.set_linewidth(0.75)

    for ax, label in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D"),
                      (ax_e, "E"), (ax_ft, "F")):
        add_panel_label(ax, label)

    # ---------------------------- save --------------------------------- #
    svg_path = os.path.join(outdir, "fig3.svg")
    png_path = os.path.join(outdir, "fig3.png")
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    print(
        "Saved", svg_path, "and", png_path, "\n"
        f"internal hard boundary L=({rr.lower_l:.3f}, {rr.upper_l:.3f}) "
        f"b=({rr.lower_b:.3f}, {rr.upper_b:.3f}); "
        f"ideal (sigma=0) boundary L=({r0.lower_l:.3f}, {r0.upper_l:.3f}) "
        f"used for the E/F sweep")


if __name__ == "__main__":
    main()