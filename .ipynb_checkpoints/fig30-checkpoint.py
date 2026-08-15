#!/usr/bin/env python3
"""
Figure 3 -- Representation noise makes decision boundaries *look* soft even
though the observer's internal policy stays a hard threshold.

Key message
-----------
A noisy internal representation L~ = L + xi (xi ~ N(0, sigma^2 * t)) keeps the
OPTIMAL policy a hard staircase in the observer's own belief b~ (panel A), but
when the experimenter refers the same behaviour to the ground-truth belief b,
marginalizing over the noise convolves the step with a Gaussian kernel and the
policy appears soft (panel B).  Panels C/D contrast the same 50,000 trials in
internal (b~) vs ground-truth (b) coordinates.  Panel E (simplex) shows that as
the representation noise grows the policy converges to the (0.5, 0.5, 0) point
on the p0-p1 edge -- qualitatively DIFFERENT from execution noise, whose limit
is the centroid (1/3, 1/3, 1/3).  Panel F shows p(sample) collapsing toward 0
as noise grows (with the s->infinity asymptote at 0), and, below it, the
corresponding p(choose H0)/p(choose H1) fading from blue/red to gray.

Colour convention: choose H0 = blue, choose H1 = red, sample = green.

Convention (per project spec)
    axis labels : fontsize 12
    tick labels : fontsize 10
    sampling cost c = 0.04 in every simulation.

"""

import os
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, FancyArrowPatch, Patch
from scipy.stats import norm, gaussian_kde

import noise_model_runner as nmr

# ------------------------------------------------------------------ output directory
fallback_outdir = str(Path.home())
try:
    outdir = os.path.dirname(os.path.abspath(__file__))
except (NameError, TypeError):
    outdir = fallback_outdir

# ------------------------------------------------------------------ styling
plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": ["Arial", "Sans"],
    "pdf.fonttype": 42,
    "mathtext.fontset": "stix", #"dejavusans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
})


LBL_FS, TICK_FS = 12, 10               # project-wide font requirement

_FAMILY = {
    "H0": ((0.00, 0.00, 0.50), (0.72, 0.78, 1.00)),   # blue  (choose H0)
    "H1": ((0.50, 0.00, 0.00), (1.00, 0.72, 0.72)),   # red   (choose H1)
    "S":  ((0.00, 0.35, 0.00), (0.62, 0.90, 0.62)),   # green (sample)
}
def fcol(key, frac=0.0):
    d, l = _FAMILY[key]; frac = float(np.clip(frac, 0, 1))
    return tuple(a + frac * (b - a) for a, b in zip(d, l))

# Fixed pastel trio for trial-type coding in panels C, D
PASTEL = {"H0": "blue", "H1": "red", "S": "green"}
PALE_H0 = (0.55, 0.62, 0.92)   # pale blue (choose H0)
PALE_H1 = (0.92, 0.55, 0.55)   # pale red  (choose H1)
GRAY_C = (0.62, 0.62, 0.62)
TRI_H = np.sqrt(3.0) / 2.0
B_CMAP = "RdBu_r"              # b=0 (H0) -> blue, b=1 (H1) -> red
B_NORM = TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)

# ------------------------------------------------------------------ settings
COST = 0.04
SIGMA_REPR = 0.25          # per-step representational noise SD (in log10 odds)
NTRIAL = 50_000
MAX_T = 10
SEED = 1
B_REP_T = 4                # representative time step used for panel B (s=sig*sqrt(t))
YMIN, YMAX = -1.9, 1.9

# noise-level sweep for panels F/G: effective s = sigma*sqrt(t), small -> huge
S_LEVELS = [0.15, 0.30, 0.55, 0.95, 1.7, 3.2]
S_LABELS = [f"{s:.2f}" for s in S_LEVELS]


def belief(L):
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(L, dtype=float)))

def log10_odds(b, eps=1e-12):
    b = np.clip(np.asarray(b, dtype=float), eps, 1 - eps)
    return np.log10(b / (1 - b))


# ================================================================== model
def solve():
    """Representation-noise model (hard internal boundary) plus the ideal
    (sigma=0) boundaries used as the fixed reference in the F/G sweep."""
    cfg_r = nmr.ModelConfig(c=COST, sigma_repr=SIGMA_REPR, tau_exec=0.0,
                            n_trials=NTRIAL, max_timestep=MAX_T, n_gh=121,
                            seed=SEED)
    rr = nmr.compute_model(noise="representation", horizon="infinite",
                           config=cfg_r, verbose=False)

    cfg_0 = nmr.ModelConfig(c=COST, sigma_repr=0.0, tau_exec=0.0, n_gh=121)
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
    pa, pb = res.stimuli.p_a, res.stimuli.p_b

    true_h = rng.integers(0, 2, size=n_trials)
    ground_L = np.zeros(n_trials)
    repr_L = np.zeros(n_trials)
    active = np.ones(n_trials, dtype=bool)
    rec = []                                    # per time step

    for t in range(1, MAX_T + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            rec.append(dict(gL=np.array([]), rL=np.array([]),
                            a=np.array([], dtype=int)))
            continue
        h = true_h[idx]
        cue = np.empty(idx.size, dtype=int)
        m0 = h == 0
        if m0.any():
            cue[m0] = rng.choice(len(ev), size=int(m0.sum()), p=pa)
        if (~m0).any():
            cue[~m0] = rng.choice(len(ev), size=int((~m0).sum()), p=pb)

        ell = ev[cue]
        ground_L[idx] += ell
        repr_L[idx] += ell + rng.normal(0.0, SIGMA_REPR, size=idx.size)

        rL = repr_L[idx]
        action = np.where(rL <= lo, 0, np.where(rL >= hi, 1, 2))
        rec.append(dict(gL=ground_L[idx].copy(), rL=rL.copy(), a=action))
        active[idx[action != 2]] = False

    return rec


# ================================================================== ground-view policy helpers
def ps_ground(L, t, lo, hi, sigma=SIGMA_REPR):
    """p(sample | ground-truth L) after marginalizing rep. noise, at step t."""
    s = sigma * np.sqrt(t)
    return norm.cdf((hi - L) / s) - norm.cdf((lo - L) / s)

def policy_ground_s(L, s, lo, hi):
    """(p0, p1, ps) as a function of ground-truth L for a fixed noise level s,
    with the internal boundary held at (lo, hi)."""
    ps = norm.cdf((hi - L) / s) - norm.cdf((lo - L) / s)
    p1 = norm.cdf((L - hi) / s)
    p0 = norm.cdf((lo - L) / s)
    return p0, p1, ps


# ================================================================== panel A / transform / B
def panel_A(ax, res):
    """Hard internal policy: p(sample) vs internal belief b~."""
    b = res.b_grid
    ax.plot(b, res.p_sample, color=fcol("S"), lw=2.2)
    for bb in (res.lower_b, res.upper_b):
        ax.axvline(bb, color="0.6", ls="--", lw=1.0, zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.10); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel(r"$p(\mathrm{sample})$", fontsize=LBL_FS)
    ax.set_xlabel(r"Internal belief $\tilde b=P(H_1\mid \tilde L)$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)
    # ax.set_title("Hard policy in\nobserver's own belief", fontsize=10, loc="left")
    ax.text(0.5, 0.55, "wait", ha="center", va="center", color=fcol("S"),
            fontsize=10, fontweight="bold")
    ax.text(0.09, 0.20, r"choose $H_0$", ha="center", color=fcol("H0"), fontsize=8)
    ax.text(0.91, 0.20, r"choose $H_1$", ha="center", color=fcol("H1"), fontsize=8)


def panel_transform(ax, res):
    """Link between panels A and B: referring the hard internal policy to the
    ground-truth belief adds representation noise.  Only the transformation
    equation is shown, with a thin arrow indicating the A -> B transition."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.annotate(r"$\tilde L = L + \xi,\ \ \xi\sim\mathcal{N}(0,\,\sigma^2 t)$",
                xy=(0.5, 0.80), ha="center", va="center", fontsize=11)
    # thin arrow, centered, indicating the A -> B transition
    ax.annotate("", xy=(0.5, 0.06), xytext=(0.5, 0.56),
                arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.0))


def panel_B(ax, res):
    """Soft apparent policy: p(sample) vs ground-truth belief b at a
    representative time step."""
    Lg = np.linspace(-3, 3, 1201)
    bg = belief(Lg)
    ps = ps_ground(Lg, B_REP_T, res.lower_l, res.upper_l)
    ax.plot(bg, ps, color=fcol("S"), lw=2.4)

    # shade the smeared transition zones + mark 50% crossings
    cr = Lg[np.where(np.diff(np.sign(ps - 0.5)) != 0)[0]]
    for L in cr:
        ax.axvline(float(belief(L)), color="0.3", ls="--", lw=1.0, zorder=1)
    for bb in (res.lower_b, res.upper_b):
        ax.axvline(bb, color="0.75", ls=":", lw=0.9, zorder=0)

    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.10); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel(r"$p(\mathrm{sample})$", fontsize=LBL_FS)
    ax.set_xlabel(r"Bayesian belief $b=P(H_1\mid L)$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)


# ================================================================== panels C / D (simulations)
def _violin_column(ax, rec, key, col_w=0.85):
    ld = np.linspace(YMIN, YMAX, 320)
    for t, r in enumerate(rec, start=1):
        y_all = r[key]; a_all = r["a"]
        dens = {}; mx = 0.0
        for act in (0, 1, 2):
            y = y_all[a_all == act]
            if y.size > 15 and np.std(y) > 1e-3:
                d = gaussian_kde(y, bw_method=0.32)(ld) * y.size
            else:
                d = np.zeros_like(ld)
            dens[act] = d; mx = max(mx, float(d.max()))
        if mx <= 0:
            continue
        for act, col in ((2, PASTEL["S"]), (0, PASTEL["H0"]), (1, PASTEL["H1"])):
            ax.fill_betweenx(ld, t, t + dens[act] / mx * col_w,
                             color=col, alpha=0.3, lw=0)
        for act, col in ((0, PALE_H0), (1, PALE_H1)):
            y = y_all[a_all == act]
            if y.size > 5:
                ax.plot([t, t + col_w], [y.mean(), y.mean()], color=col, lw=1.5)


def _hist_column(ax, rec, key, col_w=0.85, bw=0.1):
    """Histogram-style density columns (same construction as panel D of
    fig1.py): counts into bins CENTERED on the 0.1 evidence lattice, drawn as
    horizontal step bars.  Because the optimal boundary is a non-lattice real,
    it falls in a between-bin valley -> clean-cut separation of chosen vs
    waiting trials."""
    edges = np.arange(YMIN - bw / 2.0, YMAX + bw / 2.0 + 1e-9, bw)
    yc = edges[:-1] + bw / 2.0
    for t, r in enumerate(rec, start=1):
        y_all = r[key]; a_all = r["a"]
        hist = {}; maxc = 0.0
        for act in (0, 1, 2):
            y = y_all[a_all == act]
            h, _ = np.histogram(y, bins=edges)
            hist[act] = h.astype(float)
            maxc = max(maxc, float(h.max()) if h.size else 0.0)
        if maxc <= 0:
            continue
        for act, col in ((2, PASTEL["S"]), (0, PASTEL["H0"]), (1, PASTEL["H1"])):
            ax.fill_betweenx(yc, t, t + hist[act] / maxc * col_w, step="mid",
                             color=col, alpha=0.3, lw=0)
        for act, col in ((0, PALE_H0), (1, PALE_H1)):
            y = y_all[a_all == act]
            if y.size > 5:
                ax.plot([t, t + col_w], [y.mean(), y.mean()], color=col, lw=1.5)


def panel_C(ax, rec, res):
    """Same trials in INTERNAL coordinates: hard boundary cleanly separates
    chosen vs waiting trials."""
    _hist_column(ax, rec, "rL")
    for L in (res.lower_l, res.upper_l):
        ax.plot([0.6, MAX_T + 1.05], [L, L], color="0.15", ls="--", lw=1.3)
    ax.axhline(0, color="0.9", lw=0.6, zorder=0)
    ax.set_xlim(0.6, MAX_T + 1.1); ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(range(1, MAX_T + 1))
    ax.set_xlabel("Time step", fontsize=LBL_FS)
    ax.set_ylabel(r"Internal evidence $\tilde L$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)
    
    leg = [Patch(facecolor=PASTEL["H0"], edgecolor="none", alpha=0.3, label=r"chosen $H_0$"),
               Patch(facecolor=PASTEL["S"], edgecolor="none", alpha=0.3, label=r"sampling"),
               Patch(facecolor=PASTEL["H1"], edgecolor="none", alpha=0.3, label=r"chosen $H_1$")]
    ax.legend(handles=leg, frameon=False, fontsize=12, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0)


def panel_D(ax, rec, res):
    """Same trials in GROUND-TRUTH coordinates: chosen and waiting distributions
    overlap; dashed lines are the per-step 50% p(sample) crossings."""
    _violin_column(ax, rec, "gL")
    Lg = np.linspace(YMIN, YMAX, 2001)
    lower_cr, upper_cr, ts_ok = [], [], []
    for t in range(1, MAX_T + 1):
        ps = ps_ground(Lg, t, res.lower_l, res.upper_l)
        cr = Lg[np.where(np.diff(np.sign(ps - 0.5)) != 0)[0]]
        if cr.size >= 2:
            lower_cr.append(cr.min()); upper_cr.append(cr.max()); ts_ok.append(t)
    ax.plot(ts_ok, lower_cr, color="0.15", ls="--", lw=1.3, marker="o", ms=2.6)
    ax.plot(ts_ok, upper_cr, color="0.15", ls="--", lw=1.3, marker="o", ms=2.6)
    ax.axhline(0, color="0.9", lw=0.6, zorder=0)
    ax.set_xlim(0.6, MAX_T + 1.1); ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(range(1, MAX_T + 1))
    ax.set_xlabel("Time step", fontsize=LBL_FS)
    ax.set_ylabel(r"Cumulative evidence $L$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)


# ================================================================== panels E / F (noise sweep)
def _noise_color(k):
    return plt.cm.viridis(np.linspace(0.05, 0.9, len(S_LEVELS)))[k]


def _to_gray(base, frac):
    """Interpolate a base color toward gray as `frac` goes 0 -> 1."""
    frac = float(np.clip(frac, 0, 1))
    return tuple(b + frac * (g - b) for b, g in zip(base, GRAY_C))


def draw_simplex(ax):
    ax.add_patch(Polygon([[0, 0], [0.5, TRI_H], [1, 0]], closed=True,
                         facecolor="1", edgecolor="none", zorder=-2))
    ax.plot([0, 0.5, 1, 0], [0, TRI_H, 0, 0], color="k", lw=1.0, zorder=4)
    ax.text(0.5, TRI_H + 0.05, r"$p_{\mathrm{sample}}$", ha="center", va="bottom",
            fontsize=LBL_FS)
    ax.text(-0.03, -0.05, r"$p_0$", ha="left", va="top", fontsize=LBL_FS)
    ax.text(1.03, -0.05, r"$p_1$", ha="right", va="top", fontsize=LBL_FS)
    ax.set_xlim(-0.17, 1.17); ax.set_ylim(-0.14, TRI_H + 0.16)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def simplex_xy(p0, p1, ps):
    return np.stack([p1 + 0.5 * ps, TRI_H * ps], axis=-1)


def panel_simplex(ax, r0):
    """(New panel E) Simplex trajectories across representation-noise levels.
    As s grows the whole family migrates to the p0-p1 EDGE and its endpoints
    pile up at the (0.5, 0.5, 0) point -- unlike execution noise, whose limit
    is the centroid.  Trajectories colored by belief b (blue=H0 .. red=H1)."""
    draw_simplex(ax)
    lo, hi = r0.lower_l, r0.upper_l
    Lg = np.linspace(-3.2, 3.2, 700)
    bg = belief(Lg)
    lc = None
    for k, s in enumerate(S_LEVELS):
        p0, p1, ps = policy_ground_s(Lg, s, lo, hi)
        xy = simplex_xy(p0, p1, ps)
        seg = np.stack([xy[:-1], xy[1:]], axis=1)
        lc = LineCollection(seg, cmap=plt.get_cmap(B_CMAP), norm=B_NORM,
                            lw=2.2, zorder=3 + k, alpha=0.95)
        lc.set_array(0.5 * (bg[:-1] + bg[1:]))
        ax.add_collection(lc)

    # the two qualitatively different limit points
    c_xy = simplex_xy(np.array([1/3]), np.array([1/3]), np.array([1/3]))[0]
    e_xy = simplex_xy(np.array([0.5]), np.array([0.5]), np.array([0.0]))[0]
    ax.scatter([c_xy[0]], [c_xy[1]], s=70, facecolors="none",
               edgecolors="0.35", lw=1.4, zorder=25)
    ax.annotate("execution-noise\nlimit  " + r"$(\frac{1}{3},\frac{1}{3},\frac{1}{3})$",
                xy=(c_xy[0], c_xy[1]), xytext=(0.66, 0.60),
                textcoords="axes fraction", fontsize=8.5, color="0.35",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.35"))
    ax.scatter([e_xy[0]], [e_xy[1]], s=70, facecolors=fcol("S", 0.0),
               edgecolors="white", lw=0.9, zorder=26)
    ax.annotate("representation-noise\nlimit  " + r"$(\frac{1}{2},\frac{1}{2},0)$",
                xy=(e_xy[0], e_xy[1]), xytext=(0.5, -0.02),
                textcoords="axes fraction", fontsize=8.5, color=fcol("S", 0.0),
                ha="center", va="top",
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=fcol("S", 0.0)))
    # collapse direction: apex drops toward the p0-p1 edge as noise grows
    ax.annotate("", xy=(0.5, 0.16 * TRI_H), xytext=(0.5, 0.80 * TRI_H),
                arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.3))
    ax.text(0.53, 0.5 * TRI_H, r"$s\!\uparrow$", fontsize=11, color="0.4")
    ax.set_title(r"Convergence to the $p_0$-$p_1$ edge", fontsize=10, loc="left")
    return lc


def panel_peak(ax_top, ax_bot, r0):
    """(New panel F) Two stacked sub-panels sharing the belief x-axis.
    Top: p(sample) vs b for the same noise levels s -- the peak collapses
    toward 0 (never toward 1/3, the execution-noise limit); the horizontal
    line at 0 is the s->infinity asymptote.  Bottom: the corresponding
    p(choose H0) and p(choose H1), each graded from saturated blue / red
    (low noise) to gray (high noise)."""
    lo, hi = r0.lower_l, r0.upper_l
    Lg = np.linspace(-3, 3, 1001)
    bg = belief(Lg)
    n = len(S_LEVELS)

    # ---- top sub-panel : p(sample) --------------------------------------
    for k, s in enumerate(S_LEVELS):
        _, _, ps = policy_ground_s(Lg, s, lo, hi)
        ax_top.plot(bg, ps, color=_noise_color(k), lw=1.9, label=S_LABELS[k])
    ax_top.axhline(1/3, color="0.4", ls="--", lw=1.1)
    ax_top.text(0.02, 1/3 + 0.02, r"execution-noise limit $\to \frac{1}{3}$",
                fontsize=8.5, color="0.4", va="bottom")
    # s -> infinity asymptote : p(sample) = 0
    ax_top.axhline(0.0, color="0.12", ls=(0, (5, 2)), lw=1.4, zorder=6)
    ax_top.text(0.985, 0.035, r"$s\!\to\!\infty$", fontsize=9, color="0.12",
                ha="right", va="bottom")
    ax_top.annotate("", xy=(0.5, 0.06), xytext=(0.5, 0.9),
                    arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.2))
    ax_top.text(0.52, 0.5, r"$s\!\uparrow$", fontsize=11, color="0.35")

    ax_top.set_xlim(0, 1); ax_top.set_ylim(-0.03, 1.05)
    ax_top.set_yticks([0, 1/3, 0.5, 1])
    ax_top.set_yticklabels(["0", r"$\frac{1}{3}$", "0.5", "1"])
    ax_top.set_ylabel(r"$p(\mathrm{sample})$", fontsize=LBL_FS)
    ax_top.tick_params(labelsize=TICK_FS)
    # ax_top.set_title("Peak collapses toward 0", fontsize=10, loc="left")
    ax_top.legend(title=r"Noise $s=\sigma\sqrt{t}$", fontsize=8,
                  title_fontsize=8.5, frameon=True, facecolor="white",
                  edgecolor="none", framealpha=0.85, loc="upper right",
                  handlelength=1.0, labelspacing=0.22, borderaxespad=0.2)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # ---- bottom sub-panel : p(choose H0), p(choose H1) ------------------
    base0, base1 = fcol("H0", 0.0), fcol("H1", 0.0)   # dark blue / dark red
    for k, s in enumerate(S_LEVELS):
        p0, p1, _ = policy_ground_s(Lg, s, lo, hi)
        frac = k / max(n - 1, 1)
        ax_bot.plot(bg, p0, color=_to_gray(base0, frac), lw=1.7)
        ax_bot.plot(bg, p1, color=_to_gray(base1, frac), lw=1.7)
    ax_bot.set_xlim(0, 1); ax_bot.set_ylim(-0.03, 1.05); ax_bot.set_yticks([0, 0.5, 1])
    ax_bot.set_xlabel(r"Bayesian belief $b=P(H_1)$", fontsize=LBL_FS)
    ax_bot.set_ylabel(r"$p(\mathrm{choose})$", fontsize=LBL_FS)
    ax_bot.tick_params(labelsize=TICK_FS)
    leg2 = [Line2D([0], [0], color=base0, lw=2.2, label=r"$p(H_0)$"),
            Line2D([0], [0], color=base1, lw=2.2, label=r"$p(H_1)$"),
            Line2D([0], [0], color=GRAY_C, lw=2.2, label=r"$s\!\uparrow$ (to gray)")]
    ax_bot.legend(handles=leg2, frameon=False, fontsize=8, loc="upper center",
                  ncol=3, handlelength=1.1, columnspacing=0.9, handletextpad=0.4,
                  bbox_to_anchor=(0.5, 1.0))


# ================================================================== layout
def add_label(ax, txt, dx=-32, dy=8):
    ax.annotate(txt, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontsize=13, fontweight="bold", va="bottom",
                annotation_clip=False)


def main():
    rr, r0 = solve()
    rec = simulate_internal_and_ground(rr)

    fig = plt.figure(figsize=(13.6, 8.4))
    # A 2-band x 3-column master grid.
    #   band 1 (row 0) : A(top) | C | E (simplex)
    #   band 2 (row 1) : B(bot) | D | F (peak collapse: p_sample over p_choose)
    # The left column (A / equation+arrow / B) spans both bands.
    #   widths  1/4 : 3/8 : 1/4  ->  2 : 3 : 2
    outer = GridSpec(2, 3, figure=fig,
                     height_ratios=[1.0, 1.0],
                     width_ratios=[2.0, 3.0, 2.0],
                     hspace=0.42, wspace=0.36,
                     left=0.055, right=0.945, top=0.94, bottom=0.075)

    # ---- left column : A / equation+arrow / B  (spans both bands)
    gab = outer[0:2, 0].subgridspec(3, 1, height_ratios=[1.0, 0.1, 1.0],
                                    hspace=0.5)
    ax_A = fig.add_subplot(gab[0])
    ax_T = fig.add_subplot(gab[1])
    ax_B = fig.add_subplot(gab[2])
    panel_A(ax_A, rr)
    panel_transform(ax_T, rr)
    panel_B(ax_B, rr)

    # ---- middle column : C (top) over D (bottom)
    ax_C = fig.add_subplot(outer[0, 1])
    ax_D = fig.add_subplot(outer[1, 1])
    panel_C(ax_C, rec, rr)
    panel_D(ax_D, rec, rr)

    # ---- right column : E = simplex (top) , F = peak collapse (bottom)
    ax_E = fig.add_subplot(outer[0, 2])
    lc = panel_simplex(ax_E, r0)

    gF = outer[1, 2].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.08)
    ax_F_top = fig.add_subplot(gF[0])
    ax_F_bot = fig.add_subplot(gF[1], sharex=ax_F_top)
    panel_peak(ax_F_top, ax_F_bot, r0)

    # belief colorbar for the simplex (panel E)
    cbar = fig.colorbar(lc, ax=ax_E, fraction=0.045, pad=0.02,
                        orientation="vertical")
    cbar.set_label(r"Belief $b$", fontsize=LBL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    cbar.set_ticks([0, 0.5, 1])

    for ax, L in ((ax_A, "A"), (ax_B, "B"), (ax_C, "C"), (ax_D, "D"),
                  (ax_E, "E"), (ax_F_top, "F")):
        add_label(ax, L)

    fig.savefig(os.path.join(outdir, "fig3.svg"))
    fig.savefig(os.path.join(outdir, "fig3.png"), dpi=200)
    print("Saved fig3.svg / fig3.png")
    print("internal hard boundary  L=(%.3f, %.3f)  b=(%.3f, %.3f)"
          % (rr.lower_l, rr.upper_l, rr.lower_b, rr.upper_b))
    print("ideal (sigma=0) boundary L=(%.3f, %.3f) used for E/F sweep"
          % (r0.lower_l, r0.upper_l))


if __name__ == "__main__":
    main()
