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
internal (b~) vs ground-truth (b) coordinates.  Panel E shows the ground-view
p(sample) flattening over time as the representational noise accumulates
(s_t = sigma*sqrt(t)).  Panels F/G show that as the representation noise grows
the policy converges to the (0.5, 0.5, 0) point on the p0-p1 edge of the
simplex -- a signature that is qualitatively DIFFERENT from execution noise,
which instead converges to the centroid (1/3, 1/3, 1/3).

Convention (per project spec)
    axis labels : fontsize 12
    tick labels : fontsize 10
    sampling cost c = 0.04 in every simulation.

Outputs: fig3.svg (Illustrator-editable) and fig3.png (preview).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, FancyArrowPatch
from scipy.stats import norm, gaussian_kde

import noise_model_runner as nmr

# Use Arial (as in the other figure scripts).  If Arial is not installed
# locally, also register the metric-identical Liberation Sans as a fallback so
# the preview render still looks right; Arial stays first so Illustrator uses it
# when the SVG is opened on the lab machine.
_FONT_FAMILY = ["Arial"]
try:
    import glob as _glob
    import matplotlib.font_manager as _fm
    if not any("arial" in f.name.lower() for f in _fm.fontManager.ttflist):
        for _f in _glob.glob(
            "/usr/share/fonts/**/LiberationSans-*.ttf", recursive=True):
            _fm.fontManager.addfont(_f)
        _FONT_FAMILY = ["Arial", "Liberation Sans", "DejaVu Sans"]
except Exception:
    pass

plt.rcParams.update({
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.family": _FONT_FAMILY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
})

# ------------------------------------------------------------------ styling
LBL_FS, TICK_FS = 12, 10               # project-wide font requirement

_FAMILY = {
    "H0": ((0.50, 0.00, 0.00), (1.00, 0.72, 0.72)),   # red   (choose H0)
    "H1": ((0.00, 0.00, 0.50), (0.72, 0.78, 1.00)),   # blue  (choose H1)
    "S":  ((0.00, 0.35, 0.00), (0.62, 0.90, 0.62)),   # green (sample)
}
def fcol(key, frac=0.0):
    d, l = _FAMILY[key]; frac = float(np.clip(frac, 0, 1))
    return tuple(a + frac * (b - a) for a, b in zip(d, l))

PALE_H0 = (0.92, 0.55, 0.55)
PALE_H1 = (0.55, 0.62, 0.92)
GRAY_C = (0.62, 0.62, 0.62)
TRI_H = np.sqrt(3.0) / 2.0
B_CMAP = "RdBu"
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
    ax.set_title("Hard policy in\nobserver's own belief", fontsize=10, loc="left")
    ax.text(0.5, 0.55, "wait", ha="center", va="center", color=fcol("S"),
            fontsize=10, fontweight="bold")
    ax.text(0.09, 0.20, r"choose $H_0$", ha="center", color=fcol("H0"), fontsize=8)
    ax.text(0.91, 0.20, r"choose $H_1$", ha="center", color=fcol("H1"), fontsize=8)


def panel_transform(ax, res):
    """Illustrate that referring b~ to the ground-truth belief b convolves each
    hard edge with the representation-noise kernel (x-axis gets 'smeared')."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    lo_b, hi_b = res.lower_b, res.upper_b
    s = SIGMA_REPR * np.sqrt(B_REP_T)

    ax.annotate(r"$\tilde L = L + \xi,\ \ \xi\sim\mathcal{N}(0,\,\sigma^2 t)$",
                xy=(0.5, 0.98), ha="center", va="top", fontsize=10.5)

    # smearing kernel (in belief coords) around each boundary
    for cb, cL in ((lo_b, res.lower_l), (hi_b, res.upper_l)):
        xx = np.linspace(0, 1, 400)
        k = norm.pdf(log10_odds(xx), loc=cL, scale=s)
        k = k / k.max() * 0.42
        ax.fill_between(xx, 0.06, 0.06 + k, color=fcol("S", 0.35),
                        alpha=0.75, lw=0)
        ax.plot([cb, cb], [0.66, 0.80], color="0.55", ls="--", lw=1.0,
                clip_on=False)
        # fan arrows: sharp edge (top, from panel A) -> spread (bottom, panel B)
        for dx in (-2.2 * s, 0, 2.2 * s):
            xb = float(belief(cL + dx))
            ax.add_patch(FancyArrowPatch((cb, 0.66), (xb, 0.14),
                         arrowstyle="-|>", mutation_scale=8, lw=1.0,
                         color="0.45", connectionstyle="arc3,rad=0.0",
                         clip_on=False))
    ax.text(0.5, 0.50, r"each hard edge $\Rightarrow$ smeared over $b$",
            ha="center", va="center", fontsize=9.5, color=fcol("S", 0.0))


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
    ax.set_title("Soft policy in\nground-truth belief", fontsize=10, loc="left")
    ax.text(0.03, 0.60, r"$p_{\mathrm{sample}}(b)=\Phi\!\frac{\tilde\lambda_+-L}{s_t}"
            r"-\Phi\!\frac{\tilde\lambda_--L}{s_t}$",
            transform=ax.transAxes, fontsize=9, va="center")


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
        for act, col in ((2, fcol("S")), (0, fcol("H0")), (1, fcol("H1"))):
            ax.fill_betweenx(ld, t, t + dens[act] / mx * col_w,
                             color=col, alpha=0.45, lw=0)
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
        for act, col in ((2, fcol("S")), (0, fcol("H0")), (1, fcol("H1"))):
            ax.fill_betweenx(yc, t, t + hist[act] / maxc * col_w, step="mid",
                             color=col, alpha=0.5, lw=0)
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
    ax.set_title(r"Internal belief space  (dashed = hard $\tilde\lambda_\pm$)",
                 fontsize=10, loc="left")


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
    ax.set_title(r"Ground-truth space  (dashed = 50% $p(\mathrm{sample})$)",
                 fontsize=10, loc="left")

    leg = [Line2D([0], [0], marker="s", ms=7, ls="none", mfc=fcol("H0"),
                  mec="none", alpha=0.6, label=r"chosen $H_0$"),
           Line2D([0], [0], marker="s", ms=7, ls="none", mfc=fcol("H1"),
                  mec="none", alpha=0.6, label=r"chosen $H_1$"),
           Line2D([0], [0], marker="s", ms=7, ls="none", mfc=fcol("S"),
                  mec="none", alpha=0.6, label="sampling")]
    ax.legend(handles=leg, frameon=False, fontsize=8, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              handletextpad=0.3, columnspacing=1.0)


# ================================================================== new panel E (internal staircase)
def panel_Enew(ax, res):
    """Companion to panel C, drawn in the SAME orientation as panel F (internal
    belief on x, p(sample) on y): the optimal policy in the observer's own
    belief is a HARD staircase -- p(sample)=1 iff beta_- < b~ < beta_+ -- a
    boxcar, in direct contrast with the soft, broadening ground curves in F."""
    lo_b, hi_b = res.lower_b, res.upper_b
    ax.fill_between([lo_b, hi_b], 0.0, 1.0, color=fcol("S"), alpha=0.12, lw=0)
    ax.plot([0, lo_b, lo_b, hi_b, hi_b, 1.0], [0, 0, 1, 1, 0, 0],
            color=fcol("S"), lw=2.2)
    for bb in (lo_b, hi_b):
        ax.axvline(bb, color="0.15", ls="--", lw=1.1, zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.10); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"Internal belief $\tilde b=P(H_1\mid \tilde L)$", fontsize=LBL_FS)
    ax.set_ylabel(r"$p(\mathrm{sample})$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.text(0.5, 0.55, "wait", ha="center", va="center", color=fcol("S"),
            fontsize=10, fontweight="bold")
    ax.text(0.5 * lo_b, 0.20, r"choose $H_0$", ha="center", color=fcol("H0"),
            fontsize=7.5)
    ax.text(0.5 * (hi_b + 1.0), 0.20, r"choose $H_1$", ha="center",
            color=fcol("H1"), fontsize=7.5)
    ax.set_title("Hard internal\npolicy", fontsize=10, loc="left")


# ================================================================== panel E (time)
def panel_E(ax, res):
    """Ground-view p(sample) vs belief b over time: flattens / broadens as the
    representational noise s_t = sigma*sqrt(t) accumulates."""
    Lg = np.linspace(-3, 3, 1001)
    bg = belief(Lg)
    tcol = plt.cm.plasma(np.linspace(0.08, 0.92, MAX_T))
    for t in range(1, MAX_T + 1):
        ps = ps_ground(Lg, t, res.lower_l, res.upper_l)
        ax.plot(bg, ps, color=tcol[t - 1], lw=1.7)
    ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"Bayesian belief $b=P(H_1)$", fontsize=LBL_FS)
    ax.set_ylabel(r"$p(\mathrm{sample})$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_title("Noise accumulates:\ncurves broaden over time", fontsize=10, loc="left")

    sm = plt.cm.ScalarMappable(cmap="plasma",
                               norm=plt.Normalize(vmin=1, vmax=MAX_T))
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Time step", fontsize=LBL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    cbar.set_ticks([1, 5, 10])
    # widening annotation
    ax.annotate("", xy=(0.86, 0.30), xytext=(0.64, 0.72),
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.2))
    ax.text(0.60, 0.80, r"$t{:}1\!\to\!10$", fontsize=9, color="0.35")


# ================================================================== panels F / G (noise sweep)
def _noise_color(k):
    return plt.cm.viridis(np.linspace(0.05, 0.9, len(S_LEVELS)))[k]


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


def panel_F(ax, r0):
    """Simplex trajectories across representation-noise levels.  As s grows the
    whole family migrates to the p0-p1 EDGE and its endpoints pile up at the
    (0.5, 0.5, 0) point -- unlike execution noise, whose limit is the centroid."""
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


def panel_G(ax, r0):
    """p(sample) vs b for the same noise levels: peak collapses toward 0 (never
    toward 1/3, which is where execution noise would settle)."""
    lo, hi = r0.lower_l, r0.upper_l
    Lg = np.linspace(-3, 3, 1001)
    bg = belief(Lg)
    for k, s in enumerate(S_LEVELS):
        _, _, ps = policy_ground_s(Lg, s, lo, hi)
        ax.plot(bg, ps, color=_noise_color(k), lw=1.9, label=S_LABELS[k])
    ax.axhline(1/3, color="0.4", ls="--", lw=1.1)
    ax.text(0.02, 1/3 + 0.02, r"execution-noise limit $\to \frac{1}{3}$",
            fontsize=8.5, color="0.4", va="bottom")
    ax.annotate("", xy=(0.5, 0.06), xytext=(0.5, 0.9),
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.2))
    ax.text(0.52, 0.5, r"$s\!\uparrow$", fontsize=11, color="0.35")

    ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 1/3, 0.5, 1])
    ax.set_yticklabels(["0", r"$\frac{1}{3}$", "0.5", "1"])
    ax.set_xlabel(r"Bayesian belief $b=P(H_1)$", fontsize=LBL_FS)
    ax.set_ylabel(r"$p(\mathrm{sample})$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_title("Peak collapses toward 0", fontsize=10, loc="left")
    leg = ax.legend(title=r"Noise $s=\sigma\sqrt{t}$", fontsize=8.5,
                    title_fontsize=9, frameon=True, facecolor="white",
                    edgecolor="none", framealpha=0.85, loc="upper right",
                    handlelength=1.1, labelspacing=0.25, borderaxespad=0.2)


# ================================================================== layout
def add_label(ax, txt, dx=-32, dy=8):
    ax.annotate(txt, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontsize=13, fontweight="bold", va="bottom",
                annotation_clip=False)


def main():
    rr, r0 = solve()
    rec = simulate_internal_and_ground(rr)

    fig = plt.figure(figsize=(14.0, 11.65))
    # A 3-band x 3-column master grid.
    #   band 1 (rows 0) : A(top) | C        | E(new, internal staircase)
    #   band 2 (rows 1) : B(bot) | D        | F(= old E, ground p(sample)/time)
    #   band 3 (rows 2) :  ----  | G(=old F)| H(= old G)
    # The left column (A / transform / B) spans the two upper bands.
    #   widths  1/4 : 3/8 : 1/4   ->  2 : 3 : 2
    #   bottom band height = 1.5 x its previous value (0.64 -> 0.96)
    outer = GridSpec(3, 3, figure=fig,
                     height_ratios=[1.0, 1.0, 0.96],
                     width_ratios=[2.0, 3.0, 2.0],
                     hspace=0.46, wspace=0.36,
                     left=0.055, right=0.955, top=0.955, bottom=0.055)

    # ---- left column : A / transform / B  (occupies the two upper bands)
    gab = outer[0:2, 0].subgridspec(3, 1, height_ratios=[1.0, 0.5, 1.0],
                                    hspace=0.55)
    ax_A = fig.add_subplot(gab[0])
    ax_T = fig.add_subplot(gab[1])
    ax_B = fig.add_subplot(gab[2])
    panel_A(ax_A, rr)
    panel_transform(ax_T, rr)
    panel_B(ax_B, rr)

    # ---- middle column : C (top band) over D (mid band)
    ax_C = fig.add_subplot(outer[0, 1])
    ax_D = fig.add_subplot(outer[1, 1])
    panel_C(ax_C, rec, rr)
    panel_D(ax_D, rec, rr)

    # ---- right column : new E (top band) over F = old E (mid band)
    ax_E = fig.add_subplot(outer[0, 2])
    ax_F = fig.add_subplot(outer[1, 2])
    panel_Enew(ax_E, rr)
    panel_E(ax_F, rr)          # ground-view p(sample) over time -> now panel F
    # F is narrowed by its own colorbar; match E's plot box to F so the two
    # belief axes line up vertically (E directly above F).
    posF = ax_F.get_position(); posE = ax_E.get_position()
    ax_E.set_position([posF.x0, posE.y0, posF.width, posE.height])

    # ---- bottom band : G = old F (under C/D) , H = old G (under E/F)
    ax_G = fig.add_subplot(outer[2, 1])
    ax_H = fig.add_subplot(outer[2, 2])
    lc = panel_F(ax_G, r0)     # simplex trajectories -> now panel G
    panel_G(ax_H, r0)          # p(sample) vs b -> now panel H

    # belief colorbar for the simplex (panel G)
    cbar = fig.colorbar(lc, ax=ax_G, fraction=0.045, pad=0.02,
                        orientation="vertical")
    cbar.set_label(r"Belief $b$", fontsize=LBL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    cbar.set_ticks([0, 0.5, 1])

    for ax, L in ((ax_A, "A"), (ax_B, "B"), (ax_C, "C"), (ax_D, "D"),
                  (ax_E, "E"), (ax_F, "F"), (ax_G, "G"), (ax_H, "H")):
        add_label(ax, L)

    outdir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(outdir, "fig3.svg"))
    fig.savefig(os.path.join(outdir, "fig3.png"), dpi=200)
    print("Saved fig3.svg / fig3.png")
    print("internal hard boundary  L=(%.3f, %.3f)  b=(%.3f, %.3f)"
          % (rr.lower_l, rr.upper_l, rr.lower_b, rr.upper_b))
    print("ideal (sigma=0) boundary L=(%.3f, %.3f) used for F/G sweep"
          % (r0.lower_l, r0.upper_l))


if __name__ == "__main__":
    main()
