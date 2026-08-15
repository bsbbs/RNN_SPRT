#!/usr/bin/env python3
"""
Figure 1 (reorganized) -- Task schematic + Q-values (A,B) on top, followed by
deterministic-vs-soft policy comparison (C-E deterministic, F-H soft), all
sharing one color convention: blue = H0, red = H1, green = sample/wait.

This file merges and reorganizes the original fig1.py (panels A,B) with
fig_compare.py (panels A-F there, relettered C-H here), per the following
layout (9x9 grid; 1 "column unit" = 2 grid cols, 1 "row unit" = 2 grid rows):

Row band 0 (rows 0:3, 1.5 rows):
    Panel A : task schema (diagonal) + cue-frequency distribution, 2.5 cols
    Panel B : Q-value curves (infinite horizon), 1 col
    Equations: thin leader lines from curves to Q1/Q0/QS formulas, 1 col

Row band 1 (rows 3:6, 1.5 rows) -- deterministic (hard-boundary) policy:
    Panel C : p(sample)/p(choose) vs belief, 1 col
    Panel D : RGB policy field (green/red/blue) + hard boundaries, 1.5 col
    Panel E : simulated-trial cumulative-evidence histograms, 2 col

Row band 2 (rows 6:9, 1.5 rows) -- soft (tau=0.04) policy:
    Panel F : p(sample)/p(choose) vs belief, 1 col
    Panel G : RGB policy field (green/red/blue), soft boundaries, 1.5 col
    Panel H : simulated-trial cumulative-evidence KDE distributions, 2 col

Color convention (fixed throughout, matches original fig1.py):
    blue family  = H0    red family = H1    green family = sample / wait
Panels E, H additionally use the fixed pastel trio
    sample = #B3D9B3, H1 = #FFB3B3, H0 = #B3B3FF
which is reused for the D/G policy-field background for visual consistency.

Axis-label fontsize = 12, tick-label fontsize = 10 (project convention).
Sample cost c = 0.04 throughout.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import to_rgb
from scipy.stats import gaussian_kde

import noise_model_runner as nmr

# --------------------------------------------------------------------------
# Global style
# --------------------------------------------------------------------------
plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": ["Arial", "Liberation Sans"],
    "pdf.fonttype": 42,
    "mathtext.fontset": "stix",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
})

AXLAB = 12     # axis-label fontsize (project convention)
TICKLAB = 10   # tick-label fontsize (project convention)

# Single color convention used throughout the whole figure (matches fig1.py):
#   H0 = blue family, H1 = red family, S = green family (sample / wait)
_FAMILY = {
    "H0": ((0.00, 0.00, 0.50), (0.72, 0.78, 1.00)),
    "H1": ((0.50, 0.00, 0.00), (1.00, 0.72, 0.72)),
    "S":  ((0.00, 0.35, 0.00), (0.62, 0.90, 0.62)),
}
def fcol(key, frac=0.0):
    d, l = _FAMILY[key]; frac = float(np.clip(frac, 0, 1))
    return tuple(a + frac * (b - a) for a, b in zip(d, l))

PALE_H0 = (0.92, 0.55, 0.55)     # pale blue/red mean-line accents (unused here)
PALE_H1 = (0.55, 0.62, 0.92)

# Fixed pastel trio for trial-type coding in panels D, E, G, H
PASTEL = {"S": "#B3D9B3", "H1": "#FFB3B3", "H0": "#B3B3FF"}
PASTEL_RGB = {k: np.array(to_rgb(v)) for k, v in PASTEL.items()}

# --------------------------------------------------------------------------
# Model / simulation settings
# --------------------------------------------------------------------------
COST = 0.04
TAU = 0.04
NTRIAL = 50_000
MAX_T = 10
DEADLINE = 11
BW = 0.1
YMIN, YMAX = -1.6, 1.6
YMIN_F, YMAX_F = -2.0, 2.0


def solve():
    """Infinite-horizon models: deterministic (hard) and soft (tau=0.04).
    Both share the same value function settings (c=0.04, sigma_repr=0);
    'rdet' doubles as the source for the task-schema / Q-value panels
    (A, B) since that model is identical to fig1.py's 'ri'.
    """
    base = dict(c=COST, sigma_repr=0.0, n_trials=NTRIAL,
                max_timestep=MAX_T, deadline=DEADLINE, n_gh=81)
    rdet = nmr.compute_model(noise="representation", horizon="infinite",
                              config=nmr.ModelConfig(tau_exec=0.0, **base), verbose=False)
    ddet = nmr.simulate_trials(rdet, verbose=False)
    rsoft = nmr.compute_model(noise="execution", horizon="infinite",
                               config=nmr.ModelConfig(tau_exec=TAU, **base), verbose=False)
    dsoft = nmr.simulate_trials(rsoft, verbose=False)
    return rdet, ddet, rsoft, dsoft


# ==========================================================================
# Panel A -- task schema (diagonal) + cue-frequency distribution
# ==========================================================================
def panel_A(ax_sch, ax_dist, res):
    glyphs = ["\u25CF", "\u25B2", "\u2716", "\u2605", ""]
    labels = ["Fixation", "Cue 1", "Cue 2", "Cue 3", "..."]
    w, h = 1.55, 1.15
    dx, dy = 2.05, 0.92           # diagonal step increments
    x0, y0 = 0.35, 0.35
    centers = [(x0 + i * dx, y0 + i * dy) for i in range(len(labels))]

    ax_sch.set_xlim(-0.3, x0 + (len(labels) - 1) * dx + 2.7)
    ax_sch.set_ylim(-1.5, y0 + (len(labels) - 1) * dy + 2.6)
    ax_sch.axis("off")

    for (cx, cy), lab, gl in zip(centers, labels, glyphs):
        ax_sch.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h,
                                    fill=False, lw=0.9, ec="0.2", zorder=3))
        ax_sch.text(cx, cy + 0.06, gl, ha="center", va="center", fontsize=12, zorder=4)
        ax_sch.text(cx, cy - h / 2 - 0.14, lab, ha="center", va="top",
                    fontsize=6.8, zorder=4)
        if gl:
            yb = cy - h / 2 - 0.55
            ax_sch.add_patch(FancyArrowPatch((cx - 0.18, yb), (cx - 0.95, yb - 0.35),
                             arrowstyle="-|>", mutation_scale=8, lw=1.6,
                             color=fcol("H0"), clip_on=False, zorder=3))
            ax_sch.add_patch(FancyArrowPatch((cx + 0.18, yb), (cx + 0.95, yb - 0.35),
                             arrowstyle="-|>", mutation_scale=8, lw=1.6,
                             color=fcol("H1"), clip_on=False, zorder=3))

    # label the choice arrows once (under the first cue box)
    cx1, cy1 = centers[1]
    yb1 = cy1 - h / 2 - 0.55
    ax_sch.text(cx1 - 0.95, yb1 - 0.55, r"choose $H_0$", color=fcol("H0"),
                fontsize=6.8, ha="center", va="top")
    ax_sch.text(cx1 + 0.95, yb1 - 0.55, r"choose $H_1$", color=fcol("H1"),
                fontsize=6.8, ha="center", va="top")

    # green dashed sampling arcs, diagonally between consecutive boxes
    for i in range(len(centers) - 2):
        (x1, y1), (x2, y2) = centers[i], centers[i + 1]
        p1 = (x1 + w / 2 * 0.55, y1 + h / 2 * 0.75)
        p2 = (x2 - w / 2 * 0.55, y2 - h / 2 * 0.55)
        ax_sch.add_patch(FancyArrowPatch(p1, p2,
                         connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                         mutation_scale=9, lw=1.3, ls="--", color=fcol("S"),
                         clip_on=False, zorder=2))
    mx = np.mean([c[0] for c in centers[:3]])
    my = np.mean([c[1] for c in centers[:3]]) + 1.05
    ax_sch.text(mx, my, r"sample ($-c$ per step)",
                ha="center", fontsize=7.2, color=fcol("S"), rotation=18)

    cxl, cyl = centers[-1]
    ax_sch.text(cxl + 0.05, cyl, "choice:\n+1 / 0", ha="center",
                va="center", fontsize=6.8, color="0.15")
    ax_sch.text(0.0, ax_sch.get_ylim()[1] * 0.99, "Sequential sampling task",
                fontsize=8.5, ha="left", va="top")

    # cue-frequency distribution, attached to the right of the diagonal
    ev = res.stimuli.evidence_log10
    pa, pb = res.stimuli.p_a, res.stimuli.p_b
    off = 0.012
    ax_dist.vlines(ev - off, 0, pa, color=fcol("H0"), lw=1.2, alpha=0.9)
    ax_dist.plot(ev - off, pa, "o", ms=2.6, color=fcol("H0"), label=r"$H_0$")
    ax_dist.vlines(ev + off, 0, pb, color=fcol("H1"), lw=1.2, alpha=0.9)
    ax_dist.plot(ev + off, pb, "o", ms=2.6, color=fcol("H1"), label=r"$H_1$")
    markers = ["^", "v", "D", "o", "s", "*", "P", "X",
               "p", "h", "<", ">", "d", "8", "H", "."]
    ymark = 0.07 * max(pa.max(), pb.max())
    for xv, mk in zip(ev, markers):
        ax_dist.plot(xv, ymark, marker=mk, ms=3.2, color="0.25", clip_on=False, ls="none")
    ax_dist.set_xticks(ev)
    ax_dist.set_xticklabels([f"{v:+.1f}" if i in [0, 2, 4, 6, 9, 11, 13, 15, 17] else ""
                             for i, v in enumerate(ev)], fontsize=TICKLAB)
    ax_dist.tick_params(axis="x", which="major", pad=8, labelsize=TICKLAB)
    ax_dist.tick_params(axis="y", labelsize=TICKLAB)
    ax_dist.set_xlabel(r"Cue $\log_{10}$ odds", fontsize=AXLAB)
    ax_dist.xaxis.set_label_coords(0.5, -0.16)
    ax_dist.set_ylabel("Frequency", fontsize=AXLAB)
    ax_dist.set_ylim(ymark * 1.6, max(pa.max(), pb.max()) * 1.15)
    ax_dist.legend(
        frameon=False, fontsize=7, loc="upper center",
        bbox_to_anchor=(0.5, 1.16), ncol=2,
        handletextpad=0.3, columnspacing=0.9, borderaxespad=0.0,
    )
    ax_dist.set_title("Cue frequencies", fontsize=8.5, loc="left")


# ==========================================================================
# Panel B -- Q-values with leader-line equations to the right
# ==========================================================================
def panel_B(ax_v, ax_eq, res):
    b = res.b_grid
    ax_v.plot(b, res.q0, color=fcol("H0"), lw=2.0)
    ax_v.plot(b, res.q1, color=fcol("H1"), lw=2.0)
    ax_v.plot(b, res.q_wait, color=fcol("S"), lw=2.0)
    for bb in (res.lower_b, res.upper_b):
        ax_v.axvline(bb, color="0.75", ls="--", lw=1.0, zorder=0)
    ax_v.set_xlim(0, 1); ax_v.set_ylim(-0.03, 1.05)
    ax_v.set_xlabel(r"Belief $b = P(H_1)$", fontsize=AXLAB)
    ax_v.set_ylabel("Value", fontsize=AXLAB)
    ax_v.tick_params(labelsize=TICKLAB)
    ax_v.set_title("Action values", fontsize=8.5, loc="left")

    # anchor points on each curve (near the right edge, in data coords)
    b0 = 0.86
    anchors = {
        "H1": (b0, float(np.interp(b0, b, res.q1))),
        "H0": (1 - b0, float(np.interp(1 - b0, b, res.q0))),
        "S":  (0.5, float(np.interp(0.5, b, res.q_wait))),
    }

    ax_eq.set_xlim(0, 1); ax_eq.set_ylim(0, 1); ax_eq.axis("off")
    eq_text = {
        "H1": r"$Q_1 = r\,b$",
        "H0": r"$Q_0 = r\,(1-b)$",
        "S":  r"$Q_S = -c + \mathbb{E}_{e\mid b}\!\left[V(\mathcal{B}(b,e))\right]$",
    }
    eq_ypos = {"H1": 0.82, "S": 0.5, "H0": 0.18}
    for key in ("H1", "S", "H0"):
        xy_data = anchors[key]
        ax_eq.annotate(
            "", xy=(0.02, eq_ypos[key]), xycoords=ax_eq.transAxes,
            xytext=xy_data, textcoords=ax_v.transData,
            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.8,
                             shrinkA=0, shrinkB=2),
            annotation_clip=False,
        )
        ax_eq.text(0.06, eq_ypos[key], eq_text[key], color=fcol(key),
                    fontsize=9.5, ha="left", va="center")


# ==========================================================================
# Panels C, F -- p(sample) / p(choose) vs belief (mini policy panels)
# ==========================================================================
def panel_policy_mini(ax_top, ax_bot, res, show_legend):
    b = res.b_grid
    ax_top.plot(b, res.p_sample, color=fcol("S"), lw=2.0)
    ax_bot.plot(b, res.p_choose0, color=fcol("H0"), lw=2.0)
    ax_bot.plot(b, res.p_choose1, color=fcol("H1"), lw=2.0)

    betas = (float(res.lower_b), float(res.upper_b))
    if show_legend:
        leg = [Line2D([0], [0], color=fcol("S"), lw=2, label=r"$p(\mathrm{sample})$"),
               Line2D([0], [0], color=fcol("H0"), lw=2, label=r"$p(\mathrm{choose}\ H_0)$"),
               Line2D([0], [0], color=fcol("H1"), lw=2, label=r"$p(\mathrm{choose}\ H_1)$")]
        ax_top.legend(handles=leg, frameon=False, fontsize=7.5, loc="upper center",
                      ncol=1, handlelength=1.0, handletextpad=0.3,
                      columnspacing=0.8, borderaxespad=0.2,
                      bbox_to_anchor=(0.5, 1.55))

    ax_top.set_ylabel(r"$p(\mathrm{sample})$", fontsize=AXLAB)
    ax_bot.set_ylabel(r"$p(\mathrm{choose})$", fontsize=AXLAB)
    ax_bot.set_xlabel(r"Belief $b = P(H_1)$", fontsize=AXLAB)
    for ax in (ax_top, ax_bot):
        ax.set_xlim(0, 1); ax.set_ylim(-0.06, 1.12)
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=TICKLAB)
        for bb in betas:
            ax.axvline(bb, color="0.75", ls="--", lw=1.0, zorder=0)
    plt.setp(ax_top.get_xticklabels(), visible=False)


# ==========================================================================
# Panels D, G -- RGB policy field (green/red/blue blend) + boundaries
# ==========================================================================
def panel_field(ax, res, mode, show_legend):
    L = res.l_grid
    m = (L >= YMIN) & (L <= YMAX)
    Lm = L[m]
    ps, p1, p0 = res.p_sample[m], res.p_choose1[m], res.p_choose0[m]
    rgb = (ps[:, None] * PASTEL_RGB["S"] + p1[:, None] * PASTEL_RGB["H1"]
           + p0[:, None] * PASTEL_RGB["H0"])
    img = np.repeat(rgb[:, None, :], 2, axis=1)
    ax.imshow(img, extent=[0.5, MAX_T + 0.5, YMIN, YMAX], origin="lower",
              aspect="auto", interpolation="bilinear", zorder=0)

    if mode == "hard":
        lo, hi = float(res.lower_l), float(res.upper_l)
        for lam in (lo, hi):
            ax.plot([0.5, MAX_T + 0.5], [lam, lam], color="0.1", ls="--", lw=1.3)
        title = "Deterministic policy field"
    else:
        cross = np.where(np.diff(np.sign(ps - 0.5)) != 0)[0]
        for lam in Lm[cross]:
            ax.plot([0.5, MAX_T + 0.5], [lam, lam], color="0.1", ls="--", lw=1.0)
        title = "Soft policy field"

    ax.axhline(0, color=(1, 1, 1, 0.6), lw=0.6, zorder=1)
    ax.set_xlim(0.5, MAX_T + 0.5); ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(range(1, MAX_T + 1))
    ax.set_xlabel("Time step", fontsize=AXLAB)
    ax.set_ylabel(r"Cumulative evidence ($\log_{10}$ odds)", fontsize=AXLAB)
    ax.tick_params(labelsize=TICKLAB)
    ax.set_title(title, fontsize=10, loc="left", pad=3)

    if show_legend:
        leg = [Patch(facecolor=PASTEL["S"], edgecolor="none", label=r"$p(\mathrm{sample})$"),
               Patch(facecolor=PASTEL["H1"], edgecolor="none", label=r"$p(\mathrm{choose}\ H_1)$"),
               Patch(facecolor=PASTEL["H0"], edgecolor="none", label=r"$p(\mathrm{choose}\ H_0)$")]
        ax.legend(handles=leg, frameon=False, fontsize=7.5, ncol=3,
                  loc="lower center", bbox_to_anchor=(0.5, 1.14),
                  handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0)


# ==========================================================================
# Panel E -- deterministic 50k-trial sim, histogram columns
# ==========================================================================
def panel_sim_hist(ax, df, res, col_w=0.85, show_legend=True):
    edges = np.arange(YMIN - 0.05, YMAX + 0.05 + 1e-9, BW)
    yc = edges[:-1] + BW / 2.0
    for t in range(1, MAX_T + 1):
        dft = df[df["time_step"] == t]
        hist, maxc = {}, 0.0
        for a in (0, 1, 2):
            y = dft.loc[dft["action"] == a, "ground_L"].to_numpy()
            h, _ = np.histogram(y, bins=edges)
            hist[a] = h.astype(float)
            maxc = max(maxc, h.max())
        if maxc <= 0:
            continue
        for a, col in ((2, PASTEL["S"]), (0, PASTEL["H0"]), (1, PASTEL["H1"])):
            ax.fill_betweenx(yc, t, t + hist[a] / maxc * col_w, step="mid",
                             color=col, alpha=0.85, lw=0)

    for Lb in (res.lower_l, res.upper_l):
        ax.plot([0.6, MAX_T + col_w + 0.2], [Lb, Lb],
                color="0.2", ls="--", lw=1.1)
    ax.axhline(0, color="0.9", lw=0.6, zorder=0)
    ax.set_xlim(0.6, MAX_T + col_w + 0.3)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(range(1, MAX_T + 1))
    ax.set_xlabel("Time step", fontsize=AXLAB)
    ax.set_ylabel("Cumulative evidence\n(Log likelihood ratio)", fontsize=AXLAB)
    ax.tick_params(labelsize=TICKLAB)
    ax.set_title("Simulated trials (50k)", fontsize=10, loc="left", pad=3)

    if show_legend:
        leg = [Line2D([0], [0], marker="s", ms=7, ls="none", mfc=PASTEL["H0"],
                      mec="none", label=r"chosen $H_0$"),
               Line2D([0], [0], marker="s", ms=7, ls="none", mfc=PASTEL["H1"],
                      mec="none", label=r"chosen $H_1$"),
               Line2D([0], [0], marker="s", ms=7, ls="none", mfc=PASTEL["S"],
                      mec="none", label="sampling")]
        ax.legend(handles=leg, frameon=False, fontsize=8, ncol=3,
                  loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  handletextpad=0.3, columnspacing=1.0)


# ==========================================================================
# Panel H -- soft 50k-trial sim, smoothed (KDE) distributions
# ==========================================================================
def panel_sim_kde(ax, df, res, col_w=0.82, show_legend=True):
    l_density = np.linspace(YMIN_F, YMAX_F, 300)
    for t in range(1, MAX_T + 1):
        dft = df[df["time_step"] == t]
        densities, maxd = {}, 0.0
        for a in (0, 1, 2):
            y = dft.loc[dft["action"] == a, "ground_L"].to_numpy()
            if y.size > 15 and np.std(y) > 1e-3:
                dens = gaussian_kde(y, bw_method=0.35)(l_density) * y.size
            else:
                dens = np.zeros_like(l_density)
            densities[a] = dens
            maxd = max(maxd, float(dens.max()))
        if maxd <= 0.0:
            continue
        for a, col in ((2, PASTEL["S"]), (0, PASTEL["H0"]), (1, PASTEL["H1"])):
            ax.fill_betweenx(l_density, t, t + densities[a] / maxd * col_w,
                             color=col, alpha=0.85, lw=0)

    ps = res.p_sample
    cross = np.where(np.diff(np.sign(ps - 0.5)) != 0)[0]
    for lam in res.l_grid[cross]:
        ax.plot([0.6, MAX_T + col_w + 0.2], [lam, lam],
                color="0.2", ls="--", lw=1.0)
    ax.axhline(0, color="0.9", lw=0.6, zorder=0)
    ax.set_xlim(0.6, MAX_T + col_w + 0.3)
    ax.set_ylim(YMIN_F, YMAX_F)
    ax.set_xticks(range(1, MAX_T + 1))
    ax.set_xlabel("Time step", fontsize=AXLAB)
    ax.set_ylabel("Cumulative evidence\n(Log likelihood ratio)", fontsize=AXLAB)
    ax.tick_params(labelsize=TICKLAB)
    ax.set_title("Simulated trials (50k)", fontsize=10, loc="left", pad=3)

    if show_legend:
        leg = [Line2D([0], [0], marker="s", ms=7, ls="none", mfc=PASTEL["H0"],
                      mec="none", label=r"chosen $H_0$"),
               Line2D([0], [0], marker="s", ms=7, ls="none", mfc=PASTEL["H1"],
                      mec="none", label=r"chosen $H_1$"),
               Line2D([0], [0], marker="s", ms=7, ls="none", mfc=PASTEL["S"],
                      mec="none", label="sampling")]
        ax.legend(handles=leg, frameon=False, fontsize=8, ncol=3,
                  loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  handletextpad=0.3, columnspacing=1.0)


# --------------------------------------------------------------------------
def add_label(ax, letter, x_off=-40, y_off=8):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(x_off, y_off), textcoords="offset points",
                fontsize=13, fontweight="bold", va="bottom",
                annotation_clip=False)


def main():
    rdet, ddet, rsoft, dsoft = solve()

    fig = plt.figure(figsize=(16.5, 12.0))
    gs = GridSpec(9, 9, figure=fig, left=0.045, right=0.99,
                  top=0.965, bottom=0.045, wspace=1.1, hspace=1.4)

    # ---------------- Row band 0 : Panel A + Panel B + equations ----------
    gsA = gs[0:3, 0:5].subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.35)
    axA_sch = fig.add_subplot(gsA[0, 0])
    axA_dist = fig.add_subplot(gsA[0, 1])
    panel_A(axA_sch, axA_dist, rdet)
    add_label(axA_sch, "A")

    axB_v = fig.add_subplot(gs[0:3, 5:7])
    axB_eq = fig.add_subplot(gs[0:3, 7:9])
    panel_B(axB_v, axB_eq, rdet)
    add_label(axB_v, "B")

    # ---------------- Row band 1 : deterministic  (C, D, E) ---------------
    gC = gs[3:6, 0:2].subgridspec(2, 1, hspace=0.18)
    axC_top = fig.add_subplot(gC[0])
    axC_bot = fig.add_subplot(gC[1], sharex=axC_top)
    panel_policy_mini(axC_top, axC_bot, rdet, show_legend=True)
    add_label(axC_top, "C")

    axD = fig.add_subplot(gs[3:6, 2:5])
    panel_field(axD, rdet, mode="hard", show_legend=True)
    add_label(axD, "D")

    axE = fig.add_subplot(gs[3:6, 5:9])
    panel_sim_hist(axE, ddet, rdet, show_legend=True)
    add_label(axE, "E")

    # ---------------- Row band 2 : soft  (F, G, H) -------------------------
    gF = gs[6:9, 0:2].subgridspec(2, 1, hspace=0.18)
    axF_top = fig.add_subplot(gF[0])
    axF_bot = fig.add_subplot(gF[1], sharex=axF_top)
    panel_policy_mini(axF_top, axF_bot, rsoft, show_legend=False)
    add_label(axF_top, "F")

    axG = fig.add_subplot(gs[6:9, 2:5])
    panel_field(axG, rsoft, mode="soft", show_legend=False)
    add_label(axG, "G")

    axH = fig.add_subplot(gs[6:9, 5:9])
    panel_sim_kde(axH, dsoft, rsoft, show_legend=False)
    add_label(axH, "H")

    # Row-band headers
    fig.text(0.01, 0.655, "Deterministic policy", fontsize=12,
              fontweight="bold", va="bottom",
              color="0.15")
    fig.text(0.16, 0.655, r"($\tau=0$ execution noise)", fontsize=9.5,
              va="bottom", color="0.4")
    fig.text(0.01, 0.335, "Soft policy", fontsize=12,
              fontweight="bold", va="bottom", color="0.15")
    fig.text(0.10, 0.335, r"($\tau=0.04$ execution noise)", fontsize=9.5,
              va="bottom", color="0.4")

    outdir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(outdir, "fig1_new.svg"))
    fig.savefig(os.path.join(outdir, "fig1_new.png"), dpi=200)
    print("Saved. det L=(%.3f, %.3f); soft p_sample in [%.3f, %.3f]" % (
        rdet.lower_l, rdet.upper_l, rsoft.p_sample.min(), rsoft.p_sample.max()))


if __name__ == "__main__":
    main()
