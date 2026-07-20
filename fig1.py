#!/usr/bin/env python3
"""
Figure 1 -- Classic sequential-sampling task and HARD-boundary optimal
solutions (no representational noise, no execution noise: sigma_repr=0, tau=0).

Ideal observer via noise='representation', sigma_repr=0 -> exact step policies.

Layout: Panel A across the top (wide schematic | narrow cue-frequency bars);
then two condition blocks side by side, each = value panel (left) + stacked
policy p(sample)/p(H0),p(H1) (right), with a simulation panel beneath.
Block headers: "Time Unconstrained" (B/C) and "Time Constrained" (E/F).

Color = quantity, graded dark->light over time:
    green = p(sample)/Q_W,  red = p(H0)/Q0,  blue = p(H1)/Q1.

Histogram bins are CENTERED on the 0.1 evidence lattice so every optimal
boundary (a non-lattice real) falls in a between-bin valley -> clean-cut
separation of chosen vs. waiting trials in Panels D and G.

Outputs: fig1.svg (Illustrator-editable) and fig1.png (preview).
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

import noise_model_runner as nmr

plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "Arial", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

_FAMILY = {
    "H0": ((0.00, 0.00, 0.50), (0.72, 0.78, 1.00)),
    "H1": ((0.50, 0.00, 0.00), (1.00, 0.72, 0.72)),
    "S":  ((0.00, 0.35, 0.00), (0.62, 0.90, 0.62)),
}
def fcol(key, frac=0.0):
    d, l = _FAMILY[key]; frac = float(np.clip(frac, 0, 1))
    return tuple(a + frac * (b - a) for a, b in zip(d, l))

PALE_H0 = (0.92, 0.55, 0.55)     # pale red  (mean lines in D/G)
PALE_H1 = (0.55, 0.62, 0.92)     # pale blue

COST = 0.04
NTRIAL = 50_000
MAX_T = 10
DEADLINE = 11
BW = 0.1
YMIN, YMAX = -1.6, 1.6


def solve():
    cfg = dict(c=COST, sigma_repr=0.0, tau_exec=0.0, n_trials=NTRIAL,
               max_timestep=MAX_T, deadline=DEADLINE, n_gh=81)
    ri = nmr.compute_model(noise="representation", horizon="infinite",
                           config=nmr.ModelConfig(**cfg), verbose=False)
    rf = nmr.compute_model(noise="representation", horizon="finite",
                           config=nmr.ModelConfig(**cfg), verbose=False)
    return ri, nmr.simulate_trials(ri, verbose=False), rf, nmr.simulate_trials(rf, verbose=False)


# --------------------------------------------------------------------------
# Panel A
# --------------------------------------------------------------------------
def panel_A(ax_sch, ax_dist, res):
    ax_sch.set_xlim(0, 11.6); ax_sch.set_ylim(-1.15, 4.1); ax_sch.axis("off")
    glyphs = ["\u25CF", "\u25B2", "\u2716", "\u2605", ""]
    labels = ["Fixation", "Cue 1", "Cue 2", "Cue 3", "..."]
    w, gap = 1.75, 0.55
    x = 0.2; centers = []
    for lab, gl in zip(labels, glyphs):
        ax_sch.add_patch(Rectangle((x, 1.55), w, 1.4, fill=False, lw=0.9, ec="0.2"))
        ax_sch.text(x + w / 2, 2.25, gl, ha="center", va="center", fontsize=12)
        ax_sch.text(x + w / 2, 1.40, lab, ha="center", va="top", fontsize=6.8)
        cx = x + w / 2; centers.append(cx)
        # choice arrows BELOW the screen: red left = H0, blue right = H1
        if gl:
            yb = 0.75
            ax_sch.add_patch(FancyArrowPatch((cx - 0.15, yb), (cx - 0.85, yb),
                             arrowstyle="-|>", mutation_scale=8, lw=1.6,
                             color=fcol("H0"), clip_on=False))
            ax_sch.add_patch(FancyArrowPatch((cx + 0.15, yb), (cx + 0.85, yb),
                             arrowstyle="-|>", mutation_scale=8, lw=1.6,
                             color=fcol("H1"), clip_on=False))
        x += w + gap
    # label the choice arrows once (under the first cue box)
    ax_sch.text(centers[1] - 0.85, 0.25, r"choose $H_0$", color=fcol("H0"),
                fontsize=6.8, ha="center", va="top")
    ax_sch.text(centers[1] + 0.85, 0.25, r"choose $H_1$", color=fcol("H1"),
                fontsize=6.8, ha="center", va="top")
    # green dashed sampling arcs ABOVE the boxes (apex fully visible)
    for i in range(len(centers) - 2):
        ax_sch.add_patch(FancyArrowPatch((centers[i] + 0.25, 3.05),
                         (centers[i + 1] - 0.25, 3.05),
                         connectionstyle="arc3,rad=0.5", arrowstyle="-|>",
                         mutation_scale=9, lw=1.3, ls="--", color=fcol("S"),
                         clip_on=False))
    ax_sch.text(np.mean(centers[:3]), 4.02, r"sample ($-c$ per step)",
                ha="center", fontsize=7.2, color=fcol("S"))
    ax_sch.text(centers[-1] + 0.1, 2.25, "choice:\n+1 / 0", ha="center",
                va="center", fontsize=6.8, color="0.15")
    ax_sch.set_title("Sequential sampling task", fontsize=8.5, loc="left")

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
        ax_dist.plot(xv, ymark, marker=mk, ms=3.4, color="0.25", clip_on=False, ls="none")
    ax_dist.set_xticks(ev)
    #ax_dist.set_xticklabels([f"{v:+.1f}" for v in ev], fontsize=7)
    ax_dist.set_xticklabels([f"{v:+.1f}" if i in [0, 2, 4, 6, 9, 11, 13, 15, 17] else ""
                             for i, v in enumerate(ev)], fontsize=7)
    ax_dist.tick_params(axis="x", which="major", pad=10)
    ax_dist.set_xlabel(r"Cue $\log_{10}$ odds", fontsize=8)
    ax_dist.xaxis.set_label_coords(0.5, -0.22)
    ax_dist.set_ylabel("Frequency", fontsize=8)
    ax_dist.set_ylim(ymark * 1.6, max(pa.max(), pb.max()) * 1.15)
    ax_dist.legend(
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    ax_dist.set_title("Cue frequencies", fontsize=8.5, loc="left")


# --------------------------------------------------------------------------
# Value panel  (Panels B, E)
# --------------------------------------------------------------------------
def panel_value(ax, res):
    b = res.b_grid
    ax.plot(b, res.q0, color=fcol("H0"), lw=2.0, label=r"$Q_0$ (choose $H_0$)")
    ax.plot(b, res.q1, color=fcol("H1"), lw=2.0, label=r"$Q_1$ (choose $H_1$)")
    if res.horizon == "infinite":
        ax.plot(b, res.q_wait, color=fcol("S"), lw=2.0, label=r"$Q_W$ (wait)")
        betas = (res.lower_b, res.upper_b); ylo = -0.03
        leg = [Line2D([0], [0], color=fcol("H0"), lw=2, label=r"$Q_0$ (choose $H_0$)"),
               Line2D([0], [0], color=fcol("H1"), lw=2, label=r"$Q_1$ (choose $H_1$)"),
               Line2D([0], [0], color=fcol("S"), lw=2, label=r"$Q_W$ (wait)")]
    else:
        for t in range(MAX_T, 0, -1):
            ax.plot(b, res.q_wait[t], color=fcol("S", (t - 1) / (MAX_T - 1)),
                    lw=1.3, zorder=3 + (MAX_T - t))
        betas = (res.lower_b[1], res.upper_b[1]); ylo = -0.09   # show Q_W(t=10) = -c
        # downward arrow: Q_W(t=1) -> Q_W(t=10) at b=0.5
        y1 = float(res.q_wait[1][np.argmin(abs(b - 0.5))])
        ax.annotate("", xy=(0.5, -COST + 0.01), xytext=(0.5, y1 - 0.03),
                    arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.3))
        ax.text(0.53, 0.5 * (y1 - COST), r"$t{:}1\!\to\!10$", fontsize=6, color="0.35")
        leg = [Line2D([0], [0], color=fcol("H0"), lw=2, label=r"$Q_0$ (choose $H_0$)"),
               Line2D([0], [0], color=fcol("H1"), lw=2, label=r"$Q_1$ (choose $H_1$)"),
               Line2D([0], [0], color=fcol("S", 0.0), lw=2, label=r"$Q_W$ ($t{=}1$)"),
               Line2D([0], [0], color=fcol("S", 1.0), lw=2, label=r"$Q_W$ ($t{=}10$)")]
    for bb in betas:
        ax.axvline(bb, color="0.75", ls="--", lw=1.0, zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(ylo, 1.05)
    ax.set_xlabel(r"Belief $b = P(H_1)$"); ax.set_ylabel("Value")
    ax.legend(handles=leg, frameon=False, fontsize=6.2, loc="lower center",
              ncol=1, handlelength=1.3, handletextpad=0.4)


# --------------------------------------------------------------------------
# Policy panels  (C1/C2, F1/F2)
# --------------------------------------------------------------------------
def panel_policy(ax_top, ax_bot, res):
    b = res.b_grid
    if res.horizon == "infinite":
        ax_top.plot(b, res.p_sample, color=fcol("S"), lw=1.9)
        ax_bot.plot(b, res.p_choose0, color=fcol("H0"), lw=1.9)
        ax_bot.plot(b, res.p_choose1, color=fcol("H1"), lw=1.9)
        betas = (res.lower_b, res.upper_b)
        # combined legend (all three families) on the top subpanel
        leg = [Line2D([0], [0], color=fcol("S"), lw=2, label=r"$p(\mathrm{sample})$"),
               Line2D([0], [0], color=fcol("H0"), lw=2, label=r"$p(H_0)$"),
               Line2D([0], [0], color=fcol("H1"), lw=2, label=r"$p(H_1)$")]
        ax_top.legend(handles=leg, frameon=False, fontsize=5.6, loc="center left",
                      handlelength=1.1, handletextpad=0.35, labelspacing=0.25)
    else:
        for t in range(MAX_T, 0, -1):
            fr = (t - 1) / (MAX_T - 1); z = 3 + (MAX_T - t)
            ax_top.plot(b, res.p_sample[t], color=fcol("S", fr), lw=1.3, zorder=z)
            ax_bot.plot(b, res.p_choose0[t], color=fcol("H0", fr), lw=1.3, zorder=z)
            ax_bot.plot(b, res.p_choose1[t], color=fcol("H1", fr), lw=1.3, zorder=z)
        betas = (res.lower_b[1], res.upper_b[1])
        # inward arrows on top of p(sample) panel: t=1 -> t=10 narrowing
        lo1, hi1 = float(res.lower_b[1]), float(res.upper_b[1])
        ax_top.annotate("", xy=(0.5 - 0.06, 0.9), xytext=(lo1, 0.9),
                        arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.1))
        ax_top.annotate("", xy=(0.5 + 0.06, 0.9), xytext=(hi1, 0.9),
                        arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.1))
        ax_top.text(0.5, 0.72, r"$t{:}1\!\to\!10$", ha="center", fontsize=5.6, color="0.4")

    ax_top.set_ylabel(r"$p(\mathrm{sample})$", fontsize=7)
    ax_bot.set_ylabel("Choice prob.", fontsize=7)
    ax_bot.set_xlabel(r"Belief $b = P(H_1)$")
    for ax in (ax_top, ax_bot):
        ax.set_xlim(0, 1); ax.set_ylim(-0.06, 1.12); ax.set_yticks([0, 1])
        for bb in betas:
            ax.axvline(bb, color="0.75", ls="--", lw=1.0, zorder=0)
    plt.setp(ax_top.get_xticklabels(), visible=False)


# --------------------------------------------------------------------------
# Simulation panels  (D, G)
# --------------------------------------------------------------------------
def panel_sim(ax, df, res, horizon, col_w=0.85, add_legend=False):
    # bins centered on the 0.1 evidence lattice (edges at +-0.05, +-0.15, ...)
    edges = np.arange(YMIN - 0.05, YMAX + 0.05 + 1e-9, BW)
    yc = edges[:-1] + BW / 2.0
    for t in range(1, MAX_T + 1):
        dft = df[df["time_step"] == t]
        hist, maxc = {}, 0.0
        for a in (0, 1, 2):
            y = dft.loc[dft["action"] == a, "ground_L"].to_numpy()
            h, _ = np.histogram(y, bins=edges); hist[a] = h.astype(float)
            maxc = max(maxc, h.max())
        if maxc <= 0:
            continue
        for a, col in ((2, fcol("S")), (0, fcol("H0")), (1, fcol("H1"))):
            ax.fill_betweenx(yc, t, t + hist[a] / maxc * col_w, step="mid",
                             color=col, alpha=0.5, lw=0)
        for a, col in ((0, PALE_H0), (1, PALE_H1)):
            y = dft.loc[dft["action"] == a, "ground_L"].to_numpy()
            if y.size > 5:
                ax.plot([t, t + col_w], [y.mean(), y.mean()], color=col, lw=1.6)

    if horizon == "infinite":
        for L in (res.lower_l, res.upper_l):
            ax.plot([0.6, MAX_T + col_w + 0.2], [L, L], color="0.2", ls="--", lw=1.1)
    else:
        ts = np.arange(1, MAX_T + 1)
        ax.plot(ts, res.lower_l[1:MAX_T + 1], color="0.2", ls="--", lw=1.2, marker="o", ms=2.5)
        ax.plot(ts, res.upper_l[1:MAX_T + 1], color="0.2", ls="--", lw=1.2, marker="o", ms=2.5)
    ax.axhline(0, color="0.9", lw=0.6, zorder=0)
    ax.set_xlim(0.6, MAX_T + col_w + 0.3); ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(range(1, MAX_T + 1)); ax.set_xlabel("Time step")
    ax.set_ylabel(r"Cumulative evidence ($\log_{10}$ odds)")

    if add_legend:
        leg = [Line2D([0], [0], marker="s", ms=7, ls="none", mfc=fcol("H0"),
                      mec="none", alpha=0.6, label=r"chosen $H_0$"),
               Line2D([0], [0], marker="s", ms=7, ls="none", mfc=fcol("H1"),
                      mec="none", alpha=0.6, label=r"chosen $H_1$"),
               Line2D([0], [0], marker="s", ms=7, ls="none", mfc=fcol("S"),
                      mec="none", alpha=0.6, label="sampling")]
        ax.legend(handles=leg, frameon=False, fontsize=6.5, ncol=3,
                  loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  handletextpad=0.3, columnspacing=1.0)


# --------------------------------------------------------------------------
def main():
    ri, di, rf, dfn = solve()

    fig = plt.figure(figsize=(11.6, 9.3))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[0.95, 4.6], hspace=0.30,
                     left=0.055, right=0.985, top=0.965, bottom=0.075)

    # Panel A (wide schematic, narrow cue bars)
    gsA = outer[0].subgridspec(1, 2, width_ratios=[2.15, 0.9], wspace=0.22)
    axA_sch = fig.add_subplot(gsA[0, 0]); axA_dist = fig.add_subplot(gsA[0, 1])
    panel_A(axA_sch, axA_dist, ri)
    axA_sch.annotate("A", xy=(0, 1), xycoords="axes fraction",
                     xytext=(-30, 4), textcoords="offset points",
                     fontsize=11, fontweight="bold", va="bottom")

    blocks = outer[1].subgridspec(1, 2, wspace=0.26)

    def build_block(cell, res, df, letters, header, add_dleg):
        gblk = cell.subgridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.44)
        gtop = gblk[0].subgridspec(2, 2, width_ratios=[1.15, 1.0],
                                   height_ratios=[1, 1], wspace=0.44, hspace=0.16)
        ax_v = fig.add_subplot(gtop[:, 0])
        ax_c1 = fig.add_subplot(gtop[0, 1]); ax_c2 = fig.add_subplot(gtop[1, 1], sharex=ax_c1)
        ax_sim = fig.add_subplot(gblk[1])
        panel_value(ax_v, res)
        panel_policy(ax_c1, ax_c2, res)
        panel_sim(ax_sim, df, res, res.horizon, add_legend=add_dleg)
        for ax, L in ((ax_v, letters[0]), (ax_c1, letters[1]), (ax_sim, letters[2])):
            ax.annotate(L, xy=(0, 1), xycoords="axes fraction",
                        xytext=(-34, 8), textcoords="offset points",
                        fontsize=11, fontweight="bold", va="bottom")
        # block header spanning value+policy row
        pos_v = ax_v.get_position(); pos_c = ax_c1.get_position()
        xmid = 0.5 * (pos_v.x0 + pos_c.x1)
        fig.text(xmid, pos_c.y1 + 0.028, header, ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    build_block(blocks[0, 0], ri, di, ("B", "C", "D"), "Time Unconstrained", True)
    build_block(blocks[0, 1], rf, dfn, ("E", "F", "G"), "Time Constrained", True)

    outdir = "C:/Users/Bo/NYU Langone Health Dropbox/Jia He/Bo Shen/RNN SPRT/Figs_v2.3"
    fig.savefig(os.path.join(outdir, "fig1.svg"))
    fig.savefig(os.path.join(outdir, "fig1.png"), dpi=200)
    print("Saved fig1.svg / fig1.png ; inf boundaries L=("
          f"{ri.lower_l:.3f},{ri.upper_l:.3f})")


if __name__ == "__main__":
    main()
