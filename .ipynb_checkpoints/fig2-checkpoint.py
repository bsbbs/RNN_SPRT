#!/usr/bin/env python3
"""
Figure 2 -- Optimal policy softens under a cost of control.

Notation:  lambda  = entropy/KL regularizer (cost of control weight);
           tau     = softmax temperature of an example control level.
The optimal policy is softmax at tau = lambda (peak of the regularized gain).

lambda is CHOSEN: lambda ~ 0.04.

Row 0:  A (control tradeoff, at the decision boundary) | C (policy) | B (simplex)
Row 1:  D (soft-boundary AREA, grayscale, belief axis) | E (simulation, belief axis)

Panel A is evaluated at the DECISION BOUNDARY (where committing and sampling
have equal value): at b=0.5 waiting dominates, so with a small lambda the
optimal there is near-deterministic and the tradeoff degenerates.
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, "C:/Users/Bo/NYU Langone Health Dropbox/Jia He/Bo Shen/RNN SPRT/Figs_v2.3")
sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.stats import gaussian_kde
from scipy.optimize import brentq
import noise_model_runner as nmr

plt.rcParams.update({"svg.fonttype": "none", "font.family": "Arial", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

GRAY_C = (0.62, 0.62, 0.62)
DARK = {"S": (0.0, 0.32, 0.0), "H0": (0.55, 0.0, 0.0), "H1": (0.0, 0.0, 0.55)}
GRAYFRAC = [1.0, 0.80, 0.60, 0.40, 0.20, 0.0]
def fam_level(fam, k):
    g = GRAYFRAC[k]; d = DARK[fam]
    return tuple((1 - g) * di + g * gi for di, gi in zip(d, GRAY_C))
def level_color(k): return fam_level("S", k)
RED_RING = "#E4322B"
B_CMAP, B_NORM = "RdBu", TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)
TRI_H = np.sqrt(3) / 2
COST, NTRIAL = 0.04, 50_000
OPT = 2            # index of the optimal level (= lambda) in the 6-level list
DOT_S = 44


def belief(L): return 1.0 / (1.0 + 10.0 ** (-L))
def simplex_xy(p0, p1, ps): return np.stack([p1 + 0.5 * ps, TRI_H * ps], axis=-1)
def draw_simplex(ax):
    ax.add_patch(Polygon([[0, 0], [0.5, TRI_H], [1, 0]], closed=True, facecolor="1", edgecolor="none", zorder=-2))
    ax.plot([0, 0.5, 1, 0], [0, TRI_H, 0, 0], color="k", lw=1.0, zorder=4)
    ax.text(0.5, TRI_H + 0.045, r"$p_S$", ha="center", va="bottom", fontsize=8)
    ax.text(-0.02, -0.05, r"$p_0$", ha="left", va="top", fontsize=8)
    ax.text(1.02, -0.05, r"$p_1$", ha="right", va="top", fontsize=8)
    ax.set_xlim(-0.16, 1.16); ax.set_ylim(-0.12, TRI_H + 0.12)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
def simplex_curve(ax, p0, p1, ps, b, lw=2.6, z=3):
    xy = simplex_xy(np.asarray(p0), np.asarray(p1), np.asarray(ps))
    segs = np.stack([xy[:-1], xy[1:]], 1)
    lc = LineCollection(segs, cmap=plt.get_cmap(B_CMAP), norm=B_NORM, lw=lw, zorder=z)
    lc.set_array(0.5 * (b[:-1] + b[1:])); ax.add_collection(lc); return lc
def simplex_pt(ax, p0, p1, ps, color, s=DOT_S, red_ring=False):
    xy = simplex_xy(np.array([p0]), np.array([p1]), np.array([ps]))[0]
    ax.scatter([xy[0]], [xy[1]], s=s + 55, facecolors="white", edgecolors="white", zorder=20, lw=0)
    ax.scatter([xy[0]], [xy[1]], s=s, facecolors=[color], edgecolors="white", lw=0.8, zorder=21)
    if red_ring:
        ax.scatter([xy[0]], [xy[1]], s=s + 45, facecolors="none", edgecolors=RED_RING, lw=1.2, zorder=22)


def softmax_rows(Q0, Q1, QW, T):
    Z = np.vstack([Q0, Q1, QW]) / T; Z -= Z.max(0, keepdims=True)
    e = np.exp(Z); e /= e.sum(0, keepdims=True); return e[0], e[1], e[2]


def build():
    ri = nmr.compute_model(noise="representation", horizon="infinite",
                           config=nmr.ModelConfig(c=COST, sigma_repr=0.0, tau_exec=0.0, n_gh=81), verbose=False)
    b = ri.b_grid; Q0, Q1, QW = ri.q0, ri.q1, ri.q_wait
    
    LAM = 0.12 # for panel A
    
    # Panel A evaluated at the zero evidence (b -> 0.5)
    i_A = int(np.argmin(np.abs(b - 0.5)))   # index of the belief closest to 0
    Qv = np.array([Q0[i_A], Q1[i_A], QW[i_A]]); n = 3
    Qmean, Qmax = Qv.mean(), Qv.max()
    def dkl(T):
        p = np.array(softmax_rows(*[np.array([q]) for q in Qv], T)).ravel(); p = np.clip(p, 1e-12, 1)
        return float((p * np.log(p * n)).sum())
    def Gof(T):
        p = np.array(softmax_rows(*[np.array([q]) for q in Qv], T)).ravel(); return float(p @ Qv)

    Ts = np.r_[np.inf, np.logspace(np.log10(4.0), np.log10(1e-5), 900)]
    xs = np.r_[0.0, LAM * np.array([dkl(T) for T in Ts[1:]])]
    Gs = np.r_[Qmean, np.array([Gof(T) for T in Ts[1:]])]
    o = np.argsort(xs); xs, Gs = xs[o], Gs[o]; Js = Gs - xs
    xmax = LAM * np.log(n); x_peak = LAM * dkl(LAM); J_peak = Gof(LAM) - x_peak

    LEVEL_T = [np.inf, 0.7, LAM, .04, 0.023, 1e-3]
    # x/G/J for the 5 example levels (argmax = hard policy: D_KL = log n, G = Qmax)
    lx, lG, lJ = [], [], []
    for k, T in enumerate(LEVEL_T):
        if k == 0: lx.append(0.0); lG.append(Qmean)
        elif k == 5: lx.append(xmax); lG.append(Qmax)
        else: lx.append(LAM * dkl(T)); lG.append(Gof(T))
        lJ.append(lG[-1] - lx[-1])

    def policy(k):
        if k == 0: return (np.full_like(b, 1/3),) * 3
        if k == 5: return ri.p_choose0, ri.p_choose1, ri.p_sample
        return softmax_rows(Q0, Q1, QW, LEVEL_T[k])

    labels = [r"$\infty$ (random)", r"$\tau=0.7$",
              rf"$\tau={LAM:.2f}$", r"$\tau=0.04$", r"$\tau=0.023$", r"$0$ (deterministic)"]
    return dict(ri=ri, b=b, LAM1=LAM, LAM2=LEVEL_T[3], b_A=b[i_A], Qmean=Qmean, Qmax=Qmax,
                xs=xs, Gs=Gs, Js=Js, xmax=xmax, x_peak=x_peak, J_peak=J_peak,
                level_x=lx, level_G=lG, level_J=lJ, policy=policy, labels=labels)


def panel_A(ax_up, ax_lo, S):
    ax_up.plot(S["xs"], S["Gs"], color="0.2", lw=1.9, solid_capstyle="round")
    ax_lo.plot(S["xs"], S["Js"], color="0.2", lw=1.9, solid_capstyle="round")
    for k in range(6):
        c = level_color(k)
        for ax, yv in ((ax_up, S["level_G"][k]), (ax_lo, S["level_J"][k])):
            ax.scatter([S["level_x"][k]], [yv], s=DOT_S, color=c, edgecolors="white", lw=0.8, zorder=6)
            if k == OPT:
                ax.scatter([S["level_x"][k]], [yv], s=DOT_S + 45, facecolors="none",
                           edgecolors=RED_RING, lw=1.2, zorder=7)
    for ax in (ax_up, ax_lo):
        ax.axvline(S["x_peak"], color="0.5", ls="--", lw=0.9, zorder=1); ax.set_xlim(-0.02 * S["xmax"], S["xmax"] * 1.02)
    ax_up.axhline(S["Qmax"], color="0.5", ls="--", lw=0.9, zorder=1)
    ax_lo.axhline(S["J_peak"], color="0.5", ls="--", lw=0.9, zorder=1)
    # "tau = lambda" arrow at the peak (lower sub-panel)
    ax_lo.annotate(r"$\tau=\lambda$", xy=(S["x_peak"], S["J_peak"]),
                   xytext=(S["x_peak"] + 0.16 * S["xmax"], S["J_peak"] - 0.55 * (S["J_peak"] - min(S["Js"]))),
                   fontsize=9, color=RED_RING,
                   arrowprops=dict(arrowstyle="-|>", color=RED_RING, lw=1.0))
    ax_up.set_yticks([S["Qmean"], S["Qmax"]]); ax_up.set_yticklabels([r"$\overline{Q_a}$", r"$Q^{\max}$"], fontsize=7.5)
    ax_up.set_ylabel(r"Expected gain"
                     "\n"
                     r"$EV(\pi)=\sum_a\pi_a Q_a$",fontsize=9); ax_up.set_ylim(S["Qmean"] - 0.015, S["Qmax"] + 0.015)
    plt.setp(ax_up.get_xticklabels(), visible=False)
    ax_lo.set_yticks([S["Qmean"], S["J_peak"]]); ax_lo.set_yticklabels([r"$\overline{Q_a}$", r"$J^{*}$"], fontsize=7.5)
    ax_lo.set_ylabel(r"Regularized gain"
                     "\n"
                     r"$J(\pi)=\sum_a\pi_a Q_a-\lambda D_{\mathrm{KL}}(\pi\Vert\pi^{rand})$", fontsize=9); ax_lo.set_ylim(min(S["Js"]) - 0.01, S["J_peak"] + 0.01)
    ax_lo.set_xticks([0.0, S["xmax"]]); ax_lo.set_xticklabels(["0\n(random)", "Max\n(deterministic)"], fontsize=7.5)
    ax_lo.set_xlabel(
        r"Cost of control: "
        "\n"
        r"$C(\pi)=\lambda D_{\mathrm{KL}}(\pi\Vert\pi^{rand})$",
        fontsize=9,
    )
    # ax_lo.set_xlabel(r"Cost of control  $\lambda\,D_{KL}$", fontsize=7.5)


def panel_B(ax_top, ax_bot, S):
    b = S["b"]
    for k in range(5, -1, -1):
        p0, p1, ps = S["policy"](k)
        if k == 0:
            ax_top.axhline(1/3, color=GRAY_C, lw=1.4); ax_bot.axhline(1/3, color=GRAY_C, lw=1.4); continue
        lwk = 1.4
        ax_top.plot(b, ps, color=fam_level("S", k), lw=lwk, zorder=3 + k)
        ax_bot.plot(b, p0, color=fam_level("H0", k), lw=lwk, zorder=3 + k)
        ax_bot.plot(b, p1, color=fam_level("H1", k), lw=lwk, zorder=3 + k)
    ax_top.set_ylabel(r"$p(\mathrm{sample})$", fontsize=9); ax_bot.set_ylabel(r"$p(\mathrm{choose})$", fontsize=9)
    ax_bot.set_xlabel(r"Posterior belief $b$", fontsize=9)
    for ax in (ax_top, ax_bot):
        ax.set_xlim(0, 1); ax.set_ylim(-0.04, 1.05); ax.set_yticks([0, 0.5, 1]); ax.axvline(0.5, color="0.85", ls=":", lw=0.7)
    plt.setp(ax_top.get_xticklabels(), visible=False)


def panel_C(ax, S):
    draw_simplex(ax); b = S["b"]; sub = slice(0, len(b), 6); lc = None
    for k in range(5, -1, -1):
        p0, p1, ps = S["policy"](k)
        if k == 0:
            simplex_pt(ax, 1/3, 1/3, 1/3, level_color(0)); continue
        lc = simplex_curve(ax, p0[sub], p1[sub], ps[sub], b[sub], lw=2.6, z=3 + k)
        simplex_pt(ax, p0[0], p1[0], ps[0], fam_level("H0", k)) # , red_ring=(k == OPT)
        simplex_pt(ax, p0[-1], p1[-1], ps[-1], fam_level("H1", k)) # , red_ring=(k == OPT)
    order = [5, 4, 3, 2, 1, 0]                    # 0 (determ.) on TOP ... inf (random) bottom
    handles = [Line2D([0], [0], color=level_color(k), lw=1.4, label=S["labels"][k]) for k in order]
    #ax.annotate(r"$\pi_a=\dfrac{\pi_{0,a}e^{Q_a/\tau}}{\sum_b \pi_{0,b}e^{Q_b/\tau}}$", xy=(0, 1), xycoords="axes fraction",
                #xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=9, annotation_clip=False)
    ax.text(0.5, 1.03, r"$\pi_a=\dfrac{\pi^{rand}_ae^{Q_a/\tau}}{\sum_b \pi^{rand}_be^{Q_b/\tau}}$", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9)
    ax.legend(handles=handles, title=r"Control ($\tau$)", fontsize=9, title_fontsize=9, frameon=False,
              loc="center right", bbox_to_anchor=(-0.02, 0.5), handlelength=1.1, labelspacing=0.35)
    return lc


def _boundary_beliefs(ri, LAM):
    pW = np.exp(ri.q_wait/LAM)/(np.exp(ri.q0/LAM)+np.exp(ri.q1/LAM)+np.exp(ri.q_wait/LAM))
    return belief(ri.l_grid[np.where(np.diff(np.sign(pW - 0.5)))[0]])


def panel_D(ax, S):
    ri = S["ri"]; LAM = S["LAM2"]
    pW = np.exp(ri.q_wait/LAM)/(np.exp(ri.q0/LAM)+np.exp(ri.q1/LAM)+np.exp(ri.q_wait/LAM))
    bu = np.linspace(1e-3, 1 - 1e-3, 400); Lu = np.log10(bu / (1 - bu)); pWu = np.interp(Lu, ri.l_grid, pW)
    dev = np.abs(pWu - 0.5); val = (dev / dev.max()) ** 0.6
    ax.imshow(np.repeat(val[:, None], 2, axis=1), extent=[0.5, 10.5, 0, 1], origin="lower",
              aspect="auto", cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    ax.axhline(0.5, color=(1, 1, 1, 0.4), lw=0.5)
    ax.set_ylim(0, 1); ax.set_xlim(0.5, 10.5); ax.set_xticks(range(1, 11)); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Time step", fontsize=9); ax.set_ylabel(r"Posterior belief $b$", fontsize=9)
    ax.set_title(r"Soft boundary as an area (dark $=p(\mathrm{sample}){=}0.5$)", fontsize=7.5, loc="left")


def panel_E(ax, S, col_w=0.85):
    ri = S["ri"]; LAM = .04 # S["LAM"]
    rex = nmr.compute_model(noise="execution", horizon="infinite",
                            config=nmr.ModelConfig(c=COST, sigma_repr=0.0, tau_exec=LAM, n_trials=NTRIAL,
                            max_timestep=10, n_gh=81), verbose=False)
    df = nmr.simulate_trials(rex, verbose=False)
    Lg = np.linspace(-2.0, 2.0, 300)                       # y-axis = log-likelihood ratio
    for t in range(1, 11):
        dft = df[df["time_step"] == t]; dens, maxd = {}, 0.0
        for a in (0, 1, 2):
            y = dft.loc[dft["action"] == a, "ground_L"].to_numpy()
            d = gaussian_kde(y, bw_method=0.35)(Lg) * y.size if (y.size > 15 and y.std() > 1e-3) else np.zeros_like(Lg)
            dens[a] = d; maxd = max(maxd, d.max())
        if maxd <= 0: continue
        for a, col in ((2, DARK["S"]), (0, DARK["H0"]), (1, DARK["H1"])):
            ax.fill_betweenx(Lg, t, t + dens[a] / maxd * col_w, color=col, alpha=0.30, lw=0)
    pW = np.exp(ri.q_wait/LAM)/(np.exp(ri.q0/LAM)+np.exp(ri.q1/LAM)+np.exp(ri.q_wait/LAM))
    for Lc in ri.l_grid[np.where(np.diff(np.sign(pW - 0.5)))[0]]:
        ax.plot([0.6, 10 + col_w + 0.2], [Lc, Lc], color="0.2", ls="--", lw=1.0)
    ax.axhline(0, color="0.9", lw=0.5, zorder=0)
    ax.set_xlim(0.6, 10 + col_w + 0.3); ax.set_ylim(-2, 2); ax.set_xticks(range(1, 11))
    ax.set_xlabel("Time step", fontsize=9); ax.set_ylabel(r"Cumulative evidence"
                                                          "\n"
                                                          "(log likelihood ratio)", fontsize=9)
    ax.set_title(rf"Soft policy: chosen & waiting overlap ($\lambda={S['LAM2']:.2f}$)", fontsize=7.5, loc="left")


def main():
    S = build()
    fig = plt.figure(figsize=(11.6, 7.9))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.12, 1.0], hspace=0.42,
                     left=0.06, right=0.95, top=0.95, bottom=0.085)
    gtop = outer[0].subgridspec(1, 3, width_ratios=[0.92, 1.20, 1.20], wspace=0.62)
    gA = gtop[0].subgridspec(2, 1, hspace=0.1)
    axAu = fig.add_subplot(gA[0]); axAl = fig.add_subplot(gA[1], sharex=axAu); panel_A(axAu, axAl, S)
    gB = gtop[1].subgridspec(2, 1, hspace=0.14)
    axBt = fig.add_subplot(gB[0]); axBb = fig.add_subplot(gB[1], sharex=axBt); panel_B(axBt, axBb, S)
    axC = fig.add_subplot(gtop[2]); lc = panel_C(axC, S)
    gbot = outer[1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.24)
    axD = fig.add_subplot(gbot[0]); panel_D(axD, S)
    axE = fig.add_subplot(gbot[1]); panel_E(axE, S)

    if lc is not None:
        cb = fig.colorbar(lc, ax=axC, location="right", fraction=0.05, pad=0.03, aspect=18)
        cb.set_label(r"belief $b$", fontsize=9); cb.ax.tick_params(labelsize=7.5); cb.set_ticks([0, 0.5, 1])

    for ax, Lb in ((axAu, "A"), (axBt, "B"), (axC, "C"), (axD, "D"), (axE, "E")):
        ax.annotate(Lb, xy=(0, 1), xycoords="axes fraction", xytext=(-32, 6), textcoords="offset points",
                    fontsize=12, fontweight="bold", va="bottom")

    outdir = "C:/Users/Bo/NYU Langone Health Dropbox/Jia He/Bo Shen/RNN SPRT/Figs_v2.3"
    fig.savefig(os.path.join(outdir, "fig2.svg")); fig.savefig(os.path.join(outdir, "fig2.png"), dpi=200)
    print("Saved fig2 | lambda=%.4f  panelA belief b_A=%.4f  boundary beliefs=%s" % (
        S["LAM2"], S["b_A"], np.round(_boundary_beliefs(S["ri"], S["LAM2"]), 3)))


if __name__ == "__main__":
    main()
