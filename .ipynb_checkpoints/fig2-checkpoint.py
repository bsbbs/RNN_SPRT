#!/usr/bin/env python3
"""
Figure 2 -- Optimal policy softens under a cost of control.

This version is self-contained.  It computes the infinite-horizon Q values by
value iteration directly on a log10-odds grid whose spacing exactly divides all
cue increments.  Therefore every Bellman transition is an integer grid shift:
no interpolation is used in the induction of Q_S(b).

Notation
--------
tau    : softmax temperature indexing the example policies in panels A and B.
lambda : KL regularizer / cost-of-control weight used in panel C.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --------------------------------------------------------------------------
# Output directory
# --------------------------------------------------------------------------
fallback_outdir = str(Path.home())
try:
    outdir = os.path.dirname(os.path.abspath(__file__))
except (NameError, TypeError):
    outdir = fallback_outdir

    
# --------------------------------------------------------------------------
# Global style
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

GRAY_C = (0.62, 0.62, 0.62)
DARK = {
    "S": (0.0, 0.32, 0.0),
    "H0": (0.0, 0.0, 0.55),
    "H1": (0.55, 0.0, 0.0),
}
GRAYFRAC = [1.0, 0.80, 0.60, 0.40, 0.20, 0.0]
RED_RING = "#E4322B"
B_CMAP = "RdBu_r"
B_NORM = TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)
TRI_H = np.sqrt(3.0) / 2.0
DOT_S = 44

# ---------------------------- model settings ------------------------------ #
COST = 0.04
NTRIAL = 50_000
MAX_TIMESTEP = 10
SEED = 1
L_MIN, L_MAX, N_L = -4.0, 4.0, 2001
VALUE_TOL = 1e-9
VALUE_MAX_ITER = 5000
MU, SIGMA = 2.2, 5.0

# Panel C uses lambda=0.12, as in plot_kl_control_tradeoff_b05.py.
LAMBDA_A = 0.12

# Panels A/B deliberately retain the six example levels from the edited fig2.py.
LEVEL_T = [np.inf, 0.7, LAMBDA_A, 0.04, 0.023, 0.0]

LEVEL_LABELS = [
    r"$\infty$ (random)",
    r"$0.7$",
    rf"${LAMBDA_A:.2f}$",
    r"$0.04$",
    r"$0.023$",
    r"$0$ (determ.)",
]

# Map the five Panel-A points onto the existing six-level color progression.
A_COLOR_LEVELS = [0, 1, 2, 4, 5]
A_OPT_INDEX = 2


@dataclass
class ModelResult:
    evidence_log10: np.ndarray
    p_a: np.ndarray
    p_b: np.ndarray
    l_grid: np.ndarray
    b_grid: np.ndarray
    q0: np.ndarray
    q1: np.ndarray
    Q_Sait: np.ndarray
    value: np.ndarray
    p_choose0: np.ndarray
    p_choose1: np.ndarray
    p_sample: np.ndarray
    lower_l: float
    upper_l: float
    lower_b: float
    upper_b: float
    iterations: int
    residual: float


def fam_level(family, k):
    gray_weight = GRAYFRAC[k]
    dark = DARK[family]
    return tuple(
        (1.0 - gray_weight) * dark_i + gray_weight * gray_i
        for dark_i, gray_i in zip(dark, GRAY_C)
    )


def level_color(k):
    return fam_level("S", k)


def belief(log10_odds):
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(log10_odds)))


def log10_odds(b):
    b = np.clip(np.asarray(b), 1e-12, 1.0 - 1e-12)
    return np.log10(b / (1.0 - b))


def softmax_rows(q0, q1, Q_Sait, tau):
    """Stable softmax for three action-value arrays."""
    z = np.vstack([q0, q1, Q_Sait]) / tau
    z -= z.max(axis=0, keepdims=True)
    e = np.exp(z)
    e /= e.sum(axis=0, keepdims=True)
    return e[0], e[1], e[2]


def safe_kl(policy, prior):
    policy = np.asarray(policy, dtype=float)
    prior = np.asarray(prior, dtype=float)
    terms = np.zeros_like(policy)
    positive = policy > 0.0
    prior_full = np.broadcast_to(prior, policy.shape)
    terms[positive] = policy[positive] * np.log(
        policy[positive] / prior_full[positive]
    )
    return terms.sum(axis=-1)


def kl_optimal_policy(q_values, regularizer, prior):
    logits = np.log(prior) + q_values / regularizer
    logits -= np.max(logits)
    weights = np.exp(logits)
    return weights / weights.sum()


def build_stimuli():
    """Analytic cue probabilities for the equal-variance Gaussian task."""
    evidence_log10 = np.array([
        -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
         0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,
    ])
    x_targets = evidence_log10 * (SIGMA ** 2 * np.log(10.0)) / (2.0 * MU)
    p_b = np.exp(-0.5 * ((x_targets - MU) / SIGMA) ** 2)
    p_a = np.exp(-0.5 * ((x_targets + MU) / SIGMA) ** 2)
    p_b /= p_b.sum()
    p_a /= p_a.sum()
    return evidence_log10, p_a, p_b


def compute_model():
    """Compute Q_S by exact grid-shift value iteration, without interpolation."""
    evidence_log10, p_a, p_b = build_stimuli()
    l_grid = np.linspace(L_MIN, L_MAX, N_L)
    b_grid = belief(l_grid)
    d_l = l_grid[1] - l_grid[0]

    # Every cue increment is exactly an integer number of grid bins.
    shifts = np.rint(evidence_log10 / d_l).astype(int)
    reconstruction_error = np.max(np.abs(evidence_log10 - shifts * d_l))
    if reconstruction_error > 1e-12:
        raise RuntimeError(
            "Evidence increments do not align with the value-iteration grid; "
            f"maximum mismatch={reconstruction_error:.3e}."
        )

    q0 = 1.0 - b_grid
    q1 = b_grid
    v_choose = np.maximum(q0, q1)

    # index[j, i] is the grid point reached from l_grid[j] after cue i.
    next_index = np.arange(N_L)[:, None] + shifts[None, :]
    next_index = np.clip(next_index, 0, N_L - 1)

    # Predictive probability of each cue at every current belief.
    predictive = (
        (1.0 - b_grid[:, None]) * p_a[None, :]
        + b_grid[:, None] * p_b[None, :]
    )

    value = v_choose.copy()
    residual = np.inf
    Q_Sait = np.zeros_like(value)
    for iteration in range(1, VALUE_MAX_ITER + 1):
        Q_Sait = -COST + np.sum(predictive * value[next_index], axis=1)
        value_new = np.maximum(v_choose, Q_Sait)
        residual = float(np.max(np.abs(value_new - value)))
        value = value_new
        if residual < VALUE_TOL:
            break
    else:
        raise RuntimeError(
            f"Value iteration did not converge after {VALUE_MAX_ITER} iterations; "
            f"residual={residual:.3e}."
        )

    continue_region = Q_Sait > v_choose
    if not np.any(continue_region):
        raise RuntimeError("The computed policy has no continuation region.")
    lower_l = float(l_grid[continue_region][0])
    upper_l = float(l_grid[continue_region][-1])

    # Deterministic argmax policy used only for tau=0 in panels B/C.
    q_stack = np.vstack([q0, q1, Q_Sait])
    hard_action = np.argmax(q_stack, axis=0)
    p_choose0 = (hard_action == 0).astype(float)
    p_choose1 = (hard_action == 1).astype(float)
    p_sample = (hard_action == 2).astype(float)

    return ModelResult(
        evidence_log10=evidence_log10,
        p_a=p_a,
        p_b=p_b,
        l_grid=l_grid,
        b_grid=b_grid,
        q0=q0,
        q1=q1,
        Q_Sait=Q_Sait,
        value=value,
        p_choose0=p_choose0,
        p_choose1=p_choose1,
        p_sample=p_sample,
        lower_l=lower_l,
        upper_l=upper_l,
        lower_b=float(belief(lower_l)),
        upper_b=float(belief(upper_l)),
        iterations=iteration,
        residual=residual,
    )


def policy_at_level(model, k):
    """Original six example policies retained for panels B and C."""
    if k == 0:
        random = np.full_like(model.b_grid, 1.0 / 3.0)
        return random, random, random
    if k == len(LEVEL_T) - 1:
        return model.p_choose0, model.p_choose1, model.p_sample
    return softmax_rows(model.q0, model.q1, model.Q_Sait, LEVEL_T[k])


def build():
    model = compute_model()
    b = model.b_grid

    # Panel A is evaluated at zero evidence, b=0.5.
    i_a = int(np.argmin(np.abs(b - 0.5)))
    q_values = np.array([
        model.q0[i_a], model.q1[i_a], model.Q_Sait[i_a]
    ])
    prior = np.full(3, 1.0 / 3.0)
    deterministic = np.array([0.0, 0.0, 1.0])

    # At b=0.5, q0=q1, so the exact KL optimum lies on this symmetric path.
    optimal_policy = kl_optimal_policy(q_values, LAMBDA_A, prior)
    optimal_alpha = 1.5 * (optimal_policy[2] - 1.0 / 3.0)

    alpha = np.linspace(0.0, 1.0, 2001)
    policies = (
        (1.0 - alpha[:, None]) * prior[None, :]
        + alpha[:, None] * deterministic[None, :]
    )
    expected_gain = policies @ q_values
    control_cost = LAMBDA_A * safe_kl(policies, prior)
    regularized_gain = expected_gain - control_cost

    example_alpha = np.array([
        0.0,
        0.5 * optimal_alpha,
        optimal_alpha,
        0.5 * (1.0 + optimal_alpha),
        1.0,
    ])
    example_names = [
        "  Random",
        "2  Weak control",
        "  Optimal control",
        "4  Strong control",
        "  Deterministic",
    ]
    example_policies = (
        (1.0 - example_alpha[:, None]) * prior[None, :]
        + example_alpha[:, None] * deterministic[None, :]
    )
    example_cost = LAMBDA_A * safe_kl(example_policies, prior)
    example_gain = example_policies @ q_values
    example_objective = example_gain - example_cost

    optimal_cost = float(LAMBDA_A * safe_kl(optimal_policy, prior))
    optimal_gain = float(optimal_policy @ q_values)
    optimal_objective = optimal_gain - optimal_cost

    # Confirm the analytic optimum agrees with the sampled path maximum.
    numerical_peak = int(np.argmax(regularized_gain))
    if abs(alpha[numerical_peak] - optimal_alpha) >= 2.0 / (len(alpha) - 1):
        raise RuntimeError("Numerical and analytic panel-A optima do not agree.")

    return {
        "model": model,
        "b": b,
        "b_A": float(b[i_a]),
        "q_values_A": q_values,
        "prior_A": prior,
        "alpha": alpha,
        "policies_A": policies,
        "xs": control_cost,
        "Gs": expected_gain,
        "Js": regularized_gain,
        "xmax": float(control_cost[-1]),
        "Qmean": float(q_values.mean()),
        "Qmax": float(q_values.max()),
        "x_peak": optimal_cost,
        "G_peak": optimal_gain,
        "J_peak": optimal_objective,
        "optimal_policy_A": optimal_policy,
        "example_alpha": example_alpha,
        "example_names": example_names,
        "example_policies": example_policies,
        "level_x": example_cost,
        "level_G": example_gain,
        "level_J": example_objective,
    }


# ------------------------------- panel A ---------------------------------- #
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


def panel_A(ax, data):
    draw_simplex(ax)
    b = data["b"]
    sub = slice(0, len(b), 6)
    la = None
    for k in range(5, -1, -1):
        p0, p1, ps = policy_at_level(data["model"], k)
        if k == 0:
            simplex_pt(ax, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, level_color(0))
            continue
        la = simplex_curve(
            ax, p0[sub], p1[sub], ps[sub], b[sub], lw=2.6, z=3 + k,
        )
        simplex_pt(ax, p0[0], p1[0], ps[0], fam_level("H0", k))
        simplex_pt(ax, p0[-1], p1[-1], ps[-1], fam_level("H1", k))

    ax.text(
        0.5, -.03,
        r"$\pi_a \propto e^{Q_a/\tau}$", # r"{\sum_{b}\pi^{rand}_{b} e^{Q_{b}/\tau}}$"
        transform=ax.transAxes, ha="center", va="bottom", fontsize=EQ_FS,
    )
    return la


# ------------------------------- panel B ---------------------------------- #
def panel_B(ax_top, ax_bot, data):
    b = data["b"]
    for k in range(5, -1, -1):
        p0, p1, ps = policy_at_level(data["model"], k)
        if k == 0:
            ax_top.axhline(1.0 / 3.0, color=GRAY_C, lw=1.4)
            ax_bot.axhline(1.0 / 3.0, color=GRAY_C, lw=1.4)
            continue
        ax_top.plot(b, ps, color=fam_level("S", k), lw=1.4, zorder=3 + k)
        ax_bot.plot(b, p0, color=fam_level("H0", k), lw=1.4, zorder=3 + k)
        ax_bot.plot(b, p1, color=fam_level("H1", k), lw=1.4, zorder=3 + k)

    handles = [
        Line2D([0], [0], color=level_color(k), lw=1.4, label=LEVEL_LABELS[k])
        for k in [5, 4, 3, 2, 1, 0]
    ]
    ax_top.legend(
        handles=handles, title=r"Stochasticity ($\tau$)",
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
    ax_bot.set_xlabel(r"Belief ($b$)", fontsize=LBL_FS)
    for ax in (ax_top, ax_bot):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.04, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.axvline(0.5, color="0.85", ls=":", lw=0.7)
        ax.tick_params(labelsize=TICK_FS)
    plt.setp(ax_top.get_xticklabels(), visible=False)


# ------------------------------- panel C ---------------------------------- #
def panel_C(ax_up, ax_lo, data):
    ax_up.plot(data["xs"], data["Gs"], color="0.2", lw=1.9,
               solid_capstyle="round")
    ax_lo.plot(data["xs"], data["Js"], color="0.2", lw=1.9,
               solid_capstyle="round")

    for i, color_level in enumerate(A_COLOR_LEVELS):
        color = level_color(color_level)
        for ax, y_values in ((ax_up, data["level_G"]),
                             (ax_lo, data["level_J"])):
            ax.scatter(
                [data["level_x"][i]], [y_values[i]],
                s=DOT_S, color=color, edgecolors="white", lw=0.8, zorder=6,
            )
            if i == A_OPT_INDEX:
                ax.scatter(
                    [data["level_x"][i]], [y_values[i]],
                    s=DOT_S + 45, facecolors="none", edgecolors=RED_RING,
                    lw=1.2, zorder=7,
                )

    for ax in (ax_up, ax_lo):
        ax.axvline(data["x_peak"], color="0.5", ls="--", lw=0.9, zorder=1)
        ax.set_xlim(-0.025 * data["xmax"], 1.025 * data["xmax"])

    ax_up.axhline(data["Qmax"], color="0.5", ls="--", lw=0.9, zorder=1)
    ax_lo.axhline(data["J_peak"], color="0.5", ls="--", lw=0.9, zorder=1)

    # The annotation
    annotation_offsets = [
        (18, -8),
        (-22, 21),
        (8,20),
        (12, -43),
        (-124, -7),
    ]
    for i, (x, y, name, policy, offset) in enumerate(zip(
        data["level_x"], data["level_J"], data["example_names"],
        data["example_policies"], annotation_offsets,
    )):
        if i == A_OPT_INDEX:
            name += " " + r"$(\tau = \lambda)$"
        if i in [0, 2, 4]:
            policy_text = rf"$\pi=({policy[0]:.2f},\,{policy[1]:.2f},\,{policy[2]:.2f})$"
            ax_lo.annotate(
                name + "\n" + policy_text,
                xy=(x, y), xytext=offset, textcoords="offset points",
                fontsize=ANN_FS, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", linewidth=0.7, color="0.35"),
                annotation_clip=False,
            )
    ax_up.text(
        0.45, 1.0,
        r"$EV(\pi)=\sum_a\pi_a Q_a$",
        transform=ax_up.transAxes, ha="center", va="bottom", fontsize=EQ_FS)
        
    ax_up.set_yticks([data["Qmean"], data["Qmax"]])
    ax_up.set_yticklabels([r"$\overline{Q_a}$", r"$Q^{\max}$"], fontsize=TICK_FS)
    ax_up.set_ylabel("Unregularized gain", fontsize=LBL_FS)
    gain_pad = 0.08 * (data["Qmax"] - data["Qmean"])
    ax_up.set_ylim(data["Qmean"] - gain_pad, data["Qmax"] + gain_pad)
    plt.setp(ax_up.get_xticklabels(), visible=False)
    
    ax_lo.text(
        0.45, 1.05,
        r"$J(\pi)=EV(\pi)-\lambda D_{\mathrm{KL}}(\pi\Vert\pi^{rand})$",
        transform=ax_lo.transAxes, ha="center", va="bottom", fontsize=EQ_FS)
    
    ax_lo.set_yticks([data["Qmean"], data["J_peak"]])
    ax_lo.set_yticklabels([r"$\overline{Q_a}$", r"$J^{*}$"], fontsize=TICK_FS)
    ax_lo.set_ylabel("Regularized gain", fontsize=LBL_FS)
    objective_span = data["J_peak"] - float(np.min(data["Js"]))
    ax_lo.set_ylim(
        float(np.min(data["Js"])) - 0.17 * objective_span,
        data["J_peak"] + 0.18 * objective_span,
    )
    ax_lo.set_xticks([0.0, data["xmax"]])
    ax_lo.set_xticklabels(["0\n(random)", "Max\n(determ.)"], fontsize=TICK_FS)
    ax_lo.set_xlabel(
        "Cost of control:\n" + r"$\lambda D_{\mathrm{KL}}(\pi\Vert\pi^{rand})$",
        fontsize=LBL_FS,
    )


# -------------------------------- layout ---------------------------------- #
def add_panel_label(ax, label, x_offset=-30, y_offset=6):
    ax.annotate(
        label, xy=(0, 1), xycoords="axes fraction",
        xytext=(x_offset, y_offset), textcoords="offset points",
        fontsize=PANEL_FS, fontweight="bold", va="bottom",
        annotation_clip=False,
    )


def main():
    data = build()

    fig = plt.figure(figsize=(9, 8.7)) # 13.2
    outer = GridSpec(
        16, 2, figure=fig,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0] * 16,
        left=0.065, right=0.945, top=0.955, bottom=0.135,
        wspace=0.43, hspace=0.58,
    )

    # Left column: panel A above panel B.
    ax_a = fig.add_subplot(outer[0:8, 0])
    la = panel_A(ax_a, data)

    # 1 empty margin on each side, 8 for Panel B
    grid_b_width = outer[8:16, 0].subgridspec(
        1, 3,
        width_ratios=[1, 20, 1],
        wspace=0
    )
    
    grid_b = grid_b_width[0, 1].subgridspec(
        2, 1,
        hspace=0.14
    )
    
    ax_bt = fig.add_subplot(grid_b[0])
    ax_bb = fig.add_subplot(grid_b[1], sharex=ax_bt)
    
    panel_B(ax_bt, ax_bb, data)

    # Middle column: panel C centered over 1.5 row-heights.
    grid_c = outer[1:16, 1].subgridspec(2, 1, hspace=0.48)
    ax_cu = fig.add_subplot(grid_c[0])
    ax_cl = fig.add_subplot(grid_c[1], sharex=ax_cu)
    panel_C(ax_cu, ax_cl, data)

    # Belief colorbar to the Right of panel A, without shrinking its square cell.
    if la is not None:
        cax = inset_axes(
            ax_a, width="3.0%", height="60%", loc="center right",
            bbox_to_anchor=(0, 0.0, 1.0, 1.0),
            bbox_transform=ax_a.transAxes, borderpad=0,
        )
        colorbar = fig.colorbar(la, cax=cax)
        colorbar.set_label(r"Belief ($b$)", fontsize=LBL_FS, labelpad=4)
        colorbar.ax.yaxis.set_label_position("right")
        colorbar.ax.yaxis.set_ticks_position("right")
        colorbar.ax.tick_params(labelsize=TICK_FS)
        colorbar.set_ticks([0, 0.5, 1])

    add_panel_label(ax_a, "A")
    add_panel_label(ax_bt, "B")
    add_panel_label(ax_cu, "C")

    svg_path = os.path.join(outdir, "fig2.svg") 
    png_path = os.path.join(outdir, "fig2.png")
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    model = data["model"]
    print(
        "Saved", svg_path, "and", png_path, "\n"
        f"Value iterations={model.iterations}",
        f"residual={model.residual:.3e}",
        f"panel-A b={data['b_A']:.4f}",
        f"Q={np.round(data['q_values_A'], 6)}",
        f"pi*={np.round(data['optimal_policy_A'], 6)}",
        f"hard-boundary beliefs=({model.lower_b:.3f}, {model.upper_b:.3f})",
    )


if __name__ == "__main__":
    main()