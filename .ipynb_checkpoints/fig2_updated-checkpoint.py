#!/usr/bin/env python3
"""
Figure 2 -- Optimal policy softens under a cost of control.

This version is self-contained.  It computes the infinite-horizon Q values by
value iteration directly on a log10-odds grid whose spacing exactly divides all
cue increments.  Therefore every Bellman transition is an integer grid shift:
no interpolation is used in the induction of Q_W(b).

Layout
------
Left column:   C (simplex, top) and B (policy, bottom), each in a square cell.
Middle column: A (control tradeoff), centered and spanning 1.5 row-heights.
Right column:  D and E, stacked over the same 1.5-row vertical extent.

Notation
--------
lambda : KL regularizer / cost-of-control weight used in panel A.
tau    : softmax temperature indexing the example policies in panels B and C.
"""

from dataclasses import dataclass
from pathlib import Path
import sys, os
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
from scipy.stats import gaussian_kde


OUTPUT_DIR = "C:/Users/Bo/NYU Langone Health Dropbox/Jia He/Bo Shen/RNN SPRT/Figs_v2.3"
OUTPUT_STEM = "fig2_updated"

plt.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.family": "Arial",
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

# ------------------------------- styling ---------------------------------- #
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

# Panel A uses lambda=0.12, as in plot_kl_control_tradeoff_b05.py.
LAMBDA_A = 0.12

# Panels B/C deliberately retain the six example levels from the edited fig2.py.
LEVEL_T = [np.inf, 0.7, LAMBDA_A, 0.04, 0.023, 0.0]
'''
LEVEL_LABELS = [
    r"$\infty$ (random)",
    r"$\tau=0.7$",
    rf"$\tau={LAMBDA_A:.2f}$",
    r"$\tau=0.04$",
    r"$\tau=0.023$",
    r"$0$ (deterministic)",
]'''
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
    q_wait: np.ndarray
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


def softmax_rows(q0, q1, q_wait, tau):
    """Stable softmax for three action-value arrays."""
    z = np.vstack([q0, q1, q_wait]) / tau
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
    """Compute Q_W by exact grid-shift value iteration, without interpolation."""
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
    q_wait = np.zeros_like(value)
    for iteration in range(1, VALUE_MAX_ITER + 1):
        q_wait = -COST + np.sum(predictive * value[next_index], axis=1)
        value_new = np.maximum(v_choose, q_wait)
        residual = float(np.max(np.abs(value_new - value)))
        value = value_new
        if residual < VALUE_TOL:
            break
    else:
        raise RuntimeError(
            f"Value iteration did not converge after {VALUE_MAX_ITER} iterations; "
            f"residual={residual:.3e}."
        )

    continue_region = q_wait > v_choose
    if not np.any(continue_region):
        raise RuntimeError("The computed policy has no continuation region.")
    lower_l = float(l_grid[continue_region][0])
    upper_l = float(l_grid[continue_region][-1])

    # Deterministic argmax policy used only for tau=0 in panels B/C.
    q_stack = np.vstack([q0, q1, q_wait])
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
        q_wait=q_wait,
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
    return softmax_rows(model.q0, model.q1, model.q_wait, LEVEL_T[k])


def build():
    model = compute_model()
    b = model.b_grid

    # Panel A is evaluated at zero evidence, b=0.5.
    i_a = int(np.argmin(np.abs(b - 0.5)))
    q_values = np.array([
        model.q0[i_a], model.q1[i_a], model.q_wait[i_a]
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
def panel_A(ax_up, ax_lo, data):
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

    # The annotation style and pi tuples follow plot_kl_control_tradeoff_b05.py.
    annotation_offsets = [
        (18, -8),
        (-22, 21),
        (-8, 22),
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
                fontsize=10, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", linewidth=0.7, color="0.35"),
                annotation_clip=False,
            )

    ax_up.set_yticks([data["Qmean"], data["Qmax"]])
    ax_up.set_yticklabels([r"$\overline{Q_a}$", r"$Q^{\max}$"], fontsize=12)
    ax_up.set_ylabel(
        "Expected gain\n" + r"$EV(\pi)=\sum_a\pi_a Q_a$", fontsize=12,
    )
    gain_pad = 0.08 * (data["Qmax"] - data["Qmean"])
    ax_up.set_ylim(data["Qmean"] - gain_pad, data["Qmax"] + gain_pad)
    plt.setp(ax_up.get_xticklabels(), visible=False)

    ax_lo.set_yticks([data["Qmean"], data["J_peak"]])
    ax_lo.set_yticklabels([r"$\overline{Q_a}$", r"$J^{*}$"], fontsize=12)
    ax_lo.set_ylabel(
        "Regularized gain\n"
        + r"$J(\pi)=\sum_a\pi_a Q_a-\lambda D_{\mathrm{KL}}(\pi\Vert\pi^{rand})$",
        fontsize=12,
    )
    objective_span = data["J_peak"] - float(np.min(data["Js"]))
    ax_lo.set_ylim(
        float(np.min(data["Js"])) - 0.17 * objective_span,
        data["J_peak"] + 0.18 * objective_span,
    )
    ax_lo.set_xticks([0.0, data["xmax"]])
    ax_lo.set_xticklabels(["0\n(random)", "Max\n(deterministic)"], fontsize=12)
    ax_lo.set_xlabel(
        "Cost of control:\n"
        + r"$C(\pi)=\lambda D_{\mathrm{KL}}(\pi\Vert\pi^{rand})$",
        fontsize=12,
    )


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
    '''
    ax_top.legend(
        handles=handles, title=r"Control ($\tau$)",
        fontsize=10, title_fontsize=10, frameon=True,
        facecolor="white", edgecolor="none", framealpha=0.86,
        loc="upper right", ncol=1, handlelength=1.1,
        columnspacing=0.75, labelspacing=0.28, borderaxespad=0.1,
    )'''
    legend = ax_top.legend(
                handles=handles,
                title=r"Control ($\tau$)",
                fontsize=10,
                title_fontsize=10,
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=0.86,
                loc="upper left",
                bbox_to_anchor=(1.03, 1.0),
                ncol=1,
                handlelength=1.1,
                columnspacing=0.75,
                labelspacing=0.28,
                borderaxespad=0.0,
                )

    ax_top.set_ylabel(r"$p(\mathrm{sample})$", fontsize=12)
    ax_bot.set_ylabel(r"$p(\mathrm{choose})$", fontsize=12)
    ax_bot.set_xlabel(r"Belief $b = P(H_1)$", fontsize=12)
    for ax in (ax_top, ax_bot):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.04, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.axvline(0.5, color="0.85", ls=":", lw=0.7)
        ax.tick_params(labelsize=10)
    plt.setp(ax_top.get_xticklabels(), visible=False)


# ------------------------------- panel C ---------------------------------- #
def simplex_xy(p0, p1, ps):
    return np.stack([p1 + 0.5 * ps, TRI_H * ps], axis=-1)


def draw_simplex(ax):
    ax.add_patch(Polygon(
        [[0, 0], [0.5, TRI_H], [1, 0]], closed=True,
        facecolor="1", edgecolor="none", zorder=-2,
    ))
    ax.plot([0, 0.5, 1, 0], [0, TRI_H, 0, 0], color="k", lw=1.0, zorder=4)
    ax.text(0.5, TRI_H + 0.045, r"$p_S$", ha="center", va="bottom", fontsize=12)
    ax.text(-0.02, -0.05, r"$p_0$", ha="left", va="top", fontsize=12)
    ax.text(1.02, -0.05, r"$p_1$", ha="right", va="top", fontsize=12)
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
    lc = LineCollection(
        segments, cmap=plt.get_cmap(B_CMAP), norm=B_NORM, lw=lw, zorder=z,
    )
    lc.set_array(0.5 * (b[:-1] + b[1:]))
    ax.add_collection(lc)
    return lc


def simplex_pt(ax, p0, p1, ps, color, size=DOT_S):
    xy = simplex_xy(np.array([p0]), np.array([p1]), np.array([ps]))[0]
    ax.scatter([xy[0]], [xy[1]], s=size + 55, facecolors="white",
               edgecolors="white", zorder=20, lw=0)
    ax.scatter([xy[0]], [xy[1]], s=size, facecolors=[color],
               edgecolors="white", lw=0.8, zorder=21)


def panel_C(ax, data):
    draw_simplex(ax)
    b = data["b"]
    sub = slice(0, len(b), 6)
    lc = None
    for k in range(5, -1, -1):
        p0, p1, ps = policy_at_level(data["model"], k)
        if k == 0:
            simplex_pt(ax, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, level_color(0))
            continue
        lc = simplex_curve(
            ax, p0[sub], p1[sub], ps[sub], b[sub], lw=2.6, z=3 + k,
        )
        simplex_pt(ax, p0[0], p1[0], ps[0], fam_level("H0", k))
        simplex_pt(ax, p0[-1], p1[-1], ps[-1], fam_level("H1", k))

    ax.text(
        0.83, .68,
        r"$\pi_a=\dfrac{\pi^{rand}_a e^{Q_a/\tau}}"
        r"{\sum_{b}\pi^{rand}_{b} e^{Q_{b}/\tau}}$",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=12,
    )
    return lc


# ------------------------------ panels D/E -------------------------------- #
def sample_probability(model, tau):
    return softmax_rows(model.q0, model.q1, model.q_wait, tau)[2]


def boundary_beliefs(model, tau):
    p_wait = sample_probability(model, tau)
    crossing = np.where(np.diff(np.sign(p_wait - 0.5)) != 0)[0]
    return belief(model.l_grid[crossing])


def panel_D(ax, data):
    model = data["model"]
    tau = 0.04
    p_wait = sample_probability(model, tau)
    b_uniform = np.linspace(1e-3, 1.0 - 1e-3, 400)
    p_wait_uniform = np.interp(log10_odds(b_uniform), model.l_grid, p_wait)
    deviation = np.abs(p_wait_uniform - 0.5)
    gray_value = (deviation / deviation.max()) ** 0.6

    ax.imshow(
        np.repeat(gray_value[:, None], 2, axis=1),
        extent=[0.5, 10.5, 0, 1], origin="lower", aspect="auto",
        cmap="gray", vmin=0, vmax=1, interpolation="bilinear",
    )
    ax.axhline(0.5, color=(1, 1, 1, 0.4), lw=0.5)
    ax.set_ylim(0, 1)
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel(r"$b$", fontsize=12)
    ax.set_title(
        r"(dark $=p_S{=}0.5$)",
        fontsize=10, loc="left",
    )
    ax.tick_params(labelsize=10)
    plt.setp(ax.get_xticklabels(), visible=False)


def simulate_trials(model, tau, n_trials=NTRIAL, max_timestep=MAX_TIMESTEP, seed=SEED):
    """Vectorized execution-noise simulation used by panel E."""
    rng = np.random.default_rng(seed)
    true_h = rng.integers(0, 2, size=n_trials)
    ground_l = np.zeros(n_trials, dtype=float)
    active = np.ones(n_trials, dtype=bool)
    records = []

    d_l = model.l_grid[1] - model.l_grid[0]
    p0_grid, p1_grid, ps_grid = softmax_rows(
        model.q0, model.q1, model.q_wait, tau
    )

    for timestep in range(1, max_timestep + 1):
        active_idx = np.flatnonzero(active)
        if active_idx.size == 0:
            records.append({"L": np.array([]), "action": np.array([], dtype=int)})
            continue

        h_active = true_h[active_idx]
        cue_idx = np.empty(active_idx.size, dtype=int)
        mask0 = h_active == 0
        mask1 = ~mask0
        if np.any(mask0):
            cue_idx[mask0] = rng.choice(
                len(model.evidence_log10), size=mask0.sum(), p=model.p_a
            )
        if np.any(mask1):
            cue_idx[mask1] = rng.choice(
                len(model.evidence_log10), size=mask1.sum(), p=model.p_b
            )

        ground_l[active_idx] += model.evidence_log10[cue_idx]

        # Accumulated evidence also stays on the same exact grid; clip only at
        # the finite value-function domain, matching the Bellman boundary rule.
        grid_idx = np.rint((ground_l[active_idx] - L_MIN) / d_l).astype(int)
        grid_idx = np.clip(grid_idx, 0, N_L - 1)
        probs = np.column_stack([
            p0_grid[grid_idx], p1_grid[grid_idx], ps_grid[grid_idx]
        ])

        u = rng.random(active_idx.size)
        action = np.where(
            u < probs[:, 0], 0,
            np.where(u < probs[:, 0] + probs[:, 1], 1, 2),
        )
        records.append({"L": ground_l[active_idx].copy(), "action": action})
        active[active_idx[action != 2]] = False

    return records


def panel_E(ax, data, col_width=0.82):
    model = data["model"]
    tau = 0.04
    records = simulate_trials(model, tau=tau)
    l_density = np.linspace(-2.0, 2.0, 300)

    for timestep, record in enumerate(records, start=1):
        densities = {}
        max_density = 0.0
        for action in (0, 1, 2):
            y = record["L"][record["action"] == action]
            if y.size > 15 and np.std(y) > 1e-3:
                density = gaussian_kde(y, bw_method=0.35)(l_density) * y.size
            else:
                density = np.zeros_like(l_density)
            densities[action] = density
            max_density = max(max_density, float(density.max()))

        if max_density <= 0.0:
            continue
        for action, color in ((2, DARK["S"]), (0, DARK["H0"]), (1, DARK["H1"])):
            ax.fill_betweenx(
                l_density,
                timestep,
                timestep + densities[action] / max_density * col_width,
                color=color, alpha=0.30, lw=0,
            )

    p_wait = sample_probability(model, tau)
    crossings = np.where(np.diff(np.sign(p_wait - 0.5)) != 0)[0]
    for l_cross in model.l_grid[crossings]:
        ax.plot(
            [0.6, 10 + col_width + 0.2], [l_cross, l_cross],
            color="0.2", ls="--", lw=1.0,
        )

    ax.axhline(0, color="0.9", lw=0.5, zorder=0)
    ax.set_xlim(0.6, 10 + col_width + 0.3)
    ax.set_ylim(-2, 2)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Log likelihood ratio", fontsize=12)
    ax.set_title(
        rf"$\lambda=\tau={tau:.2f}$",
        fontsize=10, loc="left",
    )
    ax.tick_params(labelsize=10)


# -------------------------------- layout ---------------------------------- #
def add_panel_label(ax, label, x_offset=-30, y_offset=6):
    ax.annotate(
        label, xy=(0, 1), xycoords="axes fraction",
        xytext=(x_offset, y_offset), textcoords="offset points",
        fontsize=12, fontweight="bold", va="bottom",
        annotation_clip=False,
    )


def main():
    data = build()

    # Eight equal vertical units: C and B each use four; A/D/E use rows 1:7,
    # giving a centered height of 1.5 left-column square cells.
    fig = plt.figure(figsize=(13.2, 8.7))
    outer = GridSpec(
        16, 3, figure=fig,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0] * 16,
        left=0.065, right=0.985, top=0.955, bottom=0.085,
        wspace=0.43, hspace=0.58,
    )

    # Left column: panel C above panel B.
    ax_c = fig.add_subplot(outer[1:10, 0])
    lc = panel_C(ax_c, data)
    '''
    grid_b = outer[4:7, 0].subgridspec(2, 1, hspace=0.14)
    ax_bt = fig.add_subplot(grid_b[0])
    ax_bb = fig.add_subplot(grid_b[1], sharex=ax_bt)
    panel_B(ax_bt, ax_bb, data)
    '''
    # 1 empty margin on each side, 8 for Panel B
    grid_b_width = outer[9:16, 0].subgridspec(
        1, 3,
        width_ratios=[1, 8, 1],
        wspace=0
    )
    
    grid_b = grid_b_width[0, 1].subgridspec(
        2, 1,
        hspace=0.14
    )
    
    ax_bt = fig.add_subplot(grid_b[0])
    ax_bb = fig.add_subplot(grid_b[1], sharex=ax_bt)
    
    panel_B(ax_bt, ax_bb, data)

    # Middle column: panel A centered over 1.5 row-heights.
    grid_a = outer[2:15, 1].subgridspec(2, 1, hspace=0.18)
    ax_au = fig.add_subplot(grid_a[0])
    ax_al = fig.add_subplot(grid_a[1], sharex=ax_au)
    panel_A(ax_au, ax_al, data)

    # Right column: D/E vertically stacked and sharing the time-step x-axis.
    grid_de = outer[2:15, 2].subgridspec(2, 1, hspace=0.18)
    ax_d = fig.add_subplot(grid_de[0])
    ax_e = fig.add_subplot(grid_de[1], sharex=ax_d)
    panel_D(ax_d, data)
    panel_E(ax_e, data)

    # Belief colorbar to the LEFT of panel C, without shrinking its square cell.
    '''
    if lc is not None:
        cax = inset_axes(
            ax_c, width="4.0%", height="68%", loc="center right",
            bbox_to_anchor=(0, 0.0, 1.0, 1.0),
            bbox_transform=ax_c.transAxes, borderpad=0,
        )
        colorbar = fig.colorbar(lc, cax=cax)
        colorbar.set_label(r"Belief $b = P(H_1)$", fontsize=12, labelpad=4)
        colorbar.ax.yaxis.set_label_position("right")
        colorbar.ax.yaxis.set_ticks_position("right")
        colorbar.ax.tick_params(labelsize=10)
        colorbar.set_ticks([0, 0.5, 1])
'''
    if lc is not None:
        cax = inset_axes(
            ax_c,
            width="68%",
            height="4.0%",
            loc="upper center",
            bbox_to_anchor=(0, .12, 1.0, 1.0),
            bbox_transform=ax_c.transAxes,
            borderpad=0,
        )
    
        colorbar = fig.colorbar(
            lc,
            cax=cax,
            orientation="horizontal",
        )
    
        colorbar.set_label(
            r"$b$",
            fontsize=12,
            labelpad=2,
        )
    
        colorbar.ax.xaxis.set_label_position("top")
        colorbar.ax.xaxis.set_ticks_position("top")
        colorbar.ax.tick_params(labelsize=10, pad=2)
        colorbar.set_ticks([0, 0.5, 1])

    add_panel_label(ax_au, "A")
    add_panel_label(ax_bt, "B")
    add_panel_label(ax_c, "C")
    add_panel_label(ax_d, "D")
    add_panel_label(ax_e, "E")

    svg_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_STEM}.svg") 
    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_STEM}.png")
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    model = data["model"]
    print(
        "Saved", svg_path, "and", png_path,
        f"| value iterations={model.iterations}",
        f"residual={model.residual:.3e}",
        f"panel-A b={data['b_A']:.4f}",
        f"Q={np.round(data['q_values_A'], 6)}",
        f"pi*={np.round(data['optimal_policy_A'], 6)}",
        f"soft-boundary beliefs={np.round(boundary_beliefs(model, 0.04), 3)}",
    )


if __name__ == "__main__":
    main()
