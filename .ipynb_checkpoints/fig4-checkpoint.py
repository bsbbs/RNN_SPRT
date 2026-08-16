#!/usr/bin/env python3
"""
Figure 4 -- Extension of the sequential-sampling model beyond the
time-unconstrained condition.

The figure contrasts the analytical p(sample) policy, plotted against the
ground-truth cumulative evidence (log10 LR), for four observer types:

    Ideal observer  -> no representation noise, no execution noise
    Soft policy     -> softmax action selection (tau_exec)
    Noisy belief    -> Gaussian internal-evidence noise (sigma_repr)
    Mixed           -> both sources combined

Row 1 (panels A-D): time-*unconstrained* condition (infinite horizon).
    The optimal internal boundaries are stationary.  When mapped onto the
    ground-truth log10 LR axis, the ideal and soft-policy curves are identical
    across time steps, whereas the noisy-belief and mixed curves broaden with
    sqrt(t) because the internal-evidence noise accumulates.

Row 2 (panels E-H): time-*constrained* condition (finite horizon, deadline at
    t = 10).  Beyond t = 10 the agent can no longer earn reward but still pays
    the sampling cost, so the continuation value collapses: the boundaries
    shrink over time and p(sample) falls to 0 everywhere at the final step.

Every curve is the ANALYTICAL dynamic-programming solution obtained via
`p_sample_ground_view` (closed-form Gaussian-CDF / Gauss-Hermite marginalisation
for representation noise, exact softmax for execution noise).  No Monte-Carlo
simulation is used to draw the curves.

Notation
--------
sigma_repr : standard deviation of the Gaussian representation noise.
tau_exec   : softmax temperature of the execution noise.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from noise_model_runner import ModelConfig, compute_model, p_sample_ground_view

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
TITLE_FS = 12     # legend-title / panel-title fontsize
PANEL_FS = 16     # panel letters

T_CMAP = "viridis"
# Truncate the top of viridis: the full range ends in a pale yellow that is
# barely visible on white, so the last time step is kept dark enough to read.
T_CRANGE = (0.0, 0.72)
LINE_LW = 1.5

# ---------------------------- model settings ------------------------------ #
COST = 0.004        # sample cost
SIGMA_REPR = 0.30   # representation-noise std
TAU_EXEC = 0.010    # execution-noise (softmax) temperature
MAX_TIMESTEP = 10   # time steps 1..MAX_TIMESTEP
DEADLINE = MAX_TIMESTEP + 1   # finite-horizon deadline

L_MIN, L_MAX, N_L = -3.5, 3.5, 700   # ground-truth log10-LR axis of the curves

# ----------------------------------------------------------------------------
# The four observer conditions (shared by both rows)
# ----------------------------------------------------------------------------
CONDITIONS = [
    ("Ideal observer", dict(noise="representation", sigma_repr=0.0,        tau_exec=TAU_EXEC)),
    ("Soft policy",    dict(noise="execution",      sigma_repr=0.0,        tau_exec=TAU_EXEC)),
    ("Noisy belief",   dict(noise="representation", sigma_repr=SIGMA_REPR, tau_exec=TAU_EXEC)),
    ("Mixed",          dict(noise="mixed",          sigma_repr=SIGMA_REPR, tau_exec=TAU_EXEC)),
]

# Row 1 = time unconstrained (infinite horizon); Row 2 = time constrained (finite).
ROWS = [
    ("infinite", "Time unconstrained"),
    ("finite",   f"Time constrained (t = {MAX_TIMESTEP})"),
]
LETTERS = [["A", "B", "C", "D"], ["E", "F", "G", "H"]]


def level_color(t):
    """Graded color of time step t = 1 .. MAX_TIMESTEP."""
    c_lo, c_hi = T_CRANGE
    return plt.get_cmap(T_CMAP)(c_lo + (c_hi - c_lo) * (t - 1) / (MAX_TIMESTEP - 1))


def build_result(noise, horizon, sigma_repr, tau_exec):
    """Solve the DP for one observer/horizon combination."""
    cfg = ModelConfig(
        c=COST,
        sigma_repr=sigma_repr,
        tau_exec=tau_exec,
        max_timestep=MAX_TIMESTEP,
        deadline=DEADLINE,
    )
    return compute_model(noise=noise, horizon=horizon, config=cfg, verbose=False)


# ------------------------------- panels ----------------------------------- #
def panel(ax, log_lr, horizon, condition):
    """Draw the p(sample) curves of one observer/horizon combination."""
    name, kw = condition
    res = build_result(kw["noise"], horizon, kw["sigma_repr"], kw["tau_exec"])

    for t in range(MAX_TIMESTEP, 0, -1):
        p = p_sample_ground_view(res, log_lr, t)
        ax.plot(log_lr, p, color=level_color(t), lw=LINE_LW, zorder=3 + t)

    ax.axhline(0.5, color="0.9", lw=0.6, ls="--", zorder=0)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=TICK_FS)
    return res


# --------------------------- time-step legend ----------------------------- #
def time_legend(fig, rect, title="Time step"):
    """Graded short lines standing in for the shared time-step colorbar.

    Instead of listing all MAX_TIMESTEP levels, only the first and the last are
    labelled and an arrow runs from the top line (t = 1) to the bottom line
    (t = MAX_TIMESTEP).
    """
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    y_top, y_bot = 0.88, 0.06
    y_levels = np.linspace(y_top, y_bot, MAX_TIMESTEP)
    x_line, x_lbl, x_arr = (0.0, 0.36), 0.44, 0.80

    for t, y in enumerate(y_levels, start=1):
        ax.plot(x_line, [y, y], color=level_color(t), lw=1.4,
                solid_capstyle="butt", clip_on=False)

    ax.text(0.0, 1.0, title, ha="left", va="bottom", fontsize=TITLE_FS)
    ax.text(x_lbl, y_levels[0], r"$1$", ha="left", va="center", fontsize=LEG_FS)
    ax.text(x_lbl, y_levels[-1], rf"${MAX_TIMESTEP}$", ha="left", va="center",
            fontsize=LEG_FS)
    ax.annotate(
        "", xy=(x_arr, y_levels[-1]), xytext=(x_arr, y_levels[0]),
        arrowprops=dict(arrowstyle="->", linewidth=0.8, color="0.35"),
        annotation_clip=False,
    )
    return ax


# -------------------------------- layout ---------------------------------- #
def add_panel_label(ax, label, x_offset=-26, y_offset=4):
    ax.annotate(
        label, xy=(0, 1), xycoords="axes fraction",
        xytext=(x_offset, y_offset), textcoords="offset points",
        fontsize=PANEL_FS, fontweight="bold", va="bottom",
        annotation_clip=False,
    )


def make_figure():
    # Ground-truth log10-LR axis on which the analytical policy is evaluated.
    log_lr = np.linspace(L_MIN, L_MAX, N_L)

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.7), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.075, right=0.885, top=0.90, bottom=0.115, wspace=0.18, hspace=0.30,
    )

    results = {}
    for r, (horizon, row_label) in enumerate(ROWS):
        for c_idx, condition in enumerate(CONDITIONS):
            ax = axes[r][c_idx]
            results[(horizon, condition[0])] = panel(ax, log_lr, horizon, condition)

            if r == 0:
                ax.set_title(condition[0], fontsize=TITLE_FS)
            if c_idx == 0:
                ax.set_ylabel(r"$p(\mathrm{sample})$", fontsize=LBL_FS)
            if r == len(ROWS) - 1:
                ax.set_xlabel(r"Cumulative evidence ($L$)", fontsize=LBL_FS)

            add_panel_label(ax, LETTERS[r][c_idx],
                            x_offset=-34 if c_idx == 0 else -18)

    # Shared time-step legend, vertically centered to the right of the panels.
    legend_height = 0.32
    time_legend(fig, [0.905, 0.5 - legend_height / 2, 0.07, legend_height])

    return fig, results


def main():
    fig, results = make_figure()

    svg_path = os.path.join(outdir, "fig4.svg")
    png_path = os.path.join(outdir, "fig4.png")
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    finite_mixed = results[("finite", "Mixed")]
    print(
        "Saved", svg_path, "and", png_path, "\n"
        f"cost={COST}",
        f"sigma_repr={SIGMA_REPR}",
        f"tau_exec={TAU_EXEC}",
        f"max_timestep={MAX_TIMESTEP}",
        f"deadline={DEADLINE}\n"
        "finite-horizon mixed internal boundaries "
        f"(t=1)=({finite_mixed.lower_l[1]:.3f}, {finite_mixed.upper_l[1]:.3f})",
        f"(t={MAX_TIMESTEP})=({finite_mixed.lower_l[MAX_TIMESTEP]:.3f}, "
        f"{finite_mixed.upper_l[MAX_TIMESTEP]:.3f})",
    )


if __name__ == "__main__":
    main()