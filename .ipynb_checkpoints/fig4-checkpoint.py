"""
Figure 4 - Extension of the sequential-sampling model beyond the
time-unconstrained condition.

The figure contrasts the analytical p(sample) policy, plotted against the
ground-truth cumulative evidence (Log LR), for four observer types:

    Canonical optimum   -> no representation noise, no execution noise
    Execution noise     -> softmax action selection (tau_exec)
    Representation noise -> Gaussian internal-evidence noise (sigma_repr)
    Mixed noise         -> both sources combined

Row 1 (panels A-D): time-*unconstrained* condition (infinite horizon).
    The optimal internal boundaries are stationary. When mapped onto the
    ground-truth Log LR axis, the canonical and execution curves are identical
    across time steps, whereas the representation and mixed curves broaden with
    sqrt(t) because the internal-evidence noise accumulates.

Row 2 (panels E-H): time-*constrained* condition (finite horizon, deadline at
    t = 10). Beyond t = 10 the agent can no longer earn reward but still pays
    the sampling cost, so the continuation value collapses: the boundaries
    shrink over time and p(sample) falls to 0 everywhere at the final step.

Every curve is the ANALYTICAL dynamic-programming solution obtained via
`p_sample_ground_view` (closed-form Gaussian-CDF / Gauss-Hermite marginalisation
for representation noise, exact softmax for execution noise). No Monte-Carlo
simulation is used to draw the curves.

Conventions used throughout the project:
    * sample cost           c        = 0.04
    * axis-label fontsize            = 12
    * tick-label fontsize            = 10
"""
import os
from pathlib import Path

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from noise_model_runner import ModelConfig, compute_model, p_sample_ground_view

# --------------------------------------------------------------------------
# Output directory
# --------------------------------------------------------------------------
fallback_outdir = str(Path.home())
try:
    outdir = os.path.dirname(os.path.abspath(__file__))
except (NameError, TypeError):
    outdir = fallback_outdir


# ----------------------------------------------------------------------------
# Project-wide conventions
# ----------------------------------------------------------------------------
plt.rcParams["font.family"] = "Arial" # "Sans"   # when Arial is unavailable in this env
LABEL_FS = 12                                  # x / y axis label fontsize
TICK_FS = 10                                   # x / y tick-label fontsize

C = 0.025            # sample cost (project-wide)
SIGMA_REPR = .1 #0.25   # representation-noise std
TAU_EXEC = .010 #0.025    # execution-noise (softmax) temperature
MAX_T = 10          # time steps 1..10; deadline (finite horizon) at t = MAX_T + 1

# ----------------------------------------------------------------------------
# The four observer conditions (shared by both rows)
# ----------------------------------------------------------------------------
CONDITIONS = [
    ("Ideal observer",            dict(noise="representation", sigma_repr=0.0,        tau_exec=TAU_EXEC)),
    ("Soft policy",      dict(noise="execution",      sigma_repr=0.0,        tau_exec=TAU_EXEC)),
    ("Noisy belief", dict(noise="representation", sigma_repr=SIGMA_REPR, tau_exec=TAU_EXEC)),
    ("Mixed",          dict(noise="mixed",          sigma_repr=SIGMA_REPR, tau_exec=TAU_EXEC)),
]

# Row 1 = time unconstrained (infinite horizon); Row 2 = time constrained (finite).
ROWS = [
    ("infinite", "Time unconstrained"),
    ("finite",   "Time constrained (t = 10)"),
]
LETTERS = [["A", "B", "C", "D"], ["E", "F", "G", "H"]]


def build_result(noise: str, horizon: str, sigma_repr: float, tau_exec: float):
    """Solve the DP for one observer/horizon combination."""
    cfg = ModelConfig(
        c=C,
        sigma_repr=sigma_repr,
        tau_exec=tau_exec,
        max_timestep=MAX_T,
        deadline=MAX_T + 1,
    )
    return compute_model(noise=noise, horizon=horizon, config=cfg, verbose=False)


def make_figure():
    # Ground-truth Log LR axis on which the analytical policy is evaluated.
    log_lr = np.linspace(-3.5, 3.5, 700)
    colors = cm.viridis(np.linspace(0.0, 1.0, MAX_T))

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.7), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.11, right=0.885, top=0.88, bottom=0.11, wspace=0.18, hspace=0.28
    )

    for r, (horizon, row_label) in enumerate(ROWS):
        for c_idx, (name, kw) in enumerate(CONDITIONS):
            ax = axes[r][c_idx]
            res = build_result(kw["noise"], horizon, kw["sigma_repr"], kw["tau_exec"])

            for t in range(MAX_T, 0, -1):
                p = p_sample_ground_view(res, log_lr, t)
                ax.plot(log_lr, p, color=colors[t - 1], linewidth=1.5)
            ax.axhline(0.5, color="0.9", lw=0.6, ls="--", zorder=0)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(axis="both", labelsize=TICK_FS)

            if r == 0:
                ax.set_title(name, fontsize=LABEL_FS)
            if c_idx == 0:
                ax.set_ylabel(r"$p(\mathrm{sample})$", fontsize=LABEL_FS)

            ax.text(
                0.04, 0.93, LETTERS[r][c_idx],
                transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left",
            )

    # Shared time-step colorbar, sized like a single panel and vertically centered.
    cbar_height = 0.32
    cbar_bottom = 0.5 - cbar_height / 2
    
    cax = fig.add_axes([0.905, cbar_bottom, 0.017, cbar_height])
    
    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=1, vmax=MAX_T),
    )
    sm.set_array([])
    
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Time step", fontsize=LABEL_FS)
    cbar.set_ticks([1, 5, 10])
    cbar.ax.tick_params(labelsize=TICK_FS)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    fig.savefig(rf"figure4_time_constraint_tau={TAU_EXEC:.3f}_s={SIGMA_REPR:.2f}_c={C:.3f}.png", dpi=300)
    fig.savefig(rf"figure4_time_constraint_tau={TAU_EXEC:.3f}_s={SIGMA_REPR:.2f}_c={C:.3f}.svg")
    print("Saved figure4_time_constraint.png and figure4_time_constraint.svg")
