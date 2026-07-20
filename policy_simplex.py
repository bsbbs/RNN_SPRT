#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(tmp_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(tmp_root / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from scipy.stats import norm

from noise_model_runner import ModelConfig, compute_model


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTDIR = SCRIPT_DIR / "figures" / "policy_simplex_two_colorings"
TRIANGLE_HEIGHT = np.sqrt(3.0) / 2.0

DEFAULT_TAU_EXEC_LEVELS = [0.01, 0.1, 0.2, 1.0, 5.0, 10.0]
# DEFAULT_TAU_EXEC_LEVELS = [0.0]
# DEFAULT_SIGMA_REPR_LEVELS = [0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
DEFAULT_SIGMA_REPR_LEVELS = [0.0]
DEFAULT_RIGHT_PANEL_CURVE_INDEX = 0
DEFAULT_SIMPLEX_CMAP = "seismic"
DEFAULT_SIMPLEX_COLOR_LIMIT = None
DEFAULT_GROUND_VIEW_TIMESTEP = 1.0
DEFAULT_CUMULATIVE_EVIDENCE_RANGE = (-2.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--cost", type=float, default=0.03)
    parser.add_argument(
        "--tau-exec",
        type=float,
        nargs="+",
        default=DEFAULT_TAU_EXEC_LEVELS,
        help="One or more execution-noise softmax temperatures.",
    )
    parser.add_argument(
        "--sigma-repr",
        type=float,
        nargs="+",
        default=DEFAULT_SIGMA_REPR_LEVELS,
        help="One or more representation-noise levels.",
    )
    parser.add_argument(
        "--curve-index",
        type=int,
        default=DEFAULT_RIGHT_PANEL_CURVE_INDEX,
        help="Which generated parameter curve to show in the right probability panel.",
    )
    parser.add_argument("--n-l", type=int, default=2001)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument(
        "--ground-view-timestep",
        type=float,
        default=DEFAULT_GROUND_VIEW_TIMESTEP,
        help=(
            "Timestep used when converting representation-noise internal policy "
            "to ground-evidence policy; effective noise is sigma_repr * sqrt(t)."
        ),
    )
    parser.add_argument(
        "--evidence-range",
        type=float,
        nargs=2,
        default=DEFAULT_CUMULATIVE_EVIDENCE_RANGE,
        metavar=("MIN", "MAX"),
        help=(
            "Cumulative-evidence range for simplex color normalization. Both "
            "panels still draw the full computed policy curve."
        ),
    )
    parser.add_argument(
        "--evidence-min",
        type=float,
        default=None,
        help="Deprecated alias for the lower cumulative-evidence plotting limit.",
    )
    parser.add_argument(
        "--evidence-max",
        type=float,
        default=None,
        help="Deprecated alias for the upper cumulative-evidence plotting limit.",
    )
    parser.add_argument(
        "--simplex-linewidth",
        type=float,
        default=3.0,
        help="Line width for the probability curve mapped onto the simplex.",
    )
    parser.add_argument(
        "--simplex-cmap",
        type=str,
        default=DEFAULT_SIMPLEX_CMAP,
        help="Matplotlib colormap for cumulative-evidence coloring on the simplex.",
    )
    parser.add_argument(
        "--simplex-color-limit",
        type=float,
        default=DEFAULT_SIMPLEX_COLOR_LIMIT,
        help=(
            "Optional symmetric evidence limit used only for simplex color "
            "normalization. By default, the colormap follows --evidence-range."
        ),
    )
    return parser.parse_args()


def _simplex_color_range(
    evidence_min: float,
    evidence_max: float,
    simplex_color_limit: float | None,
) -> tuple[float, float]:
    if simplex_color_limit is None:
        return float(evidence_min), float(evidence_max)

    color_limit = abs(float(simplex_color_limit))
    if color_limit <= 0.0:
        raise ValueError("--simplex-color-limit must be positive.")
    return -color_limit, color_limit


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
        }
    )


def _save(fig, out_stem: Path, *, dpi: int = 500) -> Dict[str, str]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    svg = out_stem.with_suffix(".svg")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "svg": str(svg)}


def simplex_xy(probabilities: np.ndarray) -> np.ndarray:
    """Map (p_A, p_B, p_S) to p_S-top, p_A-left, p_B-right coordinates."""
    probs = np.asarray(probabilities, dtype=float)
    p_b = probs[..., 1]
    p_s = probs[..., 2]
    x = p_b + 0.5 * p_s
    y = TRIANGLE_HEIGHT * p_s
    return np.stack([x, y], axis=-1)


def _draw_simplex(ax) -> None:
    ax.set_facecolor("white")
    ax.patch.set_visible(True)
    ax.add_patch(
        Polygon(
            [[0.0, 0.0], [0.5, TRIANGLE_HEIGHT], [1.0, 0.0]],
            closed=True,
            facecolor="1",
            edgecolor="none",
            zorder=-2,
        )
    )
    ax.plot(
        [0.0, 0.5, 1.0, 0.0],
        [0.0, TRIANGLE_HEIGHT, 0.0, 0.0],
        color="black",
        lw=1.0,
        zorder=4,
    )
    ax.text(0.5, TRIANGLE_HEIGHT + 0.045, r"$p_S$", ha="center", va="bottom")
    ax.text(-0.025, -0.045, r"$p_A$", ha="left", va="top")
    ax.text(1.025, -0.045, r"$p_B$", ha="right", va="top")
    ax.set_xlim(-0.16, 1.16)
    ax.set_ylim(-0.12, TRIANGLE_HEIGHT + 0.14)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _colored_simplex_curve(
    ax,
    probabilities: np.ndarray,
    evidence: np.ndarray,
    *,
    color_min: float,
    color_max: float,
    linewidth: float,
    cmap,
    alpha: float = 0.96,
):
    xy = simplex_xy(probabilities)
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    segment_values = 0.5 * (evidence[:-1] + evidence[1:])
    norm = plt.Normalize(vmin=float(color_min), vmax=float(color_max), clip=True)
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=float(linewidth),
        alpha=float(alpha),
        zorder=3,
    )
    collection.set_array(segment_values)
    ax.add_collection(collection)
    return collection


def _policy_ground_view(
    result,
    ground_evidence: np.ndarray,
    *,
    ground_view_timestep: float,
) -> np.ndarray:
    """Convert internal policy p(a | L_tilde) to ground-view p(a | L)."""
    internal_policy = np.stack(
        [
            np.asarray(result.p_choose0, dtype=float),
            np.asarray(result.p_choose1, dtype=float),
            np.asarray(result.p_sample, dtype=float),
        ],
        axis=1,
    )

    if not result.has_representation_noise:
        probabilities = np.column_stack(
            [
                np.interp(
                    ground_evidence,
                    result.l_grid,
                    internal_policy[:, action],
                    left=float(internal_policy[0, action]),
                    right=float(internal_policy[-1, action]),
                )
                for action in range(3)
            ]
        )
    else:
        sigma_eff = float(result.config.sigma_repr) * np.sqrt(
            max(float(ground_view_timestep), 0.0)
        )
        if sigma_eff <= 0.0:
            probabilities = np.column_stack(
                [
                    np.interp(
                        ground_evidence,
                        result.l_grid,
                        internal_policy[:, action],
                        left=float(internal_policy[0, action]),
                        right=float(internal_policy[-1, action]),
                    )
                    for action in range(3)
                ]
            )
        elif result.noise == "representation":
            lower_l = float(result.lower_l)
            upper_l = float(result.upper_l)
            lower_z = (lower_l - ground_evidence) / sigma_eff
            upper_z = (upper_l - ground_evidence) / sigma_eff
            p_choose_a = norm.cdf(lower_z)
            p_choose_b = 1.0 - norm.cdf(upper_z)
            p_sample = norm.cdf(upper_z) - norm.cdf(lower_z)
            probabilities = np.stack([p_choose_a, p_choose_b, p_sample], axis=1)
        else:
            probabilities = np.zeros((ground_evidence.size, 3), dtype=float)
            nodes = np.sqrt(2.0) * sigma_eff * result.gh_x
            weights = result.gh_w / np.sqrt(np.pi)
            for node, weight in zip(nodes, weights):
                for action in range(3):
                    probabilities[:, action] += weight * np.interp(
                        ground_evidence + node,
                        result.l_grid,
                        internal_policy[:, action],
                        left=float(internal_policy[0, action]),
                        right=float(internal_policy[-1, action]),
                    )

    probabilities = np.clip(probabilities, 0.0, np.inf)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _build_normative_policy_curve(
    args: argparse.Namespace,
    *,
    tau_exec: float,
    sigma_repr: float,
) -> dict[str, np.ndarray | float | str]:
    cfg = ModelConfig(
        c=float(args.cost),
        tau_exec=float(tau_exec),
        n_l=int(args.n_l),
        tol=float(args.tol),
        max_iter=int(args.max_iter),
        sigma_repr=float(sigma_repr),
    )
    if float(tau_exec) <= 0.0:
        noise = "representation"
    elif float(sigma_repr) > 0.0:
        noise = "mixed"
    else:
        noise = "execution"
    result = compute_model(
        noise=noise,
        horizon="infinite",
        config=cfg,
        verbose=True,
    )
    evidence = np.asarray(result.l_grid, dtype=float)
    probabilities = _policy_ground_view(
        result,
        evidence,
        ground_view_timestep=float(args.ground_view_timestep),
    )

    return {
        "evidence": evidence,
        "belief": np.asarray(result.b_grid, dtype=float),
        "probabilities": probabilities,
        "full_evidence": evidence,
        "full_belief": np.asarray(result.b_grid, dtype=float),
        "full_probabilities": probabilities,
        "lower_l": float(result.lower_l),
        "upper_l": float(result.upper_l),
        "lower_b": float(result.lower_b),
        "upper_b": float(result.upper_b),
        "tau_exec": float(tau_exec),
        "sigma_repr": float(sigma_repr),
        "noise": noise,
        "ground_view_timestep": float(args.ground_view_timestep),
        "label": rf"$\tau={float(tau_exec):g}, \sigma_r={float(sigma_repr):g}$",
    }


def plot_normative_curve_figure(
    curves: list[dict[str, np.ndarray | float | str]],
    out_stem: Path,
    *,
    evidence_min: float,
    evidence_max: float,
    cost: float,
    curve_index: int,
    simplex_linewidth: float,
    simplex_cmap: str,
    simplex_color_limit: float | None,
) -> Dict[str, str]:
    _set_style()
    if not 0 <= int(curve_index) < len(curves):
        raise ValueError("--curve-index is out of range for the generated curves.")
    curve = curves[int(curve_index)]
    belief = np.asarray(curve["full_belief"], dtype=float)
    probabilities = np.asarray(curve["full_probabilities"], dtype=float)
    lower_b = float(curve["lower_b"])
    upper_b = float(curve["upper_b"])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.2, 3.0),
        constrained_layout=True,
        squeeze=False,
    )
    simplex_ax, curve_ax = axes[0]
    simplex_ax.set_box_aspect(1)
    curve_ax.set_box_aspect(1)

    _draw_simplex(simplex_ax)
    cmap = plt.get_cmap(str(simplex_cmap))
    color_min, color_max = _simplex_color_range(
        evidence_min,
        evidence_max,
        simplex_color_limit,
    )
    collection = None
    for curve_i, simplex_curve in enumerate(curves):
        collection = _colored_simplex_curve(
            simplex_ax,
            np.asarray(simplex_curve["probabilities"], dtype=float),
            np.asarray(simplex_curve["evidence"], dtype=float),
            color_min=color_min,
            color_max=color_max,
            linewidth=simplex_linewidth if curve_i == int(curve_index) else simplex_linewidth * 0.85,
            cmap=cmap,
            alpha=1.0 if curve_i == int(curve_index) else 0.82,
        )
    if collection is None:
        raise RuntimeError("No curves were available to plot.")
    cbar = fig.colorbar(
        collection,
        ax=simplex_ax,
        orientation="vertical",
        fraction=0.055,
        pad=0.05,
        aspect=28,
    )
    cbar.set_label("Cumulative evidence")

    curve_ax.plot(belief, probabilities[:, 0], color="blue", lw=2.0, label="Choose A")
    curve_ax.plot(belief, probabilities[:, 1], color="red", lw=2.0, label="Choose B")
    curve_ax.plot(belief, probabilities[:, 2], color="green", lw=2.0, label="Sample")
    curve_ax.axvline(lower_b, color="black", linestyle=":", linewidth=1.0)
    curve_ax.axvline(upper_b, color="black", linestyle=":", linewidth=1.0)
    curve_ax.set_xlim(0.0, 1.0)
    curve_ax.set_ylim(-0.03, 1.03)
    curve_ax.set_xlabel(r"Belief $b=P(H_1 \mid evidence)$")
    curve_ax.set_ylabel("Execution probability")
    # curve_ax.set_title("Normative probability curve")
    curve_ax.legend(fontsize=8, frameon=True, title=str(curve["label"]))

    # fig.suptitle(
    #     rf"Infinite-horizon stochastic execution policy, $c={cost:g}$",
    #     fontsize=12,
    # )
    return _save(fig, out_stem)


def main() -> None:
    args = parse_args()
    evidence_min, evidence_max = [float(v) for v in args.evidence_range]
    if args.evidence_min is not None:
        evidence_min = float(args.evidence_min)
    if args.evidence_max is not None:
        evidence_max = float(args.evidence_max)
    if not evidence_min < evidence_max:
        raise ValueError("The cumulative-evidence plotting minimum must be smaller than the maximum.")

    args.evidence_min = evidence_min
    args.evidence_max = evidence_max

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    curves = [
        _build_normative_policy_curve(args, tau_exec=tau_exec, sigma_repr=sigma_repr)
        for sigma_repr in args.sigma_repr
        for tau_exec in args.tau_exec
    ]
    figures = plot_normative_curve_figure(
        curves,
        outdir / "normative_policy_curve_simplex",
        evidence_min=evidence_min,
        evidence_max=evidence_max,
        cost=float(args.cost),
        curve_index=int(args.curve_index),
        simplex_linewidth=float(args.simplex_linewidth),
        simplex_cmap=str(args.simplex_cmap),
        simplex_color_limit=args.simplex_color_limit,
    )
    simplex_color_min, simplex_color_max = _simplex_color_range(
        evidence_min,
        evidence_max,
        args.simplex_color_limit,
    )
    summary = {
        "model": "normative_infinite_horizon_stochastic_execution",
        "cost": float(args.cost),
        "tau_exec_values": [float(v) for v in args.tau_exec],
        "sigma_repr_values": [float(v) for v in args.sigma_repr],
        "curve_index": int(args.curve_index),
        "right_panel_curve": {
            "tau_exec": float(curves[int(args.curve_index)]["tau_exec"]),
            "sigma_repr": float(curves[int(args.curve_index)]["sigma_repr"]),
            "noise": str(curves[int(args.curve_index)]["noise"]),
        },
        "n_l": int(args.n_l),
        "ground_view_timestep": float(args.ground_view_timestep),
        "cumulative_evidence_color_range": [evidence_min, evidence_max],
        "simplex_color_range": [simplex_color_min, simplex_color_max],
        "n_simplex_curves": len(curves),
        "simplex_definition": {
            "policy": "(p_A, p_B, p_S)",
            "constraint": "p_A + p_B + p_S = 1",
            "vertices": {
                "top": "p_S = 1",
                "bottom_left": "p_A = 1",
                "bottom_right": "p_B = 1",
            },
        },
        "simplex_cmap": str(args.simplex_cmap),
        "figures": figures,
    }
    summary_path = outdir / "normative_policy_curve_simplex_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({**summary, "summary_json": str(summary_path)}, indent=2))

if __name__ == "__main__":
    main()
