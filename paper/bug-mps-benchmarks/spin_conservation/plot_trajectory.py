#!/usr/bin/env python3
"""Figure 1 of the standalone conservation paper: invariant drift against time.

Reads ``l16_trajectory.json`` and draws three stacked panels sharing a time axis -- the
energy, the largest of the three total-spin components, and ``S^2`` -- each with seven
curves under the BUG composition. The uncorrected run is black. Each restored set has its
own colour, solid for the correction applied at the one centre the compression leaves and
dashed for the correction applied at every centre in turn.

This script only plots. It runs under any interpreter with matplotlib and does not import
the simulator.

Writes into the manuscript's figure directory when this repository sits beside the
manuscript tree, and next to this script otherwise. Pass ``--output-dir`` to choose.

Run: ``python plot_trajectory.py``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
# The manuscript tree is a sibling of this repository, so it is absent from a plain clone.
_MANUSCRIPT_FIGURES = HERE.parents[3] / "paper" / "spc_mps" / "figures"
DEFAULT_OUT = _MANUSCRIPT_FIGURES if _MANUSCRIPT_FIGURES.is_dir() else HERE

# One colour per restored set, black for the uncorrected run. Solid is the correction at the
# one centre the compression leaves, dashed the same set applied at every centre in turn.
COLOR = {
    "none": "#000000",
    "joint4": "#c1272d",
    "jointS2": "#0b5394",
    "joint5": "#1b7837",
}
SET_LABEL = {
    "none": "uncorrected",
    "joint4": r"$\{\hat H,\hat S^a\}$",
    "jointS2": r"$\{\hat H,\hat S^2\}$",
    "joint5": r"$\{\hat H,\hat S^a,\hat S^2\}$",
}
# Drawing order, so the uncorrected reference sits on top of the corrected curves.
ORDER = (
    ("joint4", "none"),
    ("joint4", "full"),
    ("jointS2", "none"),
    ("jointS2", "full"),
    ("joint5", "none"),
    ("joint5", "full"),
    ("none", "none"),
)
INTEGRATOR = "bug"
FLOOR = 1e-17


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "l16_trajectory.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Draw the arms that are present and name the ones that are not, instead of "
        "stopping. For previewing the layout while the runs are still going; a figure "
        "drawn this way is missing curves and is not the manuscript figure.",
    )
    return parser.parse_args()


def main() -> int:
    """Draw the figure.

    Returns:
        Zero on success.
    """
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    arms = payload["arms"]
    # The top panel is the relative energy drift, as the table defines it; every other
    # target has a vanishing initial value and is reported absolutely.
    energy_scale = abs(float(payload["initial_values"]["H"])) or 1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.2), sharex=True)
    panels = (
        (r"$|\delta_E|$", "energy"),
        (r"$\max_a\,|\delta_{S^a}|$", "spin"),
        (r"$|\delta_{S^2}|$", "s2"),
    )

    # Arms written before the sweep field existed are single-centre runs.
    indexed = {(a["integrator"], a["variant"], a.get("sweep", "none")): a for a in arms}
    missing = [k for k in ORDER if (INTEGRATOR, *k) not in indexed]
    if missing and not args.allow_partial:
        msg = f"trajectory data is missing {len(missing)} arm(s): {missing}"
        raise SystemExit(msg)
    if missing:
        print(f"PARTIAL: {len(missing)} of {len(ORDER)} arms absent, not drawn: {missing}")

    for variant, sweep in ORDER:
        if (INTEGRATOR, variant, sweep) not in indexed:
            continue
        arm = indexed[(INTEGRATOR, variant, sweep)]
        label = SET_LABEL[variant]
        if variant != "none":
            label += ", every center" if sweep == "full" else ", one center"
        times = np.asarray(arm["times"])
        drift = arm["drift"]
        energy = np.abs(np.asarray(drift["H"])) / energy_scale
        spin = np.max(np.abs(np.array([drift["Sx"], drift["Sy"], drift["Sz"]])), axis=0)
        s2 = np.abs(np.asarray(drift["S2"]))
        for ax, series in zip(axes, (energy, spin, s2), strict=False):
            ax.plot(
                times,
                np.maximum(series, FLOOR),
                color=COLOR[variant],
                ls="--" if sweep == "full" else "-",
                label=label,
            )

    for ax, (label, _) in zip(axes, panels, strict=False):
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(True, which="major", lw=0.3, alpha=0.4)
        ax.tick_params(which="both", top=True, right=True)

    axes[-1].set_xlabel(r"$t$")
    axes[0].set_ylim(top=3e-2)
    # Seven entries do not fit inside a panel at this width, so the legend goes below the
    # stack in two columns.
    handles, labels = axes[0].get_legend_handles_labels()
    # The uncorrected run is drawn last so it sits on top, but it reads first in the legend.
    order = [len(handles) - 1, *range(len(handles) - 1)]
    fig.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="lower center", bbox_to_anchor=(0.5, -0.015), frameon=False, ncol=2, fontsize=5.5,
        handlelength=1.5, columnspacing=0.8, labelspacing=0.15, borderaxespad=0.0,
    )

    fig.align_ylabels(axes)
    fig.tight_layout(pad=0.3, h_pad=0.3, rect=(0, 0.115, 1, 1))
    for suffix in ("pdf", "png"):
        path = args.output_dir / f"trajectory.{suffix}"
        fig.savefig(path, dpi=400, bbox_inches="tight")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
