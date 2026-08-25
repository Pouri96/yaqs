#!/usr/bin/env python3
"""Derive the comparison tables from the saved energy-comparison runs.

Reads any number of ``l16_energy_comparison`` JSON payloads and emits one
Markdown table per (model, bond cap), with the three variants side by side. The
columns are the two the study is about -- accuracy and energy drift -- plus the
bond trace, which must be identical across variants for the rank-preservation
claim to hold.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

VARIANT_LABELS = {
    "bug": "BUG",
    "bug_ec_1e-12": "BUG+EC (tol 1e-12)",
    "bug_ec_1e-14": "BUG+EC (tol 1e-14)",
}
VARIANT_ORDER = ("bug", "bug_ec_1e-12", "bug_ec_1e-14")


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=sorted(HERE.glob("l16_*.json")))
    return parser.parse_args()


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    """Read every result row from the given payloads.

    Args:
        paths: JSON payloads written by the comparison runner.

    Returns:
        A flat list of result rows.
    """
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Payloads written before the cap became selectable carry neither key.
        fallback = payload.get("max_bond_dim", 512)
        for row in payload["results"]:
            row.setdefault("max_bond_dim", fallback)
            rows.append(row)
    return rows


def main() -> None:
    """Print one table per model and bond cap."""
    args = parse_args()
    rows = load_rows(list(args.inputs))
    if not rows:
        print("No result payloads found.")
        return

    grouped: dict[tuple[str, int], dict[tuple[float, str], dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["model"], row["max_bond_dim"]][row["dt"], row["variant"]] = row

    for (model, cap), block in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        binding = any(row["max_bond"][-1] >= cap for row in block.values())
        print(f"\n### {model.upper()}, chi_max = {cap}" + ("  (cap binds)" if binding else "  (cap inactive)"))
        print("\n| h | variant | infidelity | \\|<H> - o0\\| | chi | firings | wall (s) |")
        print("|---:|:---|---:|---:|---:|---:|---:|")
        for dt in sorted({key[0] for key in block}):
            for variant in VARIANT_ORDER:
                row = block.get((dt, variant))
                if row is None:
                    continue
                fired = "--" if not row["conserve_energy"] else f"{row['firings']}/{row['hook_calls']}"
                print(
                    f"| {dt:g} | {VARIANT_LABELS[variant]} | {row['infidelity']:.4e} | "
                    f"{row['energy_drift'][-1]:.3e} | {row['max_bond'][-1]} | {fired} | "
                    f"{row['wall_seconds']:.1f} |"
                )

        # Rank preservation and state inertness, stated as ratios against BUG.
        print("\nRatios against uncorrected BUG (1.000 = inert):")
        for dt in sorted({key[0] for key in block}):
            base = block.get((dt, "bug"))
            if base is None:
                continue
            parts = []
            for variant in VARIANT_ORDER[1:]:
                row = block.get((dt, variant))
                if row is None:
                    continue
                ratio = row["infidelity"] / base["infidelity"] if base["infidelity"] > 0 else float("nan")
                same_bond = row["max_bond"] == base["max_bond"]
                parts.append(f"{VARIANT_LABELS[variant]}: I x{ratio:.6f}, bonds {'identical' if same_bond else 'DIFFER'}")
            print(f"  h={dt:g}  " + ";  ".join(parts))


if __name__ == "__main__":
    main()
