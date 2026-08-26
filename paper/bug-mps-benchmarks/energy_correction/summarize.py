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

def variant_label(row: dict[str, Any]) -> str:
    """Return the display label for one result row.

    Returns:
        ``BUG`` for the uncorrected variant, otherwise ``BUG+EC (tol ...)``.
    """
    if not row["conserve_energy"]:
        return "BUG"
    return f"BUG+EC (tol {row['conserve_tol']:.0e})"


def variant_key(row: dict[str, Any]) -> tuple[int, float]:
    """Return a sort key placing the uncorrected variant first, then loosest guard first.

    Returns:
        A tuple usable as a sort key.
    """
    return (int(row["conserve_energy"]), -row["conserve_tol"])


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
            rows_at_dt = sorted((row for key, row in block.items() if key[0] == dt), key=variant_key)
            for row in rows_at_dt:
                fired = "--" if not row["conserve_energy"] else f"{row['firings']}/{row['hook_calls']}"
                print(
                    f"| {dt:g} | {variant_label(row)} | {row['infidelity']:.4e} | "
                    f"{row['energy_drift'][-1]:.3e} | {row['max_bond'][-1]} | {fired} | "
                    f"{row['wall_seconds']:.1f} |"
                )

        # Rank preservation and state inertness, stated as ratios against BUG.
        print("\nRatios against uncorrected BUG (1.000 = inert):")
        for dt in sorted({key[0] for key in block}):
            rows_at_dt = sorted((row for key, row in block.items() if key[0] == dt), key=variant_key)
            base = next((row for row in rows_at_dt if not row["conserve_energy"]), None)
            if base is None:
                continue
            parts = []
            for row in rows_at_dt:
                if not row["conserve_energy"]:
                    continue
                ratio = row["infidelity"] / base["infidelity"] if base["infidelity"] > 0 else float("nan")
                same_bond = row["max_bond"] == base["max_bond"]
                bonds = "identical" if same_bond else "DIFFER"
                parts.append(f"{variant_label(row)}: I x{ratio:.6f}, bonds {bonds}")
            print(f"  h={dt:g}  " + ";  ".join(parts))


if __name__ == "__main__":
    main()
