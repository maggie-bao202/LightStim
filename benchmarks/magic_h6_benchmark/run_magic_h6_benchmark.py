"""
End-to-end Magic-H6 benchmark runner for LightStim.

Runs both H6 levels:
- Level 1: [[6,2,2]] with detector/observable scoring via run_simulation
- Level 2: [[36,4,4]] framework-native circuit, tracker detector/observable scoring
  via run_simulation_level2 (LER is a conservative upper bound -- see the protocol
  module docstring's "Known limitations")

Results are appended to CSV with checkpoint skipping.

Usage
-----
    # Quick smoke test (both levels)
    PYTHONPATH=. python benchmarks/magic_h6_benchmark/run_magic_h6_benchmark.py --quick

    # Full sweep on level 1 only
    PYTHONPATH=. python benchmarks/magic_h6_benchmark/run_magic_h6_benchmark.py \
        --levels 1 --p-values 1e-3 2e-3 5e-3

    # Level 2 custom sweep (level 2 needs lower p than level 1)
    PYTHONPATH=. python benchmarks/magic_h6_benchmark/run_magic_h6_benchmark.py \
        --levels 2 --p-values-l2 3e-4 5e-4 1e-3 --num-samples-l2 500000
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))

from lightstim.protocols.magic_h6_benchmark import (
    build_h6_circuit,
    run_simulation,
    run_simulation_level2,
)

_RESULT_KEYS = frozenset(
    {
        "shots",
        "accepted",
        "failed",
        "post_selection_rate",
        "logical_error_rate",
        "seconds",
    }
)

_COLUMNS = [
    "level",
    "code",
    "k",
    "rounds",
    "mode",
    "p",
    "max_shots_l1",
    "max_errors_l1",
    "num_samples_l2",
    "shots",
    "accepted",
    "failed",
    "post_selection_rate",
    "logical_error_rate",
    "seconds",
]


def _ck_key(row: dict) -> tuple:
    # Normalize every field the same way whether it arrives as a native value
    # (fresh row_prefix) or as a string (csv.DictReader). Numerics collapse to a
    # canonical float repr so e.g. p=0.01 and "0.01" hash identically; anything
    # non-numeric (code, mode) falls back to str. Without this the checkpoint
    # never matches on reload and every config is recomputed.
    def _norm(v) -> str:
        try:
            return f"{float(v):.6e}"
        except (TypeError, ValueError):
            return str(v)

    return tuple(
        _norm(v)
        for k, v in sorted(row.items())
        if k not in _RESULT_KEYS
    )


def _load_done(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return {_ck_key(r) for r in reader}


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        if header:
            writer.writeheader()
        writer.writerow(row)


def _run_level1(args, out_path: Path) -> None:
    circuit, info, system = build_h6_circuit(level=1)
    data_indices = list(system.data_indices)
    done = _load_done(out_path)

    for p in args.p_values:
        row_prefix = {
            "level": 1,
            "code": info["code"],
            "k": info["k"],
            "rounds": 1,
            "mode": args.mode,
            "p": p,
            "max_shots_l1": args.max_shots_l1,
            "max_errors_l1": args.max_errors_l1,
            "num_samples_l2": args.num_samples_l2,
        }
        if _ck_key(row_prefix) in done:
            print(f"  SKIP L1 p={p:.2e}")
            continue

        print(f"  L1 p={p:.2e} ...", flush=True)
        t0 = time.perf_counter()
        stats = run_simulation(
            circuit,
            p,
            mode=args.mode,
            data_indices=data_indices,
            max_shots=args.max_shots_l1,
            max_errors=args.max_errors_l1,
            batch_size=args.batch_size_l1,
            num_workers=args.num_workers,
            print_progress=args.print_progress,
        )
        elapsed = time.perf_counter() - t0

        row = {
            **row_prefix,
            "shots": stats.shots,
            "accepted": stats.post_selected_shots,
            "failed": stats.errors,
            "post_selection_rate": stats.post_selection_rate,
            "logical_error_rate": stats.logical_error_rate,
            "seconds": elapsed,
        }
        _append_row(out_path, row)
        print(
            "    -> "
            f"accept={stats.post_selection_rate:.4f}, "
            f"LER={stats.logical_error_rate:.3e}, "
            f"shots={stats.shots}",
            flush=True,
        )


def _run_level2(args, out_path: Path) -> None:
    circuit, info, system = build_h6_circuit(level=2)
    data_indices = list(system.data_indices)
    done = _load_done(out_path)

    for p in args.p_values_l2:
        row_prefix = {
            "level": 2,
            "code": info["code"],
            "k": info["k"],
            "rounds": 1,
            "mode": args.mode,
            "p": p,
            "max_shots_l1": args.max_shots_l1,
            "max_errors_l1": args.max_errors_l1,
            "num_samples_l2": args.num_samples_l2,
        }
        if _ck_key(row_prefix) in done:
            print(f"  SKIP L2 p={p:.2e}")
            continue

        print(f"  L2 p={p:.2e} ...", flush=True)
        t0 = time.perf_counter()
        stats = run_simulation_level2(
            circuit,
            p,
            info,
            mode=args.mode,
            data_indices=data_indices,
            num_samples=args.num_samples_l2,
        )
        elapsed = time.perf_counter() - t0

        row = {
            **row_prefix,
            "shots": stats["shots"],
            "accepted": stats["accepted"],
            "failed": stats["failed"],
            "post_selection_rate": stats["post_selection_rate"],
            "logical_error_rate": stats["logical_error_rate"],
            "seconds": elapsed,
        }
        _append_row(out_path, row)
        print(
            "    -> "
            f"accept={stats['post_selection_rate']:.4f}, "
            f"LER={stats['logical_error_rate']:.3e}, "
            f"shots={stats['shots']}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", nargs="+", type=int, choices=[1, 2], default=[1, 2])
    ap.add_argument(
        "--p-values",
        nargs="+",
        type=float,
        default=[3e-3, 5e-3, 1e-2],
        help="Level-1 physical error rates to sweep.",
    )
    ap.add_argument(
        "--p-values-l2",
        nargs="+",
        type=float,
        default=[3e-4, 5e-4, 1e-3],
        help=(
            "Level-2 physical error rates to sweep. Lower than level 1: the "
            "20-patch circuit post-selects on ~120 detectors, so acceptance "
            "collapses above ~1e-3."
        ),
    )
    ap.add_argument(
        "--mode",
        choices=["full", "idle"],
        default="full",
        help="Noise-injection mode for both levels.",
    )

    # Level-1 simulation controls. (The [[6,2,2]] block always runs exactly one
    # native syndrome-extraction round; there is no --rounds knob.)
    ap.add_argument("--max-shots-l1", type=int, default=300_000)
    ap.add_argument("--max-errors-l1", type=int, default=80)
    ap.add_argument("--batch-size-l1", type=int, default=20_000)

    # Level-2 simulation controls.
    ap.add_argument("--num-samples-l2", type=int, default=200_000)

    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--print-progress", action="store_true")
    ap.add_argument("--quick", action="store_true", help="Tiny smoke run for both levels.")
    ap.add_argument(
        "--output",
        default=None,
        help="CSV path (default: benchmarks/magic_h6_benchmark/results/h6_results.csv)",
    )
    args = ap.parse_args()

    if args.quick:
        args.p_values = [1e-2]
        args.p_values_l2 = [1e-3]
        args.max_shots_l1 = 20_000
        args.max_errors_l1 = 20
        args.batch_size_l1 = 5_000
        args.num_samples_l2 = 20_000

    out_path = (
        Path(args.output)
        if args.output
        else SCRIPT_DIR / "results" / "h6_results.csv"
    )

    print("=" * 64)
    print("Magic-H6 End-to-End Benchmark")
    print(f"levels         : {args.levels}")
    print(f"p_values (L1)  : {args.p_values}")
    print(f"p_values (L2)  : {args.p_values_l2}")
    print(f"mode           : {args.mode}")
    print(f"rounds (L1)    : 1 (fixed)")
    print(f"max_shots_l1   : {args.max_shots_l1}")
    print(f"max_errors_l1  : {args.max_errors_l1}")
    print(f"num_samples_l2 : {args.num_samples_l2}")
    print(f"num_workers    : {args.num_workers}")
    print(f"output         : {out_path}")
    print("=" * 64)

    if 1 in args.levels:
        _run_level1(args, out_path)
    if 2 in args.levels:
        _run_level2(args, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
