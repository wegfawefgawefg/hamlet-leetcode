from __future__ import annotations

"""
Speed test (safer) — finishes reliably by capping size/steps/time.

Changes vs speed_test.py (minimal):
- Add size caps (max_size) and step caps (max_steps).
- Reduce default time_budget_s to trip earlier.
- Optional max_total_time_s to stop whole run if it takes too long.
- Keep deterministic seeds.
"""

import time
import statistics
from typing import List, Tuple

from solver import solve_best, generate_board, default_densities
from games import SCENARIOS


History = List[Tuple[int, float]]


def evaluate(
    program,
    *,
    base_seed: int = 1234,
    non_dot_density: float = 0.5,
    time_budget_s: float = 1.0,
    boards_per_size: int = 1,
    start_size: int = 6,
    growth: float = 1.6,
    max_steps: int = 12,
    max_size: int = 120,
    max_total_time_s: float | None = 20.0,
) -> History:
    size = max(2, int(start_size))
    history: History = []
    t_start = time.perf_counter()

    for step in range(max_steps):
        if size > max_size:
            break
        if (
            max_total_time_s is not None
            and (time.perf_counter() - t_start) > max_total_time_s
        ):
            break

        times = []
        for i in range(boards_per_size):
            seed = base_seed + step * 1000 + i
            board = generate_board(size, size, default_densities(non_dot_density), seed)
            t0 = time.perf_counter()
            solve_best(board, program)
            times.append(time.perf_counter() - t0)
        avg = statistics.mean(times)
        history.append((size, avg))
        if avg >= time_budget_s:
            break

        # grow next size (still capped by max_size on next loop)
        size = max(size + 1, int(size * growth))

    return history


def main():
    # Print scenario solves (same as before)
    for sc in SCENARIOS:
        print(f"\n== {sc['name']} ==\n{sc['blurb']}")
        board = sc["board"]
        program = sc["program"]
        t0 = time.perf_counter()
        best_h, best_s = solve_best(board, program)
        took = time.perf_counter() - t0
        print(f"Scenario board solve: {took:.3f}s, score={best_s}")

    # Safer random stress — uses caps to finish
    print("\n== Random stress (program = scenarios[0]) ==")
    hist = evaluate(
        SCENARIOS[0]["program"],
        time_budget_s=1.0,  # trip earlier
        boards_per_size=1,
        start_size=6,
        growth=1.6,
        max_steps=12,
        max_size=120,  # hard cap on side length
        max_total_time_s=20.0,  # hard cap on whole loop
    )
    for sz, t in hist:
        print(f"  {sz:4d} -> {t:.3f}s")


if __name__ == "__main__":
    main()
