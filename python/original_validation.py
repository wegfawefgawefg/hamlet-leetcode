from __future__ import annotations

"""
Validation script: run the new generic solver on the original board/rules
and print the result for quick eyeballing.
"""

from solver import solve_best, viz
from games import ORIGINAL_BOARD, ORIGINAL_PROGRAM


def main():
    best_h, best_s = solve_best(ORIGINAL_BOARD, ORIGINAL_PROGRAM)
    if best_h is None:
        print("No valid hamlet found.")
        return
    print(viz(ORIGINAL_BOARD, best_h))
    print(f"score of {best_s}")


if __name__ == "__main__":
    main()
