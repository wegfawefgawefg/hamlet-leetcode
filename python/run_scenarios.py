from __future__ import annotations

"""
Run all scenarios (including the ORIGINAL problem first) and print boards with
the chosen hamlet highlighted in green.

Usage:
    python3 run_scenarios_v2.py

Copy `viz_in_board` into solver.py if you want it available elsewhere.
"""

from typing import Iterable, Tuple, Set

from solver import solve_best, dims
from games import SCENARIOS, ORIGINAL_BOARD, ORIGINAL_PROGRAM

Coord = Tuple[int, int]

# ----------------------------- Pretty, in-board visualization -----------------------------
ANSI_COLORS = {
    "green": "\x1b[32m",
    "bold": "\x1b[1m",
    "reset": "\x1b[0m",
}


def viz_in_board(
    board, ham: Iterable[Coord], *, color: str = "green", show_coords: bool = True
) -> str:
    """Return a string of the *original* board with hamlet cells colored.

    - Only non-barren hamlet cells are colorized; '.' stays plain for readability.
    - Uses ANSI escapes. If your terminal doesn't support ANSI, you'll see raw codes.
    - `show_coords=True` prints 0–9 repeating indices along top/left for orientation.
    """
    w, h = dims(board)
    ham_set: Set[Coord] = set(ham)
    col = ANSI_COLORS.get(color, "")
    reset = ANSI_COLORS["reset"]

    rows = []
    if show_coords:
        header = ["   "] + [f"{x % 10}" for x in range(w)]
        rows.append(" ".join(header))
    for y in range(h):
        line_cells = []
        for x in range(w):
            ch = board[y][x]
            if (x, y) in ham_set and ch != ".":
                # bold green letter for hamlet tiles
                cell = f"{ANSI_COLORS['bold']}{col}{ch}{reset}"
            else:
                cell = ch
            line_cells.append(cell)
        if show_coords:
            rows.append(f"{y % 10:>2} " + " ".join(line_cells))
        else:
            rows.append(" ".join(line_cells))
    return "\n".join(rows)


# ----------------------------- Runner -----------------------------


def main():
    # Prepend the ORIGINAL problem as a scenario at the top
    scenarios = [
        {
            "name": "ORIGINAL Problem",
            "blurb": "Original rules; sanity check that the generic solver matches.",
            "board": ORIGINAL_BOARD,
            "program": ORIGINAL_PROGRAM,
        }
    ] + SCENARIOS

    for sc in scenarios:
        name = sc["name"]
        blurb = sc["blurb"]
        board = sc["board"]
        program = sc["program"]

        best_h, best_s = solve_best(board, program)
        print("\n==", name, "==")
        print(blurb)

        if best_h is None:
            print("No valid hamlet found.")
            continue

        print(f"score = {best_s}")
        print(viz_in_board(board, best_h, color="green", show_coords=True))
        # Also print the hamlet coordinate list (sorted) for debugging:
        coords = sorted(list(best_h))
        print("hamlet coords:", coords)


if __name__ == "__main__":
    main()
