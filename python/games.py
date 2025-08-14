from __future__ import annotations

"""
Game/scenario definitions for the predicate-agnostic solver.

Each scenario is a dict with keys:
- name: str
- board: Board
- program: ScoreProgram
- blurb: str

You can add more symbols. Legend (commonly used):
  . barren, W water, G game, C cave, D defense, F fertile,
  U uranium, S swamp, R raiders, V volcano, M mine
"""

from solver import (
    Board,
    ScoreProgram,
    RequireAtLeast,
    CountTerm,
    adj_to_symbol_in_world,
    adj_to_symbol_in_ham,
    not_adjacent_to_symbol_in_ham,
    ham_contains_symbol,
    ham_lacks_symbol,
    on_ham_perimeter,
)

# ----------------------------- Original problem -----------------------------
ORIGINAL_BOARD: Board = [
    ["W", "G", ".", "F", ".", "W", "."],
    [".", "C", ".", "C", ".", ".", "C"],
    [".", ".", "D", ".", "G", "D", "F"],
    ["G", "F", "W", ".", "F", "W", "."],
    [".", ".", ".", "W", ".", ".", "."],
    ["G", "C", ".", "W", ".", "C", "W"],
    ["F", ".", "C", ".", "F", "D", "F"],
    [".", "W", "F", "G", ".", "G", "."],
]

# Matches the original rules exactly (no circularity):
ORIGINAL_PROGRAM = ScoreProgram(
    requires=[
        RequireAtLeast("W", 1),
        RequireAtLeast("G", 1),
    ],
    terms=[
        CountTerm("W", weight=1.0, cap=1),  # presence bonus
        CountTerm("G", weight=1.0, cap=2),
        CountTerm(
            "D", weight=1.0, predicate=adj_to_symbol_in_world(".")
        ),  # boundary vs world '.'
        CountTerm("C", weight=1.0, cap=1),
        CountTerm("F", weight=1.0),
    ],
)

# ----------------------------- New hamlet-dependent scenarios -----------------------------

SCENARIOS = [
    {
        "name": "Irrigated Farms (hamlet-adj) vs Swamp",
        "blurb": (
            "Farms only count if the hamlet itself places them next to Water. Swamps are negative."
        ),
        "board": [
            [".", ".", "W", "F", ".", ".", "."],
            [".", "F", "F", "F", "S", "S", "."],
            [".", ".", "W", ".", "F", ".", "."],
            [".", "F", "F", "F", "S", ".", "."],
            [".", ".", ".", ".", ".", ".", "."],
        ],
        "program": ScoreProgram(
            requires=[RequireAtLeast("W", 1)],
            terms=[
                CountTerm(
                    "F", weight=1.0, predicate=adj_to_symbol_in_ham("W")
                ),  # hamlet-based irrigation
                CountTerm("W", weight=1.0, cap=2),
                CountTerm("S", weight=-1.5),
            ],
        ),
    },
    {
        "name": "Volcanic Mining",
        "blurb": (
            "Mines count only if adjacent to a Volcano **in the hamlet**, and they become invalid if the hamlet contains any Water at all."
        ),
        "board": [
            [".", ".", "V", ".", ".", ".", "."],
            [".", "M", "M", "V", "M", "M", "."],
            [".", ".", "M", ".", "M", ".", "."],
            [".", ".", ".", ".", ".", ".", "."],
        ],
        "program": ScoreProgram(
            requires=[],
            terms=[
                # Count M only if touching V in ham AND hamlet has no Water anywhere
                CountTerm(
                    "M",
                    weight=2.0,
                    predicate=lambda b, c, h: adj_to_symbol_in_ham("V")(b, c, h)
                    and ham_lacks_symbol("W")(b, c, h),
                ),
                CountTerm("V", weight=0.5, cap=1),  # small bonus for having a V at all
                CountTerm("W", weight=-2.0),  # water is actively bad for this economy
            ],
        ),
    },
    {
        "name": "Perimeter Forts vs Raiders",
        "blurb": (
            "Defenses only count when placed on the **hamlet perimeter**; Raider camps inside the hamlet are heavy penalties."
        ),
        "board": [
            [".", "D", "D", "D", ".", "D", "D", "D", "."],
            ["D", "F", "F", ".", "R", ".", "F", "F", "D"],
            ["D", "F", "G", "F", "W", "F", "G", "F", "D"],
            [".", ".", "D", ".", ".", ".", "D", ".", "."],
        ],
        "program": ScoreProgram(
            requires=[RequireAtLeast("W", 1)],
            terms=[
                CountTerm("D", weight=2.0, predicate=on_ham_perimeter()),
                CountTerm("G", weight=1.0, cap=2),
                CountTerm("F", weight=1.0),
                CountTerm("R", weight=-2.5),
            ],
        ),
    },
    {
        "name": "Crystal Cavern with Uranium",
        "blurb": (
            "Tempting crystal cave (C) surrounded by pockets of Uranium (U). A small crescent hamlet that just touches C and W without U beats the big cluster."
        ),
        "board": [
            [".", ".", ".", "D", ".", ".", ".", "."],
            [".", "F", "F", "C", "F", "U", "F", "."],
            [".", "F", "U", "W", "F", "F", "F", "."],
            ["D", "F", "F", "W", "F", "U", "F", "."],
            [".", "F", "U", "W", "F", "F", "G", "."],
            [".", ".", ".", "D", ".", ".", ".", "."],
        ],
        "program": ScoreProgram(
            requires=[RequireAtLeast("W", 1)],
            terms=[
                CountTerm("C", weight=3.0, cap=1),
                CountTerm("F", weight=1.0, cap=2),
                CountTerm("W", weight=1.0, cap=1),
                CountTerm("G", weight=1.0, cap=2),
                CountTerm("D", weight=1.0, predicate=adj_to_symbol_in_world(".")),
                CountTerm("U", weight=-2.5),
            ],
        ),
    },
]
