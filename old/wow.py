from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)
from collections import deque

"""
Generic Hamlet Solver (No circularity, supports negative terms)
---------------------------------------------------------------

This version:
- Allows arbitrary positive/negative scoring terms with optional caps.
- Predicates may be hamlet-invariant (fast path) or hamlet-dependent (falls
  back to exhaustive enumeration).
- Uses two strategies automatically:
    * Monotone-fast path: if all term weights >= 0 and predicates are
      hamlet-invariant, the best hamlet is the full non-empty connected
      component that maximizes the program score.
    * Exhaustive search: otherwise, enumerate all connected hamlets (subsets)
      per component with safe pruning (requirements feasibility + optimistic
      upper bounds for positive terms). This handles negative weights and any
      mix of terms.

Note: Exhaustive enumeration is exponential in component size; intended for
small/medium boards or for experimentation with complex constraints.
"""

Coord = Tuple[int, int]
Board = List[List[str]]
EMPTY = "."

# ----------------------------- Board helpers -----------------------------


def dims(board: Board) -> Tuple[int, int]:
    h = len(board)
    w = len(board[0]) if h else 0
    return w, h


def in_bounds(board: Board, x: int, y: int) -> bool:
    w, h = dims(board)
    return 0 <= x < w and 0 <= y < h


def get(board: Board, c: Coord) -> Optional[str]:
    x, y = c
    if not in_bounds(board, x, y):
        return None
    return board[y][x]


def neighbors4(x: int, y: int) -> Iterable[Coord]:
    yield (x - 1, y)
    yield (x + 1, y)
    yield (x, y - 1)
    yield (x, y + 1)


# ----------------------------- Components -----------------------------


def nonempty_components(board: Board) -> List[List[Coord]]:
    """Return list of components (each list of coords) of non-'.' cells."""
    w, h = dims(board)
    seen = [[False] * w for _ in range(h)]
    out: List[List[Coord]] = []

    for y in range(h):
        for x in range(w):
            if board[y][x] == EMPTY or seen[y][x]:
                continue
            comp: List[Coord] = []
            dq = deque([(x, y)])
            seen[y][x] = True
            while dq:
                cx, cy = dq.popleft()
                comp.append((cx, cy))
                for nx, ny in neighbors4(cx, cy):
                    if (
                        in_bounds(board, nx, ny)
                        and not seen[ny][nx]
                        and board[ny][nx] != EMPTY
                    ):
                        seen[ny][nx] = True
                        dq.append((nx, ny))
            out.append(comp)
    return out


# ----------------------------- Predicates -----------------------------
# A predicate decides whether a tile should be counted for a term.
# If `ham_invariant=True`, the predicate MUST NOT depend on the hamlet set.
Predicate = Callable[[Board, Coord, FrozenSet[Coord]], bool]


def always_true(_: Board, __: Coord, ___: FrozenSet[Coord]) -> bool:
    return True


def is_adjacent_to_empty(board: Board, c: Coord, _: FrozenSet[Coord]) -> bool:
    """True if any 4-neighbor on the board is '.' (hamlet-invariant)."""
    x, y = c
    for nx, ny in neighbors4(x, y):
        if in_bounds(board, nx, ny) and board[ny][nx] == EMPTY:
            return True
    return False


# ----------------------------- Constraints / Terms -----------------------------
@dataclass(frozen=True)
class RequireAtLeast:
    symbol: str
    n: int = 1

    def ok(self, board: Board, ham: FrozenSet[Coord]) -> bool:
        cnt = 0
        for c in ham:
            if get(board, c) == self.symbol:
                cnt += 1
                if cnt >= self.n:
                    return True
        return False


@dataclass(frozen=True)
class CountTerm:
    symbol: str
    weight: float = 1.0  # may be negative
    cap: Optional[int] = None  # None means unlimited
    predicate: Predicate = always_true
    ham_invariant: bool = True  # set False if predicate depends on `ham`

    def count(self, board: Board, ham: FrozenSet[Coord]) -> int:
        cnt = 0
        for c in ham:
            if get(board, c) == self.symbol and self.predicate(board, c, ham):
                cnt += 1
        return cnt

    def score(self, board: Board, ham: FrozenSet[Coord]) -> float:
        cnt = self.count(board, ham)
        if self.cap is not None:
            cnt = min(cnt, self.cap)
        return self.weight * cnt


@dataclass
class ScoreProgram:
    requires: List[RequireAtLeast]
    terms: List[CountTerm]

    def valid(self, board: Board, ham: FrozenSet[Coord]) -> bool:
        return all(req.ok(board, ham) for req in self.requires)

    def score(self, board: Board, ham: FrozenSet[Coord]) -> float:
        return sum(t.score(board, ham) for t in self.terms)

    def monotone_fast_path_ok(self) -> bool:
        """Safe only when all terms are nonnegative and ham-invariant.
        In that case, adding cells never decreases score, and requirements are
        monotone, so the best hamlet within a component is the full component.
        """
        for t in self.terms:
            if t.weight < 0:
                return False
            if not t.ham_invariant:
                return False
        return True


# ----------------------------- Solvers -----------------------------


def solve_best(
    board: Board, program: ScoreProgram
) -> Tuple[Optional[FrozenSet[Coord]], float]:
    comps = nonempty_components(board)
    if program.monotone_fast_path_ok():
        return _solve_best_components_fast(board, program, comps)
    else:
        return _solve_best_exhaustive(board, program, comps)


def _solve_best_components_fast(
    board: Board, program: ScoreProgram, comps: List[List[Coord]]
) -> Tuple[Optional[FrozenSet[Coord]], float]:
    best_h: Optional[FrozenSet[Coord]] = None
    best_s: float = float("-inf")

    for comp in comps:
        ham = frozenset(comp)
        if not program.valid(board, ham):
            continue
        s = program.score(board, ham)
        if s > best_s:
            best_s, best_h = s, ham
    if best_h is None:
        return None, 0.0
    return best_h, best_s


# ------- Exhaustive connected-subset enumeration with safe pruning -------


def _solve_best_exhaustive(
    board: Board, program: ScoreProgram, comps: List[List[Coord]]
) -> Tuple[Optional[FrozenSet[Coord]], float]:
    best_h: Optional[FrozenSet[Coord]] = None
    best_s: float = float("-inf")

    for comp in comps:
        if not comp:
            continue
        # Indexing for canonical enumeration
        nodes = sorted(comp)  # deterministic
        index_of: Dict[Coord, int] = {c: i for i, c in enumerate(nodes)}
        N = len(nodes)
        # adjacency restricted to component
        adj: List[List[int]] = [[] for _ in range(N)]
        for i, (x, y) in enumerate(nodes):
            for nx, ny in neighbors4(x, y):
                if in_bounds(board, nx, ny) and board[ny][nx] != EMPTY:
                    j = index_of.get((nx, ny))
                    if j is not None:
                        adj[i].append(j)

        # Precompute symbol labels for each node
        labels: List[str] = [board[y][x] for (x, y) in nodes]

        # For each root, enumerate subsets whose minimum index is that root
        for r in range(N):
            # Pool counts among indices >= r for feasibility checks and UB
            pool_indices = list(range(r, N))
            pool_symbol_counts: Dict[str, int] = {}
            for i in pool_indices:
                pool_symbol_counts[labels[i]] = pool_symbol_counts.get(labels[i], 0) + 1

            # If any requirement impossible within this pool, skip this root entirely
            impossible = False
            for req in program.requires:
                if pool_symbol_counts.get(req.symbol, 0) < req.n:
                    impossible = True
                    break
            if impossible:
                continue

            # For positive-weight ham-invariant terms, precompute pool counts filtered by predicate
            pos_terms = [t for t in program.terms if t.weight > 0 and t.ham_invariant]
            pos_pool_counts: Dict[int, int] = {}
            for ti, t in enumerate(pos_terms):
                cnt = 0
                for i in pool_indices:
                    if labels[i] == t.symbol and t.predicate(
                        board, nodes[i], frozenset()
                    ):
                        cnt += 1
                pos_pool_counts[ti] = cnt

            # DFS state
            cur_set: List[int] = [r]
            in_set = [False] * N
            in_set[r] = True
            cur_counts: Dict[str, int] = {labels[r]: 1}
            # current term counts (post-predicate), for quick scoring and UBs
            cur_term_counts: List[int] = []
            for t in program.terms:
                # Note: if predicate depends on ham, recompute at score time; here we assume ham-invariant for speed
                if t.ham_invariant and t.predicate(board, nodes[r], frozenset()):
                    cur_term_counts.append(1 if labels[r] == t.symbol else 0)
                else:
                    cur_term_counts.append(
                        1
                        if (
                            labels[r] == t.symbol
                            and t.predicate(board, nodes[r], frozenset())
                        )
                        else 0
                    )

            # frontier (set for dedup) of neighbors with index >= r
            frontier_set = set(j for j in adj[r] if j >= r and not in_set[j])
            frontier = sorted(frontier_set)

            def score_current(ham_indices: Sequence[int]) -> float:
                ham = frozenset(nodes[i] for i in ham_indices)
                if not program.valid(board, ham):
                    return float("-inf")
                return program.score(board, ham)

            # Evaluate singleton immediately
            best_candidate_score = score_current(cur_set)
            if best_candidate_score > best_s:
                best_s = best_candidate_score
                best_h = frozenset(nodes[i] for i in cur_set)

            # backtracking
            def dfs(frontier: List[int], cur_set: List[int], in_set: List[bool]):
                nonlocal best_s, best_h

                if not frontier:
                    return

                # Try each frontier node (include-or-skip branching)
                # We enforce canonical generation by only adding nodes with index >= r
                for k in range(len(frontier)):
                    v = frontier[k]
                    if in_set[v]:
                        continue

                    # Branch 1: include v
                    in_set[v] = True
                    cur_set.append(v)

                    # Update frontier: remove v and add its neighbors
                    new_frontier_set = set(
                        frontier[k + 1 :]
                    )  # all later candidates still allowed
                    for w in adj[v]:
                        if w >= r and not in_set[w]:
                            new_frontier_set.add(w)
                    new_frontier = sorted(new_frontier_set)

                    # Score/prune
                    s_val = score_current(cur_set)
                    if s_val > best_s:
                        best_s = s_val
                        best_h = frozenset(nodes[i] for i in cur_set)

                    # Optimistic UB pruning for positive ham-invariant terms
                    ub_extra = 0.0
                    if pos_terms:
                        # Count how many of each positive term we've already taken among indices >= r
                        # and how many remain in the pool (regardless of connectivity)
                        taken_counts: Dict[int, int] = {}
                        for ti, t in enumerate(pos_terms):
                            taken = 0
                            for i in cur_set:
                                if labels[i] == t.symbol and t.predicate(
                                    board, nodes[i], frozenset()
                                ):
                                    taken += 1
                            remaining = pos_pool_counts[ti] - taken
                            if t.cap is not None:
                                need = max(0, min(t.cap, pos_pool_counts[ti]) - taken)
                                ub_extra += t.weight * need
                            else:
                                ub_extra += t.weight * max(0, remaining)

                    if s_val + ub_extra > best_s:
                        dfs(new_frontier, cur_set, in_set)

                    # Backtrack: exclude v from current set
                    cur_set.pop()
                    in_set[v] = False

                    # Branch 2: skip v (do nothing and continue to next)
                    # Note: skipping may lose connectivity to some nodes; that's fine – they may
                    # re-enter via other frontier nodes added later.
                    # No extra pruning here; continue loop.

            dfs(frontier, cur_set, in_set)

    if best_h is None:
        return None, 0.0
    return best_h, best_s


# ----------------------------- Example / Original Program -----------------------------
ORIGINAL_BOARD = [
    ["W", "G", ".", "F", ".", "W", "."],
    [".", "C", ".", "C", ".", ".", "C"],
    [".", ".", "D", ".", "G", "D", "F"],
    ["G", "F", "W", ".", "F", "W", "."],
    [".", ".", ".", "W", ".", ".", "."],
    ["G", "C", ".", "W", ".", "C", "W"],
    ["F", ".", "C", ".", "F", "D", "F"],
    [".", "W", "F", "G", ".", "G", "."],
]

ORIGINAL_PROGRAM = ScoreProgram(
    requires=[
        RequireAtLeast("W", 1),
        RequireAtLeast("G", 1),
    ],
    terms=[
        CountTerm("W", weight=1.0, cap=1, predicate=always_true, ham_invariant=True),
        CountTerm("G", weight=1.0, cap=2, predicate=always_true, ham_invariant=True),
        CountTerm("D", weight=1.0, predicate=is_adjacent_to_empty, ham_invariant=True),
        CountTerm("C", weight=1.0, cap=1, predicate=always_true, ham_invariant=True),
        CountTerm("F", weight=1.0, predicate=always_true, ham_invariant=True),
        # Negative example (uranium penalty): CountTerm('U', weight=-1.0)
    ],
)


if __name__ == "__main__":
    best_h, best_s = solve_best(ORIGINAL_BOARD, ORIGINAL_PROGRAM)
    if best_h is None:
        print("No valid hamlet found.")
    else:
        # pretty print bounding box of best hamlet
        xs = [x for x, _ in best_h]
        ys = [y for _, y in best_h]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        for y in range(min_y, max_y + 1):
            row = []
            for x in range(min_x, max_x + 1):
                ch = ORIGINAL_BOARD[y][x]
                row.append(" " if ch == EMPTY else ch)
            print("".join(row))
        print(f"score of {best_s}")
