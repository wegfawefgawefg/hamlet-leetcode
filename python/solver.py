from __future__ import annotations

"""
Generic Hamlet Solver (predicate-agnostic, hamlet-dependent safe)
-----------------------------------------------------------------

- No ham_invariant/ham-dependent flags. Every predicate receives (board, coord, ham)
  and may use the hamlet arbitrarily.
- Solver ALWAYS works for any predicates/weights, using connected-subset
  backtracking per component with safe pruning (requirements feasibility +
  optimistic upper bounds for positive terms). Correctness > speed.
- Includes handy predicate builders for common game rules (adjacency, perimeter,
  forbid/require symbol presence in hamlet, etc.).

API
---
- ScoreProgram(requires, terms)
- RequireAtLeast(symbol: str, n: int)
- CountTerm(symbol: str, weight: float = 1.0, cap: int | None = None,
            predicate: Predicate | None = None)
- solve_best(board: Board, program: ScoreProgram) -> (best_hamlet: frozenset[Coord] | None, score: float)
- helpers: neighbors4, generate_board(...), predicate builders
"""

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple
from collections import deque
import random

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


# ----------------------------- Predicates -----------------------------
Predicate = Callable[[Board, Coord, FrozenSet[Coord]], bool]


def always_true(_: Board, __: Coord, ___: FrozenSet[Coord]) -> bool:
    return True


def adj_to_symbol_in_world(sym: str) -> Predicate:
    """True if any world-neighbor is `sym` (board-only condition)."""

    def pred(board: Board, c: Coord, ham: FrozenSet[Coord]) -> bool:
        x, y = c
        for nx, ny in neighbors4(x, y):
            if in_bounds(board, nx, ny) and board[ny][nx] == sym:
                return True
        return False

    return pred


def adj_to_symbol_in_ham(sym: str) -> Predicate:
    """True if any hamlet neighbor is `sym`."""

    def pred(board: Board, c: Coord, ham: FrozenSet[Coord]) -> bool:
        x, y = c
        for nx, ny in neighbors4(x, y):
            if (nx, ny) in ham and board[ny][nx] == sym:
                return True
        return False

    return pred


def not_adjacent_to_symbol_in_ham(sym: str) -> Predicate:
    def pred(board: Board, c: Coord, ham: FrozenSet[Coord]) -> bool:
        x, y = c
        for nx, ny in neighbors4(x, y):
            if (nx, ny) in ham and board[ny][nx] == sym:
                return False
        return True

    return pred


def ham_contains_symbol(sym: str) -> Predicate:
    """Cell counts only if the hamlet contains `sym` somewhere (global dependency)."""

    def pred(board: Board, c: Coord, ham: FrozenSet[Coord]) -> bool:
        for x, y in ham:
            if board[y][x] == sym:
                return True
        return False

    return pred


def ham_lacks_symbol(sym: str) -> Predicate:
    """Cell counts only if the hamlet contains NO `sym`."""

    def pred(board: Board, c: Coord, ham: FrozenSet[Coord]) -> bool:
        for x, y in ham:
            if board[y][x] == sym:
                return False
        return True

    return pred


def on_ham_perimeter() -> Predicate:
    """True if at least one 4-neighbor is outside the hamlet (within bounds)."""

    def pred(board: Board, c: Coord, ham: FrozenSet[Coord]) -> bool:
        x, y = c
        for nx, ny in neighbors4(x, y):
            if in_bounds(board, nx, ny) and (nx, ny) not in ham:
                return True
        return False

    return pred


# ----------------------------- Constraints / Terms -----------------------------
@dataclass(frozen=True)
class RequireAtLeast:
    symbol: str
    n: int = 1

    def ok(self, board: Board, ham: FrozenSet[Coord]) -> bool:
        cnt = 0
        for x, y in ham:
            if board[y][x] == self.symbol:
                cnt += 1
                if cnt >= self.n:
                    return True
        return False


@dataclass(frozen=True)
class CountTerm:
    symbol: str
    weight: float = 1.0  # may be negative
    cap: Optional[int] = None  # None means unlimited
    predicate: Optional[Predicate] = None

    def score(self, board: Board, ham: FrozenSet[Coord]) -> float:
        # Full recompute (generic but slower): evaluate predicate per cell in hamlet
        cnt = 0
        pred = self.predicate or always_true
        for x, y in ham:
            if board[y][x] == self.symbol and pred(board, (x, y), ham):
                cnt += 1
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


# ----------------------------- Solver -----------------------------


def solve_best(
    board: Board, program: ScoreProgram
) -> Tuple[Optional[FrozenSet[Coord]], float]:
    comps = _nonempty_components(board)
    best_h: Optional[FrozenSet[Coord]] = None
    best_s: float = float("-inf")
    for comp in comps:
        h, s = _solve_component_enum(board, program, comp)
        if h is not None and s > best_s:
            best_h, best_s = h, s
    if best_h is None:
        return None, 0.0
    return best_h, best_s


def _nonempty_components(board: Board) -> List[List[Coord]]:
    w, h = dims(board)
    seen = [[False] * w for _ in range(h)]
    out: List[List[Coord]] = []
    for y in range(h):
        for x in range(w):
            if seen[y][x] or board[y][x] == EMPTY:
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


def _solve_component_enum(
    board: Board, program: ScoreProgram, comp: List[Coord]
) -> Tuple[Optional[FrozenSet[Coord]], float]:
    if not comp:
        return None, float("-inf")

    nodes = sorted(comp)  # deterministic order
    N = len(nodes)
    index_of: Dict[Coord, int] = {c: i for i, c in enumerate(nodes)}
    adj: List[List[int]] = [[] for _ in range(N)]
    labels: List[str] = [board[y][x] for (x, y) in nodes]

    # adjacency restricted to component
    for i, (x, y) in enumerate(nodes):
        for nx, ny in neighbors4(x, y):
            j = index_of.get((nx, ny))
            if j is not None:
                adj[i].append(j)

    # pre-count symbol totals in component (for feasibility/UB)
    total_sym: Dict[str, int] = {}
    for s in labels:
        total_sym[s] = total_sym.get(s, 0) + 1

    best_h: Optional[FrozenSet[Coord]] = None
    best_s: float = float("-inf")

    # For each canonical root r, enumerate all connected subsets whose min index is r
    for r in range(N):
        # If pool from r..N-1 can't satisfy requirements at all, skip root
        pool_total: Dict[str, int] = {}
        for i in range(r, N):
            s = labels[i]
            pool_total[s] = pool_total.get(s, 0) + 1
        feasible = True
        for req in program.requires:
            if pool_total.get(req.symbol, 0) < req.n:
                feasible = False
                break
        if not feasible:
            continue

        in_set = [False] * N
        in_set[r] = True
        cur_list: List[int] = [r]
        cur_sym: Dict[str, int] = {labels[r]: 1}

        # initial frontier: neighbors of r with index >= r
        frontier = sorted({j for j in adj[r] if j >= r})

        def current_ham() -> FrozenSet[Coord]:
            return frozenset(nodes[i] for i in cur_list)

        def current_score() -> float:
            return program.score(board, current_ham())

        # seed best with singleton
        s0 = current_score()
        if program.valid(board, current_ham()):
            if s0 > best_s:
                best_s, best_h = s0, current_ham()

        # helper: optimistic UB for positive terms using symbol totals only
        def optimistic_ub(current_score_val: float) -> float:
            ub = current_score_val
            # remaining symbol counts in the whole component (loose but safe)
            remaining_sym: Dict[str, int] = {}
            for sym, tot in total_sym.items():
                taken = cur_sym.get(sym, 0)
                remaining_sym[sym] = max(0, tot - taken)
            for t in program.terms:
                if t.weight > 0:
                    # current term count (recompute using ham) for accuracy
                    # (still safe even if cap applies)
                    cur_term = (
                        t.score(board, current_ham()) / t.weight if t.weight != 0 else 0
                    )
                    if t.cap is not None:
                        needed = max(0, t.cap - int(cur_term))
                    else:
                        needed = float("inf")
                    addable = remaining_sym.get(t.symbol, 0)
                    gain = min(needed, addable)
                    if gain == float("inf"):
                        # no cap, assume all remaining symbols could qualify
                        gain = addable
                    ub += t.weight * gain
            return ub

        def requirements_possible() -> bool:
            for req in program.requires:
                have = cur_sym.get(req.symbol, 0)
                avail = total_sym.get(req.symbol, 0) - have
                if have + avail < req.n:
                    return False
            return True

        def dfs(front: List[int]):
            nonlocal best_s, best_h
            if not front:
                return
            # try each candidate (include branch then skip branch)
            for idx in range(len(front)):
                v = front[idx]
                if in_set[v]:
                    continue

                # include v
                in_set[v] = True
                cur_list.append(v)
                sym_v = labels[v]
                cur_sym[sym_v] = cur_sym.get(sym_v, 0) + 1

                # new frontier: later candidates plus neighbors of v
                new_front = set(front[idx + 1 :])
                for w in adj[v]:
                    if w >= r and not in_set[w]:
                        new_front.add(w)
                new_front = sorted(new_front)

                # prune by requirements feasibility (loose but safe)
                if requirements_possible():
                    cs = current_score()
                    if program.valid(board, current_ham()) and cs > best_s:
                        best_s, best_h = cs, current_ham()
                    # optimistic UB pruning
                    if optimistic_ub(cs) > best_s:
                        dfs(new_front)

                # backtrack include
                cur_sym[sym_v] -= 1
                if cur_sym[sym_v] == 0:
                    del cur_sym[sym_v]
                cur_list.pop()
                in_set[v] = False
                # skip v branch is implicit by moving to next loop iteration

        dfs(frontier)

    return best_h, best_s


# ----------------------------- Generation helpers -----------------------------
LETTERS = ["W", "G", "C", "D", "F", "U", "S", "R", "V", "M"]  # extend as desired


def default_densities(non_dot_density: float = 0.5) -> Dict[str, float]:
    non_dot = max(0.0, min(1.0, non_dot_density))
    empty_p = 1.0 - non_dot
    per = non_dot / len(LETTERS) if LETTERS else 0.0
    d = {EMPTY: empty_p}
    d.update({ch: per for ch in LETTERS})
    return d


def normalize_densities(densities: Dict[str, float]) -> List[Tuple[str, float]]:
    allowed = set(LETTERS + [EMPTY])
    filtered = {k: max(0.0, v) for k, v in densities.items() if k in allowed}
    if EMPTY not in filtered:
        filtered[EMPTY] = 0.0
    total = sum(filtered.values()) or 1.0
    cum = 0.0
    cdf: List[Tuple[str, float]] = []
    for sym in [EMPTY] + LETTERS:
        p = filtered.get(sym, 0.0) / total
        cum += p
        cdf.append((sym, cum))
    cdf[-1] = (cdf[-1][0], 1.0)
    return cdf


def generate_board(
    width: int, height: int, densities: Optional[Dict[str, float]] = None, seed: int = 0
) -> Board:
    if densities is None:
        densities = default_densities(0.5)
    cdf = normalize_densities(densities)
    rng = random.Random(seed)

    def sample_symbol() -> str:
        r = rng.random()
        for sym, cum in cdf:
            if r <= cum:
                return sym
        return cdf[-1][0]

    return [[sample_symbol() for _ in range(width)] for _ in range(height)]


# ----------------------------- Pretty printing -----------------------------


def viz(board: Board, ham: Sequence[Coord]) -> str:
    if not ham:
        return ""
    xs = [x for x, _ in ham]
    ys = [y for _, y in ham]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    out: List[str] = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            ch = board[y][x]
            out.append(" " if ch == EMPTY else ch)
        out.append("\n")
    return "".join(out)
