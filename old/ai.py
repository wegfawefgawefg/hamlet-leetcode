from __future__ import annotations

import random
import time
import statistics
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Coord = Tuple[int, int]
Board = List[List[str]]

LETTERS = ["W", "G", "C", "D", "F"]  # non-empty tiles
EMPTY = "."


# ---------- Core helpers (cleaned; no debug prints) ----------


def filter_type(board: Board, points: Iterable[Coord], query: str) -> List[Coord]:
    return [p for p in points if get(board, p) == query]


def get_dims(board: Board) -> Tuple[int, int]:
    height = len(board)
    width = len(board[0]) if height else 0
    return (width, height)


def get(board: Board, coord: Coord) -> Optional[str]:
    x, y = coord
    width, height = get_dims(board)
    if not (0 <= x < width and 0 <= y < height):
        return None
    return board[y][x]


def moore(coord: Coord, dims: Tuple[int, int]) -> List[Coord]:
    # 4-neighborhood (von Neumann), keeping original name
    x, y = coord
    width, height = dims
    out: List[Coord] = []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            out.append((nx, ny))
    return out


def is_on_boundary(coord: Coord, board: Board) -> bool:
    dims = get_dims(board)
    for n in moore(coord, dims):
        if get(board, n) == EMPTY:
            return True
    return False


def score(hamlet: Sequence[Coord], board: Board) -> int:
    waters = filter_type(board, hamlet, "W")
    games = filter_type(board, hamlet, "G")
    caves = filter_type(board, hamlet, "C")
    defenses = filter_type(board, hamlet, "D")
    fertiles = filter_type(board, hamlet, "F")

    s = 0
    s += 0 if len(waters) == 0 else 1
    s += min(len(games), 2)

    # defense: only count those on a boundary
    s += sum(1 for d in defenses if is_on_boundary(d, board))

    # cave: max 1 benefit
    if len(caves) >= 1:
        s += 1

    # fertiles: all count
    s += len(fertiles)
    return s


def find_hamlets_exhaustively(board: Board) -> set[Tuple[Coord, ...]]:
    """
    Enumerates all distinct hamlets (connected sets of non-'.' cells).
    Note: This can explode combinatorially for large boards.
    """
    distinct_hamlets: set[Tuple[Coord, ...]] = set()
    dims = get_dims(board)
    width, height = dims

    # seeds: every coordinate
    for y in range(height):
        for x in range(width):
            seed = (x, y)
            if get(board, seed) == EMPTY:
                continue

            # singleton hamlet
            distinct_hamlets.add((seed,))

            # DFS-ish growth while deduping by sorted tuple
            growing_hamlet: List[Coord] = [seed]
            checked: set[Coord] = {seed}
            search_stack: List[Coord] = moore(seed, dims)

            while search_stack:
                n = search_stack.pop()
                if n in checked:
                    continue
                checked.add(n)

                if get(board, n) == EMPTY:
                    continue

                growing_hamlet.append(n)
                frozen = tuple(sorted(growing_hamlet))
                if frozen in distinct_hamlets:
                    continue
                distinct_hamlets.add(frozen)
                search_stack.extend(moore(n, dims))

    return distinct_hamlets


def viz(hamlet: Sequence[Coord], board: Board) -> str:
    if not hamlet:
        return ""
    ps = list(hamlet)
    xs = [x for x, _ in ps]
    ys = [y for _, y in ps]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    chars: List[str] = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            t = get(board, (x, y))
            chars.append(" " if (t is None or t == EMPTY) else t)
        chars.append("\n")
    return "".join(chars)


# ---------- Board generator (seed-deterministic) ----------


def default_densities(non_dot_density: float = 0.5) -> Dict[str, float]:
    """
    Returns a density map with `.` at (1 - non_dot_density) and the five letters
    sharing `non_dot_density` evenly.
    """
    non_dot = max(0.0, min(1.0, non_dot_density))
    empty_p = 1.0 - non_dot
    per = non_dot / len(LETTERS) if LETTERS else 0.0
    d = {EMPTY: empty_p}
    d.update({ch: per for ch in LETTERS})
    return d


def normalize_densities(densities: Dict[str, float]) -> List[Tuple[str, float]]:
    """
    Normalize and return as cumulative distribution: [(symbol, cum_p), ...]
    Unknown symbols are ignored; missing ones become 0. Ensures '.' present.
    """
    allowed = set(LETTERS + [EMPTY])
    filtered = {k: max(0.0, v) for k, v in densities.items() if k in allowed}
    if EMPTY not in filtered:
        filtered[EMPTY] = 0.0
    total = sum(filtered.values()) or 1.0
    cum = 0.0
    cdf: List[Tuple[str, float]] = []
    for sym in [EMPTY] + LETTERS:  # enforce deterministic order
        p = filtered.get(sym, 0.0) / total
        cum += p
        cdf.append((sym, cum))
    cdf[-1] = (cdf[-1][0], 1.0)  # avoid rounding gaps
    return cdf


def generate_board(
    width: int,
    height: int,
    densities: Optional[Dict[str, float]] = None,
    seed: int = 0,
) -> Board:
    """
    Make a width x height board with probabilities given by `densities`.
    Deterministic for the same seed. If densities is None, defaults to 50% non-dot.
    """
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


# ---------- Algo A (bottled) + stubs for B/C ----------


def algo_a(board: Board) -> Tuple[Optional[Tuple[Coord, ...]], int]:
    all_unique_hamlets = find_hamlets_exhaustively(board)
    if not all_unique_hamlets:
        return None, 0
    best_h, best_s = None, -(10**9)
    for h in all_unique_hamlets:
        s = score(h, board)
        if s > best_s:
            best_h, best_s = h, s
    return best_h, best_s


def algo_b(board: Board) -> Tuple[Optional[Tuple[Coord, ...]], int]:
    """
    Linear-ish pass: find connected components of non-'.' cells, score each component,
    and return the highest-scoring one. Since the score is monotonic w.r.t. inclusion,
    the optimal hamlet per component is the full component.
    """
    width, height = get_dims(board)
    if width == 0 or height == 0:
        return None, 0

    visited: set[Coord] = set()
    best_comp: Optional[Tuple[Coord, ...]] = None
    best_score: int = -(10**9)
    dims = (width, height)

    for y in range(height):
        for x in range(width):
            if board[y][x] == EMPTY or (x, y) in visited:
                continue

            # Component DFS
            stack: List[Coord] = [(x, y)]
            comp: List[Coord] = []
            has_w = False
            games = 0  # capped at 2
            has_cave = False
            fertile = 0
            def_on_boundary = 0

            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in visited:
                    continue
                visited.add((cx, cy))

                if board[cy][cx] == EMPTY:
                    continue

                comp.append((cx, cy))
                t = board[cy][cx]

                if t == "W":
                    has_w = True
                elif t == "G" and games < 2:
                    games += 1
                elif t == "C":
                    has_cave = True
                elif t == "F":
                    fertile += 1
                elif t == "D":
                    # boundary if any neighbor is '.'
                    for nx, ny in moore((cx, cy), dims):
                        if board[ny][nx] == EMPTY:
                            def_on_boundary += 1
                            break

                # push only non-empty neighbors
                for nx, ny in moore((cx, cy), dims):
                    if (nx, ny) not in visited and board[ny][nx] != EMPTY:
                        stack.append((nx, ny))

            comp_score = (
                (1 if has_w else 0)
                + games
                + (1 if has_cave else 0)
                + def_on_boundary
                + fertile
            )
            if comp_score > best_score:
                best_score = comp_score
                best_comp = tuple(sorted(comp))

    if best_comp is None:
        return None, 0
    return best_comp, best_score


def algo_c(board: Board) -> Tuple[Optional[Tuple[Coord, ...]], int]:
    """
    Faster CC scan:
      - Precompute which 'D' cells are on a boundary (adjacent to '.').
      - Use 2D visited grid and deque-based flood fill over non-'.' cells.
      - Aggregate component score on the fly; only retain coords for the best comp.
      - Avoids get()/moore() calls and sorting to reduce overhead.
    """
    from collections import deque

    height = len(board)
    if height == 0:
        return None, 0
    width = len(board[0])
    if width == 0:
        return None, 0

    EMPTY = "."
    # Precompute boundary D mask
    d_on_boundary = [[False] * width for _ in range(height)]
    for y in range(height):
        row = board[y]
        for x in range(width):
            if row[x] == "D":
                if (
                    (x > 0 and row[x - 1] == EMPTY)
                    or (x + 1 < width and row[x + 1] == EMPTY)
                    or (y > 0 and board[y - 1][x] == EMPTY)
                    or (y + 1 < height and board[y + 1][x] == EMPTY)
                ):
                    d_on_boundary[y][x] = True

    visited = [[False] * width for _ in range(height)]
    best_score = -(10**9)
    best_comp: Optional[List[Coord]] = None

    for y0 in range(height):
        row0 = board[y0]
        for x0 in range(width):
            if row0[x0] == EMPTY or visited[y0][x0]:
                continue

            # Flood fill this component
            stack = deque([(x0, y0)])
            visited[y0][x0] = True

            comp_nodes: List[Coord] = []
            has_w = False
            games = 0  # capped at 2
            has_cave = False
            fertile = 0
            def_on_boundary = 0

            while stack:
                x, y = stack.pop()
                t = board[y][x]
                comp_nodes.append((x, y))

                if t == "W":
                    has_w = True
                elif t == "G" and games < 2:
                    games += 1
                elif t == "C":
                    has_cave = True
                elif t == "F":
                    fertile += 1
                elif t == "D" and d_on_boundary[y][x]:
                    def_on_boundary += 1

                # neighbors (4-dir), push only non-empty and unvisited
                nx = x - 1
                if nx >= 0 and not visited[y][nx] and board[y][nx] != EMPTY:
                    visited[y][nx] = True
                    stack.append((nx, y))
                nx = x + 1
                if nx < width and not visited[y][nx] and board[y][nx] != EMPTY:
                    visited[y][nx] = True
                    stack.append((nx, y))
                ny = y - 1
                if ny >= 0 and not visited[ny][x] and board[ny][x] != EMPTY:
                    visited[ny][x] = True
                    stack.append((x, ny))
                ny = y + 1
                if ny < height and not visited[ny][x] and board[ny][x] != EMPTY:
                    visited[ny][x] = True
                    stack.append((x, ny))

            comp_score = (
                (1 if has_w else 0)
                + games
                + (1 if has_cave else 0)
                + def_on_boundary
                + fertile
            )
            if comp_score > best_score:
                best_score = comp_score
                best_comp = comp_nodes  # keep unsorted; harness doesn't require sorting

    if best_comp is None:
        return None, 0
    return tuple(best_comp), best_score


# ---------- Speed evaluation harness ----------


class EvalResult:
    def __init__(self):
        self.history: List[Tuple[int, float]] = []  # (size, avg_time)
        self.best_size: int = 0
        self.best_avg_time: float = 0.0
        self.stop_size: Optional[int] = None
        self.stop_avg_time: Optional[float] = None

    @property
    def best_cells(self) -> int:
        return self.best_size * self.best_size


def evaluate_algorithm(
    algo_fn,
    *,
    base_seed: int = 1337,
    non_dot_density: float = 0.5,
    time_budget_s: float = 5.0,
    boards_per_step: int = 1,
    start_size: int = 6,
    growth_factor: float = 1.5,
    max_steps: int = 100,
) -> EvalResult:
    """
    For each board size (square), generate `boards_per_step` boards (seeded deterministically),
    time the average solve time, and grow the size until avg_time >= time_budget_s.
    Score = largest size whose avg_time < time_budget_s.
    """
    res = EvalResult()
    densities = default_densities(non_dot_density)
    size = max(2, int(start_size))

    for step in range(max_steps):
        print(f"{size=}")
        times: List[float] = []
        for i in range(boards_per_step):
            seed = base_seed + step * 10_000 + i  # same sequence across algos
            board = generate_board(size, size, densities, seed)
            t0 = time.perf_counter()
            try:
                _ = algo_fn(board)
            except NotImplementedError:
                raise
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        avg_t = statistics.mean(times)
        res.history.append((size, avg_t))

        if avg_t < time_budget_s:
            res.best_size = size
            res.best_avg_time = avg_t
            # grow size
            next_size = max(size + 1, int(size * growth_factor))
            size = next_size
        else:
            res.stop_size = size
            res.stop_avg_time = avg_t
            break

    return res


def print_scorecard(name: str, r: EvalResult):
    print(f"\n== {name} ==")
    if r.history:
        print("Size -> avg seconds:")
        for sz, t in r.history:
            print(f"  {sz:4d} -> {t:.3f}s")
    if r.best_size > 0:
        print(
            f"Best size under budget: {r.best_size} (cells={r.best_cells}) "
            f"@ {r.best_avg_time:.3f}s"
        )
    if r.stop_size is not None:
        print(f"Crossed budget at size: {r.stop_size} with avg {r.stop_avg_time:.3f}s")


# ---------- Demo / Compare ----------

if __name__ == "__main__":
    algos = [
        ("algo_a", algo_a),
        ("algo_b", algo_b),
        ("algo_c", algo_c),
    ]

    for name, fn in algos:
        try:
            result = evaluate_algorithm(
                fn,
                base_seed=42,  # fixed so it's fair across algos
                non_dot_density=0.5,  # ~50% letters total
                time_budget_s=5.0,
                boards_per_step=1,  # 1 board per size; bump if you want smoother timing
                start_size=6,
                growth_factor=1.6,  # grows reasonably fast
            )
            print_scorecard(name, result)
        except NotImplementedError as e:
            print(f"\n== {name} ==\nSkipped: {e}")
