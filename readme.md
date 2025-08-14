# Hamlet Solver

A fast, flexible **hamlet** (connected-subset) solver with pluggable scoring rules, written in **Rust**.

Design goals:

* **Predicate-agnostic**: any rule can depend on the *entire hamlet* (no special flags).
* **Composable scoring**: mix requirements, positive/negative weights, caps, and custom predicates.
* **Deterministic**: random board generation is seedable.
* **Usable**: simple CLI to run the original puzzle, handcrafted scenarios, or a stress benchmark.

This crate is a faithful (yet idiomatic) Rust port of the Python prototype, with higher performance and a small, clean API for designing game-y settlement rules.

---

## TL;DR

```bash
# Build
cargo build --release

# Verify the original puzzle (sanity check)
cargo run --release -- original

# Run all scenarios (incl. ORIGINAL first) with a green-highlighted hamlet
cargo run --release -- scenarios

# Push the solver with a deterministic random stress test (tweak params)
cargo run --release -- speed \
  --scen 2 \
  --time-budget-s 3.0 \
  --non-dot-density 0.75 \
  --start-size 96 \
  --growth 1.7 \
  --max-steps 18 \
  --max-size 2500 \
  --max-total-time-s 180
```

The `scenarios` command prints the board with the chosen hamlet **in bold green**, plus the score and exact coordinates.

---

## Project Layout

```
hamlet_solver/
├─ Cargo.toml
└─ src/
   ├─ main.rs        # CLI entrypoints (original, scenarios, speed)
   ├─ solver.rs      # Core solver + predicates + board generator
   ├─ games.rs       # ORIGINAL problem + hand-crafted scenarios (Lazy-initialized)
   └─ viz.rs         # ANSI pretty-print with in-board green highlighting
```

> **Note**: `games::SCENARIOS` is defined with `once_cell::sync::Lazy` so we can allocate `Vec`, `String`, and `Arc` at runtime.

---

## Concepts

* **Board**: `Vec<Vec<char>>` grid. `'.'` means barren; any other char is a resource (e.g., `W`, `G`, `F`, `D`, `U`, `S`, `R`, `V`, `M`).
* **Hamlet**: a **4-connected** set of non-barren cells.
* **Program**: a `ScoreProgram` with two parts:

  * **Requires**: `RequireAtLeast { symbol, n }` — constraints that must hold.
  * **Terms**: `CountTerm { symbol, weight, cap, predicate }`

    * `weight` may be **negative** (penalties) or positive (bonuses)
    * `cap` limits contribution from that symbol (e.g., at most 2 `G`)
    * `predicate` is a closure `Fn(board, coord, hamlet) -> bool` and may freely depend on the **hamlet**

Because predicates can depend on the **hamlet itself**, the solver explores connected **subsets** of each component and prunes with feasibility checks and optimistic upper bounds for positive terms. This trades speed for maximum expressiveness.

---

## CLI

```
cargo run --release -- <command> [options]
```

### `original`

Runs the canonical board/rules and prints the best hamlet.

### `scenarios`

Runs a curated set of scenarios (including ORIGINAL first). Good for eyeballing behavior of penalties and hamlet-dependent predicates.

### `speed`

Deterministic random stress test. Grows board size until an average per-board solve reaches `time-budget-s`, or a hard cap is hit.

**Options:**

* `--scen <idx>` — Which scenario program to use (default 0). Indexing follows `games::SCENARIOS`.
* `--time-budget-s <secs>` — Stop growing size when a single solve takes longer than this (default `1.0`).
* `--non-dot-density <0..1>` — Probability mass assigned to non-barren letters (default `0.5`).
* `--start-size <N>` — Initial side length (default `6`).
* `--growth <factor>` — Size growth factor per step (default `1.6`).
* `--max-steps <N>` — Safety cap on number of growth steps (default `12`).
* `--max-size <N>` — Safety cap on side length (default `120`).
* `--max-total-time-s <secs>` — Safety cap on total benchmark time (default `20`).

---

## API (quick)

```rust
// solver.rs
pub type Coord = (usize, usize);
pub type Board = Vec<Vec<char>>;
pub type Hamlet = std::collections::HashSet<Coord>;

pub struct RequireAtLeast { pub symbol: char, pub n: usize }

pub struct CountTerm {
    pub symbol: char,
    pub weight: f64,                 // negative allowed
    pub cap: Option<usize>,          // None = unlimited
    pub predicate: Option<Arc<Predicate>>, // Fn(&Board, Coord, &Hamlet) -> bool
}

pub struct ScoreProgram {
    pub requires: Vec<RequireAtLeast>,
    pub terms: Vec<CountTerm>,
}

pub fn solve_best(board: &Board, program: &ScoreProgram) -> (Hamlet, f64);

// Common predicate builders
pub fn adj_to_symbol_in_world(sym: char) -> Arc<Predicate>;
pub fn adj_to_symbol_in_ham(sym: char)   -> Arc<Predicate>;
pub fn not_adjacent_to_symbol_in_ham(sym: char) -> Arc<Predicate>;
pub fn ham_contains_symbol(sym: char) -> Arc<Predicate>;
pub fn ham_lacks_symbol(sym: char)    -> Arc<Predicate>;
pub fn on_ham_perimeter() -> Arc<Predicate>;

// Random board gen
pub fn default_densities(non_dot_density: f64) -> HashMap<char, f64>;
pub fn generate_board(w: usize, h: usize, densities: Option<HashMap<char, f64>>, seed: u64) -> Board;
```

### Writing a new rule

```rust
use std::sync::Arc;
use hamlet_solver::solver::*;

let program = ScoreProgram {
    requires: vec![ RequireAtLeast { symbol: 'W', n: 2 } ],
    terms: vec![
        // Farms count only if next to Water **in the hamlet**
        CountTerm { symbol: 'F', weight: 1.0, cap: None, predicate: Some(adj_to_symbol_in_ham('W')) },
        // Mines count only if next to a hamlet Volcano and NO Water present anywhere in hamlet
        CountTerm { symbol: 'M', weight: 2.0, cap: None, predicate: Some(Arc::new(|b, c, h| (adj_to_symbol_in_ham('V'))(b,c,h) && (ham_lacks_symbol('W'))(b,c,h))) },
        // Swamp is bad
        CountTerm { symbol: 'S', weight: -1.5, cap: None, predicate: None },
    ],
};
```

---

## Visualization

`viz::viz_in_board(board, hamlet, show_coords)` renders the **original board** with the chosen hamlet highlighted in **bold green** using ANSI escapes. Most terminals (Linux/macOS, modern Windows Terminal) support this.

---

## Performance Notes

* Worst-case complexity is exponential in the size of a connected component (because we enumerate connected subsets). Predicates that are hamlet-dependent (e.g., perimeter, adjacency-in-hamlet, global “hamlet contains X”) force the general search.
* The solver prunes using:

  * **Feasibility** of `RequireAtLeast` given remaining symbols.
  * **Optimistic UB** for **positive** terms using remaining symbol counts.
* Tips to scale:

  * Lower `non-dot-density` to fragment components.
  * Reduce negative terms and complex hamlet-dependent predicates when exploring huge boards.
  * Use `--time-budget-s`, `--max-size`, and `--max-total-time-s` to keep benchmarks bounded.

---

## Troubleshooting

* **Compilation error about statics and allocations**: `games::SCENARIOS` is a `Lazy<Vec<Scenario>>` (via `once_cell`). Ensure `Cargo.toml` has:

  ```toml
  once_cell = "1.19"
  ```
* **No color on Windows**: use Windows Terminal (ANSI), or set `TERM` appropriately. The `colored` crate prints ANSI sequences.

---

## Credits

Originally prototyped in Python for experimentation; this is a faithful Rust port with additional ergonomics and performance. Designed for tinkering with rule systems for base/city placement, resource synergies, and trade-offs.
