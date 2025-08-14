use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub type Coord = (usize, usize);
pub type Board = Vec<Vec<char>>;
pub type Hamlet = HashSet<Coord>;

pub type Predicate = dyn Fn(&Board, Coord, &Hamlet) -> bool + Send + Sync + 'static;

pub const EMPTY: char = '.';

// ----------------------------- Board helpers -----------------------------
pub fn dims(board: &Board) -> (usize, usize) {
    let h = board.len();
    let w = if h > 0 { board[0].len() } else { 0 };
    (w, h)
}

pub fn in_bounds(board: &Board, x: usize, y: usize) -> bool {
    let (w, h) = dims(board);
    x < w && y < h
}

pub fn get(board: &Board, c: Coord) -> Option<char> {
    let (x, y) = c;
    if in_bounds(board, x, y) {
        Some(board[y][x])
    } else {
        None
    }
}

pub fn neighbors4(x: usize, y: usize) -> impl Iterator<Item = Coord> {
    let mut v = Vec::with_capacity(4);
    if x > 0 {
        v.push((x - 1, y));
    }
    v.push((x + 1, y));
    if y > 0 {
        v.push((x, y - 1));
    }
    v.push((x, y + 1));
    v.into_iter()
}

// ----------------------------- Predicates (builders) -----------------------------
pub fn always_true() -> Arc<Predicate> {
    Arc::new(|_, _, _| true)
}

pub fn adj_to_symbol_in_world(sym: char) -> Arc<Predicate> {
    Arc::new(move |board, c, _ham| {
        let (x, y) = c;
        for (nx, ny) in neighbors4(x, y) {
            if in_bounds(board, nx, ny) && board[ny][nx] == sym {
                return true;
            }
        }
        false
    })
}

pub fn adj_to_symbol_in_ham(sym: char) -> Arc<Predicate> {
    Arc::new(move |board, c, ham| {
        let (x, y) = c;
        for (nx, ny) in neighbors4(x, y) {
            if ham.contains(&(nx, ny)) && board[ny][nx] == sym {
                return true;
            }
        }
        false
    })
}

pub fn not_adjacent_to_symbol_in_ham(sym: char) -> Arc<Predicate> {
    Arc::new(move |board, c, ham| {
        let (x, y) = c;
        for (nx, ny) in neighbors4(x, y) {
            if ham.contains(&(nx, ny)) && board[ny][nx] == sym {
                return false;
            }
        }
        true
    })
}

pub fn ham_contains_symbol(sym: char) -> Arc<Predicate> {
    Arc::new(move |board, _c, ham| ham.iter().any(|&(x, y)| board[y][x] == sym))
}

pub fn ham_lacks_symbol(sym: char) -> Arc<Predicate> {
    Arc::new(move |board, _c, ham| !ham.iter().any(|&(x, y)| board[y][x] == sym))
}

pub fn on_ham_perimeter() -> Arc<Predicate> {
    Arc::new(move |board, c, ham| {
        let (x, y) = c;
        for (nx, ny) in neighbors4(x, y) {
            if in_bounds(board, nx, ny) && !ham.contains(&(nx, ny)) {
                return true;
            }
        }
        false
    })
}

// ----------------------------- Constraints / Terms -----------------------------
#[derive(Clone)]
pub struct RequireAtLeast {
    pub symbol: char,
    pub n: usize,
}
impl RequireAtLeast {
    pub fn ok(&self, board: &Board, ham: &Hamlet) -> bool {
        let mut cnt = 0usize;
        for &(x, y) in ham.iter() {
            if board[y][x] == self.symbol {
                cnt += 1;
                if cnt >= self.n {
                    return true;
                }
            }
        }
        false
    }
}

#[derive(Clone)]
pub struct CountTerm {
    pub symbol: char,
    pub weight: f64,
    pub cap: Option<usize>,
    pub predicate: Option<Arc<Predicate>>, // evaluated per cell in hamlet
}
impl CountTerm {
    pub fn score(&self, board: &Board, ham: &Hamlet) -> f64 {
        let pred = self.predicate.as_ref().map(|p| p.as_ref());
        let mut cnt = 0usize;
        for &(x, y) in ham.iter() {
            if board[y][x] == self.symbol {
                let ok = match pred {
                    Some(p) => p(board, (x, y), ham),
                    None => true,
                };
                if ok {
                    cnt += 1;
                }
            }
        }
        let cnt = match self.cap {
            Some(c) => cnt.min(c),
            None => cnt,
        };
        self.weight * (cnt as f64)
    }
    pub fn count_units(&self, board: &Board, ham: &Hamlet) -> usize {
        let pred = self.predicate.as_ref().map(|p| p.as_ref());
        let mut cnt = 0usize;
        for &(x, y) in ham.iter() {
            if board[y][x] == self.symbol {
                let ok = match pred {
                    Some(p) => p(board, (x, y), ham),
                    None => true,
                };
                if ok {
                    cnt += 1;
                }
            }
        }
        match self.cap {
            Some(c) => cnt.min(c),
            None => cnt,
        }
    }
}

#[derive(Clone)]
pub struct ScoreProgram {
    pub requires: Vec<RequireAtLeast>,
    pub terms: Vec<CountTerm>,
}
impl ScoreProgram {
    pub fn valid(&self, board: &Board, ham: &Hamlet) -> bool {
        self.requires.iter().all(|r| r.ok(board, ham))
    }
    pub fn score(&self, board: &Board, ham: &Hamlet) -> f64 {
        self.terms.iter().map(|t| t.score(board, ham)).sum()
    }
}

// ----------------------------- Solver -----------------------------
pub fn solve_best(board: &Board, program: &ScoreProgram) -> (Hamlet, f64) {
    let comps = nonempty_components(board);
    let mut best_h = Hamlet::new();
    let mut best_s = f64::NEG_INFINITY;
    for comp in comps {
        let (h, s) = solve_component_enum(board, program, &comp);
        if s > best_s {
            best_s = s;
            best_h = h;
        }
    }
    if best_s == f64::NEG_INFINITY {
        (Hamlet::new(), 0.0)
    } else {
        (best_h, best_s)
    }
}

fn nonempty_components(board: &Board) -> Vec<Vec<Coord>> {
    let (w, h) = dims(board);
    let mut seen = vec![vec![false; w]; h];
    let mut out: Vec<Vec<Coord>> = Vec::new();

    for y in 0..h {
        for x in 0..w {
            if seen[y][x] || board[y][x] == EMPTY {
                continue;
            }
            let mut comp: Vec<Coord> = Vec::new();
            let mut dq = VecDeque::new();
            dq.push_back((x, y));
            seen[y][x] = true;
            while let Some((cx, cy)) = dq.pop_front() {
                comp.push((cx, cy));
                for (nx, ny) in neighbors4(cx, cy) {
                    if in_bounds(board, nx, ny) && !seen[ny][nx] && board[ny][nx] != EMPTY {
                        seen[ny][nx] = true;
                        dq.push_back((nx, ny));
                    }
                }
            }
            out.push(comp);
        }
    }
    out
}

fn solve_component_enum(board: &Board, program: &ScoreProgram, comp: &Vec<Coord>) -> (Hamlet, f64) {
    if comp.is_empty() {
        return (Hamlet::new(), f64::NEG_INFINITY);
    }

    let mut nodes = comp.clone();
    nodes.sort();
    let n = nodes.len();
    let mut index_of: HashMap<Coord, usize> = HashMap::new();
    for (i, &c) in nodes.iter().enumerate() {
        index_of.insert(c, i);
    }

    // adjacency list within component
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let labels: Vec<char> = nodes.iter().map(|&(x, y)| board[y][x]).collect();
    for (i, &(x, y)) in nodes.iter().enumerate() {
        for (nx, ny) in neighbors4(x, y) {
            if let Some(&j) = index_of.get(&(nx, ny)) {
                adj[i].push(j);
            }
        }
    }

    // total symbol counts in component
    let mut total_sym: HashMap<char, usize> = HashMap::new();
    for &s in labels.iter() {
        *total_sym.entry(s).or_insert(0) += 1;
    }

    let mut best_h = Hamlet::new();
    let mut best_s = f64::NEG_INFINITY;

    for r in 0..n {
        // pool feasibility from r..n-1
        let mut pool_total: HashMap<char, usize> = HashMap::new();
        for i in r..n {
            *pool_total.entry(labels[i]).or_insert(0) += 1;
        }
        let feasible_pool = program
            .requires
            .iter()
            .all(|req| pool_total.get(&req.symbol).cloned().unwrap_or(0) >= req.n);
        if !feasible_pool {
            continue;
        }

        let mut in_set = vec![false; n];
        in_set[r] = true;
        let mut cur_list: Vec<usize> = vec![r];
        let mut cur_sym: HashMap<char, usize> = HashMap::new();
        *cur_sym.entry(labels[r]).or_insert(0) += 1;

        // initial frontier
        let mut frontier_set: HashSet<usize> = HashSet::new();
        for &j in &adj[r] {
            if j >= r && !in_set[j] {
                frontier_set.insert(j);
            }
        }
        let mut frontier: Vec<usize> = frontier_set.into_iter().collect();
        frontier.sort_unstable();

        let mut current_ham = || -> Hamlet { cur_list.iter().map(|&i| nodes[i]).collect() };
        let mut current_score = || -> f64 {
            let h = current_ham();
            program.score(board, &h)
        };

        // seed with singleton
        let h0 = current_ham();
        if program.valid(board, &h0) {
            let s0 = program.score(board, &h0);
            if s0 > best_s {
                best_s = s0;
                best_h = h0;
            }
        }

        // optimistic UB for positive terms using remaining symbol counts
        let optimistic_ub =
            |cur_score_val: f64, ham: &Hamlet, cur_sym_map: &HashMap<char, usize>| -> f64 {
                let mut ub = cur_score_val;
                let mut remaining_sym: HashMap<char, usize> = HashMap::new();
                for (sym, tot) in total_sym.iter() {
                    let taken = cur_sym_map.get(sym).cloned().unwrap_or(0);
                    let rem = tot.saturating_sub(taken);
                    remaining_sym.insert(*sym, rem);
                }
                for t in program.terms.iter() {
                    if t.weight > 0.0 {
                        let cur_term_units = t.count_units(board, ham);
                        let need = match t.cap {
                            Some(c) => c.saturating_sub(cur_term_units),
                            None => remaining_sym.get(&t.symbol).cloned().unwrap_or(0),
                        };
                        let addable = remaining_sym.get(&t.symbol).cloned().unwrap_or(0);
                        let gain = std::cmp::min(need, addable);
                        ub += t.weight * (gain as f64);
                    }
                }
                ub
            };

        let requirements_possible = |cur_sym_map: &HashMap<char, usize>| -> bool {
            for req in program.requires.iter() {
                let have = cur_sym_map.get(&req.symbol).cloned().unwrap_or(0);
                let avail = total_sym
                    .get(&req.symbol)
                    .cloned()
                    .unwrap_or(0)
                    .saturating_sub(have);
                if have + avail < req.n {
                    return false;
                }
            }
            true
        };

        fn dfs(
            idx_frontier: &Vec<usize>,
            start_k: usize,
            r: usize,
            nodes: &Vec<Coord>,
            labels: &Vec<char>,
            adj: &Vec<Vec<usize>>,
            board: &Board,
            program: &ScoreProgram,
            in_set: &mut Vec<bool>,
            cur_list: &mut Vec<usize>,
            cur_sym: &mut HashMap<char, usize>,
            best_s: &mut f64,
            best_h: &mut Hamlet,
            optimistic_ub_fn: &dyn Fn(f64, &Hamlet, &HashMap<char, usize>) -> f64,
            requirements_possible_fn: &dyn Fn(&HashMap<char, usize>) -> bool,
        ) {
            let mut k = start_k; // index into frontier vector
            while k < idx_frontier.len() {
                let v = idx_frontier[k];
                if in_set[v] {
                    k += 1;
                    continue;
                }

                // include v
                in_set[v] = true;
                cur_list.push(v);
                *cur_sym.entry(labels[v]).or_insert(0) += 1;

                // build new frontier: later candidates + neighbors of v
                let mut new_front_set: HashSet<usize> =
                    idx_frontier[k + 1..].iter().copied().collect();
                for &w in &adj[v] {
                    if w >= r && !in_set[w] {
                        new_front_set.insert(w);
                    }
                }
                let mut new_front: Vec<usize> = new_front_set.into_iter().collect();
                new_front.sort_unstable();

                if requirements_possible_fn(cur_sym) {
                    let ham = cur_list.iter().map(|&i| nodes[i]).collect::<Hamlet>();
                    if program.valid(board, &ham) {
                        let cs = program.score(board, &ham);
                        if cs > *best_s {
                            *best_s = cs;
                            *best_h = ham.clone();
                        }
                        let ub = optimistic_ub_fn(cs, &ham, cur_sym);
                        if ub > *best_s {
                            dfs(
                                &new_front,
                                0,
                                r,
                                nodes,
                                labels,
                                adj,
                                board,
                                program,
                                in_set,
                                cur_list,
                                cur_sym,
                                best_s,
                                best_h,
                                optimistic_ub_fn,
                                requirements_possible_fn,
                            );
                        }
                    } else {
                        // even if not valid yet, still can expand if UB says possible
                        let cs = program.score(board, &ham);
                        let ub = optimistic_ub_fn(cs, &ham, cur_sym);
                        if ub > *best_s {
                            dfs(
                                &new_front,
                                0,
                                r,
                                nodes,
                                labels,
                                adj,
                                board,
                                program,
                                in_set,
                                cur_list,
                                cur_sym,
                                best_s,
                                best_h,
                                optimistic_ub_fn,
                                requirements_possible_fn,
                            );
                        }
                    }
                }

                // backtrack
                if let Some(e) = cur_sym.get_mut(&labels[v]) {
                    *e -= 1;
                    if *e == 0 {
                        cur_sym.remove(&labels[v]);
                    }
                }
                cur_list.pop();
                in_set[v] = false;

                // skip branch (implicit by advancing k)
                k += 1;
            }
        }

        dfs(
            &frontier,
            0,
            r,
            &nodes,
            &labels,
            &adj,
            board,
            program,
            &mut in_set,
            &mut cur_list,
            &mut cur_sym,
            &mut best_s,
            &mut best_h,
            &optimistic_ub,
            &requirements_possible,
        );
    }

    (best_h, best_s)
}

// ----------------------------- Generation helpers -----------------------------
pub static LETTERS: &[char] = &['W', 'G', 'C', 'D', 'F', 'U', 'S', 'R', 'V', 'M'];

pub fn default_densities(non_dot_density: f64) -> HashMap<char, f64> {
    let non_dot = non_dot_density.clamp(0.0, 1.0);
    let empty_p = 1.0 - non_dot;
    let per = if LETTERS.is_empty() {
        0.0
    } else {
        non_dot / (LETTERS.len() as f64)
    };
    let mut d: HashMap<char, f64> = HashMap::new();
    d.insert(EMPTY, empty_p);
    for &ch in LETTERS {
        d.insert(ch, per);
    }
    d
}

fn normalize_densities(densities: &HashMap<char, f64>) -> Vec<(char, f64)> {
    let mut filtered: HashMap<char, f64> = HashMap::new();
    for (&k, &v) in densities.iter() {
        if k == EMPTY || LETTERS.contains(&k) {
            filtered.insert(k, v.max(0.0));
        }
    }
    if !filtered.contains_key(&EMPTY) {
        filtered.insert(EMPTY, 0.0);
    }
    let total: f64 = filtered.values().sum::<f64>().max(1.0);
    let mut cum = 0.0;
    let mut cdf: Vec<(char, f64)> = Vec::new();
    let mut order: Vec<char> = vec![EMPTY];
    order.extend_from_slice(LETTERS);
    for sym in order.into_iter() {
        let p = *filtered.get(&sym).unwrap_or(&0.0) / total;
        cum += p;
        cdf.push((sym, cum));
    }
    if let Some(last) = cdf.last_mut() {
        last.1 = 1.0;
    }
    cdf
}

pub fn generate_board(
    width: usize,
    height: usize,
    densities: Option<HashMap<char, f64>>,
    seed: u64,
) -> Board {
    let dens = densities.unwrap_or_else(|| default_densities(0.5));
    let cdf = normalize_densities(&dens);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut pick = |r: f64| -> char {
        for (sym, cum) in cdf.iter() {
            if r <= *cum {
                return *sym;
            }
        }
        cdf.last().unwrap().0
    };

    let mut board: Board = vec![vec![EMPTY; width]; height];
    for y in 0..height {
        for x in 0..width {
            let r: f64 = rng.gen();
            board[y][x] = pick(r);
        }
    }
    board
}

// ----------------------------- Utilities -----------------------------
pub fn sorted_coords(ham: &Hamlet) -> Vec<Coord> {
    let mut v: Vec<Coord> = ham.iter().cloned().collect();
    v.sort();
    v
}
