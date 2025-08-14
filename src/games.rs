use crate::solver::*;
use once_cell::sync::Lazy;
use std::sync::Arc;

#[derive(Clone)]
pub struct Scenario {
    pub name: String,
    pub blurb: String,
    pub board: Board,
    pub program: ScoreProgram,
}

pub fn original_problem() -> (Board, ScoreProgram) {
    let board: Board = vec![
        vec!['W', 'G', '.', 'F', '.', 'W', '.'],
        vec!['.', 'C', '.', 'C', '.', '.', 'C'],
        vec!['.', '.', 'D', '.', 'G', 'D', 'F'],
        vec!['G', 'F', 'W', '.', 'F', 'W', '.'],
        vec!['.', '.', '.', 'W', '.', '.', '.'],
        vec!['G', 'C', '.', 'W', '.', 'C', 'W'],
        vec!['F', '.', 'C', '.', 'F', 'D', 'F'],
        vec!['.', 'W', 'F', 'G', '.', 'G', '.'],
    ];

    let program = ScoreProgram {
        requires: vec![
            RequireAtLeast { symbol: 'W', n: 1 },
            RequireAtLeast { symbol: 'G', n: 1 },
        ],
        terms: vec![
            CountTerm {
                symbol: 'W',
                weight: 1.0,
                cap: Some(1),
                predicate: None,
            },
            CountTerm {
                symbol: 'G',
                weight: 1.0,
                cap: Some(2),
                predicate: None,
            },
            CountTerm {
                symbol: 'D',
                weight: 1.0,
                cap: None,
                predicate: Some(adj_to_symbol_in_world('.')),
            },
            CountTerm {
                symbol: 'C',
                weight: 1.0,
                cap: Some(1),
                predicate: None,
            },
            CountTerm {
                symbol: 'F',
                weight: 1.0,
                cap: None,
                predicate: None,
            },
        ],
    };

    (board, program)
}

// NOTE: we use Lazy so we can allocate Strings, Vecs, and Arcs at runtime
// instead of inside a `static` initializer.
pub static SCENARIOS: Lazy<Vec<Scenario>> = Lazy::new(|| {
    vec![
    Scenario {
        name: "Irrigated Farms (hamlet-adj) vs Swamp".into(),
        blurb: "Farms only count if the hamlet itself places them next to Water. Swamps are negative.".into(),
        board: vec![
            vec!['.','.','W','F','.', '.', '.'],
            vec!['.','F','F','F','S','S','.'],
            vec!['.','.','W','.', 'F','.', '.'],
            vec!['.','F','F','F','S','.', '.'],
            vec!['.','.','.', '.', '.', '.', '.'],
        ],
        program: ScoreProgram {
            requires: vec![ RequireAtLeast { symbol: 'W', n: 1 } ],
            terms: vec![
                CountTerm { symbol: 'F', weight: 1.0, cap: None, predicate: Some(adj_to_symbol_in_ham('W')) },
                CountTerm { symbol: 'W', weight: 1.0, cap: Some(2), predicate: None },
                CountTerm { symbol: 'S', weight: -1.5, cap: None, predicate: None },
            ],
        },
    },
    Scenario {
        name: "Volcanic Mining".into(),
        blurb: "Mines count only if adjacent to a Volcano in the hamlet, and they become invalid if the hamlet contains any Water.".into(),
        board: vec![
            vec!['.','.','V','.', '.', '.', '.'],
            vec!['.','M','M','V','M','M','.'],
            vec!['.','.','M','.', 'M','.', '.'],
            vec!['.','.','.', '.', '.', '.', '.'],
        ],
        program: ScoreProgram {
            requires: vec![],
            terms: vec![
                CountTerm { symbol: 'M', weight: 2.0, cap: None, predicate: Some(Arc::new(|b,c,h| (adj_to_symbol_in_ham('V'))(b,c,h) && (ham_lacks_symbol('W'))(b,c,h))) },
                CountTerm { symbol: 'V', weight: 0.5, cap: Some(1), predicate: None },
                CountTerm { symbol: 'W', weight: -2.0, cap: None, predicate: None },
            ],
        },
    },
    Scenario {
        name: "Perimeter Forts vs Raiders".into(),
        blurb: "Defenses only count when placed on the hamlet perimeter; Raider camps inside the hamlet are heavy penalties.".into(),
        board: vec![
            vec!['.','D','D','D','.', 'D','D','D','.'],
            vec!['D','F','F','.', 'R','.', 'F','F','D'],
            vec!['D','F','G','F','W','F','G','F','D'],
            vec!['.','.', 'D','.', '.', '.', 'D','.', '.'],
        ],
        program: ScoreProgram {
            requires: vec![ RequireAtLeast { symbol: 'W', n: 1 } ],
            terms: vec![
                CountTerm { symbol: 'D', weight: 2.0, cap: None, predicate: Some(on_ham_perimeter()) },
                CountTerm { symbol: 'G', weight: 1.0, cap: Some(2), predicate: None },
                CountTerm { symbol: 'F', weight: 1.0, cap: None, predicate: None },
                CountTerm { symbol: 'R', weight: -2.5, cap: None, predicate: None },
            ],
        },
    },
    Scenario {
        name: "Crystal Cavern with Uranium".into(),
        blurb: "Tempting crystal cave (C) surrounded by pockets of Uranium (U). A small crescent hamlet that just touches C and W without U beats the big cluster.".into(),
        board: vec![
            vec!['.','.','.', 'D','.', '.', '.', '.'],
            vec!['.','F','F','C','F','U','F','.'],
            vec!['.','F','U','W','F','F','F','.'],
            vec!['D','F','F','W','F','U','F','.'],
            vec!['.','F','U','W','F','F','G','.'],
            vec!['.','.','.', 'D','.', '.', '.', '.'],
        ],
        program: ScoreProgram {
            requires: vec![ RequireAtLeast { symbol: 'W', n: 1 } ],
            terms: vec![
                CountTerm { symbol: 'C', weight: 3.0, cap: Some(1), predicate: None },
                CountTerm { symbol: 'F', weight: 1.0, cap: Some(2), predicate: None },
                CountTerm { symbol: 'W', weight: 1.0, cap: Some(1), predicate: None },
                CountTerm { symbol: 'G', weight: 1.0, cap: Some(2), predicate: None },
                CountTerm { symbol: 'D', weight: 1.0, cap: None, predicate: Some(adj_to_symbol_in_world('.')) },
                CountTerm { symbol: 'U', weight: -2.5, cap: None, predicate: None },
            ],
        },
    },
]
});
