use clap::{Parser, Subcommand};

mod games;
mod solver;
mod viz;

use std::time::Instant;

use crate::{
    games::{original_problem, Scenario, SCENARIOS},
    solver::{default_densities, generate_board, solve_best, sorted_coords},
    viz::viz_in_board,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Hamlet solver (Rust port)", long_about=None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Verify original board/rules and print the result
    Original,
    /// Run all scenarios and visualize hamlets
    Scenarios,
    /// Random stress test using a scenario's program
    Speed {
        /// Scenario index in games::SCENARIOS (default 0)
        #[arg(long, default_value_t = 0)]
        scen: usize,
        /// Time budget per size to stop
        #[arg(long, default_value_t = 1.0)]
        time_budget_s: f64,
        /// Non-empty density across letters (0..1)
        #[arg(long, default_value_t = 0.5)]
        non_dot_density: f64,
        /// Start side length
        #[arg(long, default_value_t = 6)]
        start_size: usize,
        /// Growth factor per step
        #[arg(long, default_value_t = 1.6)]
        growth: f64,
        /// Max steps
        #[arg(long, default_value_t = 12)]
        max_steps: usize,
        /// Max side length cap
        #[arg(long, default_value_t = 120)]
        max_size: usize,
        /// Max total seconds for the whole test
        #[arg(long, default_value_t = 20.0)]
        max_total_time_s: f64,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Original => run_original(),
        Command::Scenarios => run_scenarios(),
        Command::Speed {
            scen,
            time_budget_s,
            non_dot_density,
            start_size,
            growth,
            max_steps,
            max_size,
            max_total_time_s,
        } => run_speed(
            scen,
            time_budget_s,
            non_dot_density,
            start_size,
            growth,
            max_steps,
            max_size,
            max_total_time_s,
        ),
    }
}

fn run_original() {
    println!(
        "== ORIGINAL Problem ==\nOriginal rules; sanity check that the generic solver matches."
    );
    let (board, program) = original_problem();
    let t0 = Instant::now();
    let (best_h, best_s) = solve_best(&board, &program);
    let dt = t0.elapsed();
    println!("score = {:.3}", best_s);
    println!("{}", viz_in_board(&board, &best_h, true));
    println!("hamlet coords: {:?}", sorted_coords(&best_h));
    println!("took: {:.3}s", dt.as_secs_f64());
}

fn run_scenarios() {
    let mut scenarios = vec![Scenario {
        name: "ORIGINAL Problem".to_string(),
        blurb: "Original rules; sanity check that the generic solver matches.".to_string(),
        board: original_problem().0,
        program: original_problem().1,
    }];
    scenarios.extend(SCENARIOS.iter().cloned());

    for sc in scenarios {
        println!("\n== {} ==\n{}", sc.name, sc.blurb);
        let t0 = Instant::now();
        let (best_h, best_s) = solve_best(&sc.board, &sc.program);
        let dt = t0.elapsed();
        println!("score = {:.3}", best_s);
        println!("{}", viz_in_board(&sc.board, &best_h, true));
        println!("hamlet coords: {:?}", sorted_coords(&best_h));
        println!("took: {:.3}s", dt.as_secs_f64());
    }
}

fn run_speed(
    scen: usize,
    time_budget_s: f64,
    non_dot_density: f64,
    start_size: usize,
    growth: f64,
    max_steps: usize,
    max_size: usize,
    max_total_time_s: f64,
) {
    use std::cmp::max;
    let scen_ref = &*SCENARIOS;
    let program = if scen_ref.is_empty() {
        original_problem().1
    } else {
        scen_ref[scen.min(scen_ref.len() - 1)].program.clone()
    };
    println!("== Random stress (using scenario {} program) ==", scen);
    let mut size = start_size.max(2);
    let t_start = Instant::now();
    for step in 0..max_steps {
        if size > max_size {
            break;
        }
        if t_start.elapsed().as_secs_f64() > max_total_time_s {
            break;
        }
        let board = generate_board(
            size,
            size,
            Some(default_densities(non_dot_density)),
            1234 + (step as u64) * 1000,
        );
        let t0 = Instant::now();
        let _ = solve_best(&board, &program);
        let avg = t0.elapsed().as_secs_f64();
        println!("  {:4} -> {:.3}s", size, avg);
        if avg >= time_budget_s {
            break;
        }
        size = max(size + 1, ((size as f64) * growth) as usize);
    }
}
