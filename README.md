# CVRP Metaheuristic Benchmark

This project implements a reproducible capacitated vehicle routing problem (CVRP) benchmark with three algorithms written from scratch: a permutation-based genetic algorithm (GA), an edge-based ant colony optimizer (ACO), and a random-key particle swarm optimizer (PSO).

## Run

```bash
pip install -r requirements.txt
python run_experiment.py
python scripts/run_single.py --algorithm ga --seed 0
```

On a multicore machine, independent seeds can be evaluated in parallel without changing algorithm results:

```bash
python run_experiment.py --workers 10
```

The default experiment uses one fixed 50-customer instance, 30 seeds, and a common fitness-evaluation budget. Results are written under `results/` as CSV, Markdown, PNG, and PDF files. Use `--config` to provide another JSON configuration and `--quick` for a short smoke test.

The same split decoder is used by GA and PSO: a customer permutation is scanned from left to right and a new vehicle route starts whenever adding the next demand would exceed capacity. ACO constructs feasible routes directly. Every route starts and ends at the depot, and every customer is visited exactly once.

The comparison is based on a common number of objective evaluations rather than raw generations/iterations. All stochastic components are seeded, and the fixed instance is generated once and saved as `data/cvrp_instance.csv`.

Two reproducible configurations are included after parameter validation:

- `configs/baseline_config.json` preserves the original GA mutation rate of 0.20 used by the primary three-algorithm benchmark.
- `configs/optimized_config.json` uses the held-out validated GA mutation rate of 0.05. ACO and PSO parameters are unchanged because their sensitivity candidates did not generalize on held-out seeds.

Run the optimized configuration with:

```bash
python run_experiment.py --config configs/optimized_config.json --workers 10
```

## Algorithms

GA uses tournament selection, ordered crossover, inversion mutation, and elitism. ACO uses pheromone/heuristic transition probabilities, capacity-aware route construction, evaporation, and iteration/global-best reinforcement. PSO uses continuous random keys; sorting a particle position gives a customer permutation, which is decoded by the same CVRP decoder. Random keys make standard continuous PSO updates applicable, while the sorting map introduces a representation mismatch that is discussed in the generated report.

## Outputs

- `results/raw/all_runs.csv`: one row per algorithm and seed.
- `results/raw/convergence_*.csv`: best-so-far traces.
- `results/summary/summary_statistics.csv`: descriptive statistics.
- `results/summary/statistical_tests.csv`: Kruskal-Wallis, pairwise Mann-Whitney U, and Benjamini-Hochberg adjusted p-values.
- `results/summary/parameter_sensitivity.csv`: one-factor sensitivity runs.
- `results/summary/convergence_metrics.csv`: evaluation counts required to reach 90% of each run's total improvement.
- `results/summary/optimization_validation.csv`: candidate parameters evaluated on held-out seeds 10--29.
- `results/summary/optimized_comparison.csv`: baseline and validated GA summary.
- `results/summary/experiment_report.md`: automatically generated interpretation.
- `results/figures/`: instance, convergence, boxplot, route, and sensitivity figures.

Only NumPy, pandas, matplotlib, and the Python standard library are required; SciPy is optional for the requested non-parametric tests.

The completed outputs can be re-analysed without rerunning the optimizers:

```bash
python scripts/analyze_results.py
```
