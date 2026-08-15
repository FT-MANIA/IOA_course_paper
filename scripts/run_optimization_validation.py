"""Validate sensitivity-selected parameters on held-out random seeds."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_experiment import execute_jobs
from src.problem import load_instance
from src.utils import load_config


def main() -> None:
    """Evaluate selected candidates on seeds 10--29 and compare them pairwise."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.json")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    instance = load_instance("data/cvrp_instance.csv", config["problem"]["vehicle_capacity"])
    baseline = pd.read_csv("results/raw/all_runs.csv")
    candidates = {
        "GA": ("mutation_rate", 0.05),
        "ACO": ("beta", 2.0),
        "PSO": ("w_end", 0.6),
    }
    rows: list[dict] = []
    for algorithm, (parameter, candidate_value) in candidates.items():
        config_key = algorithm.lower()
        local = dict(config[config_key])
        baseline_value = local[parameter]
        local[parameter] = candidate_value
        completed = execute_jobs(
            algorithm,
            instance,
            local,
            range(10, 30),
            config["experiment"]["max_fitness_evaluations"],
            args.workers,
        )
        baseline_lookup = baseline[baseline["Algorithm"] == algorithm].set_index("Seed")["BestDistance"]
        for seed, result in completed:
            base_distance = float(baseline_lookup.loc[seed])
            rows.append(
                {
                    "Algorithm": algorithm,
                    "Parameter": parameter,
                    "BaselineValue": baseline_value,
                    "CandidateValue": candidate_value,
                    "Seed": seed,
                    "BaselineDistance": base_distance,
                    "CandidateDistance": result.best_distance,
                    "Improvement": base_distance - result.best_distance,
                    "CandidateVehicles": len(result.best_routes),
                    "FitnessEvaluations": result.fitness_evaluations,
                }
            )
    output = Path("results/summary/optimization_validation.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    frame = pd.DataFrame(rows)
    for algorithm, group in frame.groupby("Algorithm"):
        statistic, pvalue = wilcoxon(
            group["BaselineDistance"], group["CandidateDistance"], alternative="greater"
        )
        print(
            json.dumps(
                {
                    "algorithm": algorithm,
                    "mean_improvement": float(group["Improvement"].mean()),
                    "wins": int((group["Improvement"] > 0).sum()),
                    "held_out_runs": len(group),
                    "wilcoxon_one_sided_p": float(pvalue),
                    "statistic": float(statistic),
                }
            )
        )


if __name__ == "__main__":
    main()
