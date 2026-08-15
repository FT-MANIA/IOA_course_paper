"""Run one algorithm and print its solution summary."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.problem import load_instance, generate_instance
from src.utils import load_config
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.ant_colony import AntColonyOptimizer
from src.algorithms.particle_swarm import ParticleSwarmOptimizer

def main():
    p=argparse.ArgumentParser(); p.add_argument("--algorithm",choices=["ga","aco","pso"],required=True); p.add_argument("--seed",type=int,default=0); p.add_argument("--config",default="configs/default_config.json"); p.add_argument("--evaluations",type=int); a=p.parse_args(); c=load_config(a.config); path=Path("data/cvrp_instance.csv"); inst=load_instance(path,c["problem"]["vehicle_capacity"]) if path.exists() else generate_instance(path,c["problem"]["num_customers"],c["problem"]["vehicle_capacity"],c["problem"]["problem_seed"])
    cls,key={"ga":(GeneticAlgorithm,"ga"),"aco":(AntColonyOptimizer,"aco"),"pso":(ParticleSwarmOptimizer,"pso")}[a.algorithm]; result=cls(inst,c[key],a.seed).optimize(a.evaluations or c["experiment"]["max_fitness_evaluations"]); print(json.dumps({"algorithm":a.algorithm,"seed":a.seed,"best_distance":result.best_distance,"vehicles":len(result.best_routes),"fitness_evaluations":result.fitness_evaluations,"runtime_seconds":result.runtime},indent=2))
if __name__ == "__main__": main()
