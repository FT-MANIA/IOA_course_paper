"""Run the reproducible CVRP GA/ACO/PSO benchmark and generate all outputs."""
import argparse, csv, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from src.problem import generate_instance, load_instance
from src.utils import load_config
from src.decoder import validate_solution
from src.metrics import describe, convergence90
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.ant_colony import AntColonyOptimizer
from src.algorithms.particle_swarm import ParticleSwarmOptimizer
from src.visualization.plots import plot_instance, plot_convergence, plot_boxplot, plot_route, plot_sensitivity

ALGORITHMS={"GA":(GeneticAlgorithm,"ga"),"ACO":(AntColonyOptimizer,"aco"),"PSO":(ParticleSwarmOptimizer,"pso")}

def execute_job(job):
    """Execute one independent optimizer run; suitable for process workers."""
    name, instance, algorithm_config, seed, budget = job
    result = ALGORITHMS[name][0](instance, algorithm_config, seed).optimize(budget)
    validate_solution(result.best_routes, instance.demands, instance.capacity, instance.num_customers)
    return seed, result

def execute_jobs(name, instance, algorithm_config, seeds, budget, workers):
    """Run independent seeds sequentially or in separate processes."""
    jobs=[(name,instance,algorithm_config,seed,budget) for seed in seeds]
    if workers <= 1:
        return [execute_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(execute_job,jobs))

def write_csv(path, rows, fields):
    """Write dictionaries as UTF-8 CSV."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def statistical_tests(values):
    """Run requested non-parametric tests when SciPy is installed."""
    try:
        from scipy.stats import kruskal, mannwhitneyu
    except ImportError: return [{"Test":"Unavailable","Statistic":"","PValue":"","AdjustedPValue":""}]
    names=list(values); groups=[values[n] for n in names]; rows=[]; stat,p=kruskal(*groups); rows.append({"Test":"Kruskal-Wallis","Statistic":stat,"PValue":p,"AdjustedPValue":p})
    pairs=[]
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            s,p=mannwhitneyu(values[names[i]],values[names[j]],alternative="two-sided"); pairs.append({"Test":f"{names[i]} vs {names[j]}","Statistic":s,"PValue":p})
    ps=np.array([r["PValue"] for r in pairs]); order=np.argsort(ps); adj=np.empty(len(ps)); running=1.0
    for rank,idx in reversed(list(enumerate(order,1))): running=min(running,ps[idx]*len(ps)/rank); adj[idx]=running
    for r,a in zip(pairs,adj): r["AdjustedPValue"]=a; rows.append(r)
    return rows

def report(summary, tests, best_mean, robust, fastest, config):
    """Generate a concise data-driven interpretation for the paper."""
    lines=["# Experiment Report","",f"The benchmark used a fixed {config['problem']['num_customers']}-customer instance, capacity {config['problem']['vehicle_capacity']}, {config['experiment']['num_runs']} independent seeds, and a common budget of {config['experiment']['max_fitness_evaluations']} objective evaluations.","","## Summary","","| Algorithm | Mean | Std | Median | Min | Max | IQR | Mean runtime |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary: lines.append(f"| {r['Algorithm']} | {r['Mean']:.2f} | {r['Std']:.2f} | {r['Median']:.2f} | {r['Min']:.2f} | {r['Max']:.2f} | {r['IQR']:.2f} | {r['MeanRuntime']:.3f} |")
    lines += ["",f"Based on observed means, the best mean performance was **{best_mean}**; the smallest standard deviation (robustness proxy) was **{robust}**; and the lowest observed 90% convergence evaluation was **{fastest}**.","","## Statistical analysis","", "Kruskal-Wallis and pairwise Mann-Whitney U tests with Benjamini-Hochberg correction are reported in `statistical_tests.csv`. These tests compare the 30 final distances and do not assume normality.","","## Exploration–Exploitation Analysis","","- **GA:** random initialization, OX crossover, and inversion mutation provide exploration; tournament selection and elitism exploit good permutations. A very low mutation rate can cause premature convergence, while a high rate can disrupt useful route adjacency.","- **ACO:** probabilistic feasible path construction and pheromone evaporation support exploration; heuristic distance information and pheromone reinforcement support exploitation. High beta can over-prefer nearby nodes, while low rho can preserve stale trails too long.","- **PSO:** inertia and stochastic coefficients support exploration; pbest/gbest attraction supports exploitation. Random-key sorting makes continuous updates easy to implement, but small Euclidean movement in key space does not always imply a small route change, creating a representation mismatch.","","## Interpretation","","The conclusions above are generated from the measured outputs rather than a preset ranking. Runtime is hardware-dependent and is treated as auxiliary evidence; distance quality, dispersion, evaluation-based convergence, and vehicle feasibility are the primary comparison criteria."]
    return "\n".join(lines)+"\n"

def run(config, quick=False, workers=1):
    """Execute benchmark, sensitivity analysis, plots, summary, and report."""
    if quick:
        config=json.loads(json.dumps(config)); config["experiment"]["num_runs"]=3; config["experiment"]["max_fitness_evaluations"]=2000; config["experiment"]["sensitivity_runs"]=2; config["experiment"]["sensitivity_fitness_evaluations"]=2000
    root=Path("."); out=root/"results"; (out/"raw").mkdir(parents=True,exist_ok=True); (out/"summary").mkdir(parents=True,exist_ok=True); (out/"figures").mkdir(parents=True,exist_ok=True)
    data_path=root/"data/cvrp_instance.csv"; inst=load_instance(data_path,config["problem"]["vehicle_capacity"]) if data_path.exists() else generate_instance(data_path,config["problem"]["num_customers"],config["problem"]["vehicle_capacity"],config["problem"]["problem_seed"])
    plot_instance(inst,out/"figures/cvrp_instance.png"); rows=[]; traces={}; route_best={}; values={}; convergence_points={}
    for name,(cls,key) in ALGORITHMS.items():
        traces[name]=[]; values[name]=[]; route_best[name]=None; convergence_points[name]=[]
        completed=execute_jobs(name,inst,config[key],range(config["experiment"]["num_runs"]),config["experiment"]["max_fitness_evaluations"],workers)
        for seed,result in completed:
            values[name].append(result.best_distance); traces[name].append(result.history)
            if result.history:
                curve=[result.history[0]["best_distance"]]+[h["best_distance"] for h in result.history]
                point=convergence90(curve); convergence_points[name].append(0 if point == 0 else result.history[point-1]["fitness_evaluations"])
            else:
                convergence_points[name].append(result.fitness_evaluations)
            if route_best[name] is None or result.best_distance<route_best[name][0]: route_best[name]=(result.best_distance,result.best_routes)
            rows.append({"Algorithm":name,"Seed":seed,"BestDistance":result.best_distance,"RuntimeSeconds":result.runtime,"FitnessEvaluations":result.fitness_evaluations,"NumberVehicles":len(result.best_routes),"BestRoutes":json.dumps(result.best_routes)})
        write_csv(out/"raw"/f"convergence_{name}.csv",[{"Seed":i,"Evaluation":h["fitness_evaluations"],"BestDistance":h["best_distance"]} for i,t in enumerate(traces[name]) for h in t], ["Seed","Evaluation","BestDistance"])
    write_csv(out/"raw/all_runs.csv",rows,list(rows[0])); plot_convergence(traces,out/"figures/convergence_curve.png",config["experiment"]["max_fitness_evaluations"]); plot_boxplot(values,out/"figures/final_distance_boxplot.png")
    summary=[]
    for name in ALGORITHMS:
        d=describe(values[name]); summary.append({"Algorithm":name,**d,"MeanRuntime":float(np.mean([r["RuntimeSeconds"] for r in rows if r["Algorithm"]==name]))})
        plot_route(inst,route_best[name][1],name,route_best[name][0],out/"figures"/f"best_route_{name}.png")
    write_csv(out/"summary/summary_statistics.csv",summary,list(summary[0])); write_csv(out/"summary/statistical_tests.csv",statistical_tests(values),["Test","Statistic","PValue","AdjustedPValue"])
    conv={name:float(np.mean(convergence_points[name])) for name in ALGORITHMS}; best_mean=min(summary,key=lambda r:r["Mean"])["Algorithm"]; robust=min(summary,key=lambda r:r["Std"])["Algorithm"]; fastest=min(conv,key=conv.get)
    convergence_rows=[{"Algorithm":name,"MeanEvaluationsTo90":conv[name],"StdEvaluationsTo90":float(np.std(convergence_points[name],ddof=1)),"MedianEvaluationsTo90":float(np.median(convergence_points[name]))} for name in ALGORITHMS]
    write_csv(out/"summary/convergence_metrics.csv",convergence_rows,list(convergence_rows[0]))
    sensitivity=[]; sens_specs={"GA":("mutation_rate",[.05,.1,.2,.3,.4]),"ACO":("beta",[1,2,3,4,5]),"PSO":("w_end",[.2,.3,.4,.5,.6])}; sens_budget=config["experiment"].get("sensitivity_fitness_evaluations",config["experiment"]["max_fitness_evaluations"])
    for name,(param,levels) in sens_specs.items():
        base=config[ALGORITHMS[name][1]]
        for level in levels:
            local=dict(base); local[param]=level
            completed=execute_jobs(name,inst,local,range(config["experiment"]["sensitivity_runs"]),sens_budget,workers)
            for seed,result in completed:
                sensitivity.append({"Algorithm":name,param:level,"Seed":seed,"BestDistance":result.best_distance})
        plot_sensitivity([r for r in sensitivity if r["Algorithm"]==name],name,param,out/"figures"/f"parameter_sensitivity_{name}.png")
    write_csv(out/"summary/parameter_sensitivity.csv",sensitivity,["Algorithm","mutation_rate","beta","w_end","Seed","BestDistance"])
    (out/"summary/experiment_report.md").write_text(report(summary,statistical_tests(values),best_mean,robust,fastest,config),encoding="utf-8")
    return summary,best_mean,robust,fastest

def main():
    """Parse CLI arguments and run the experiment."""
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default_config.json"); p.add_argument("--quick",action="store_true"); p.add_argument("--workers",type=int,default=1); a=p.parse_args(); c=load_config(a.config); summary,best,robust,fastest=run(c,a.quick,a.workers)
    print("="*52); print("CVRP Metaheuristic Benchmark"); print("="*52); print(f"Customers: {c['problem']['num_customers']} | Capacity: {c['problem']['vehicle_capacity']}"); print(f"Runs per algorithm: {3 if a.quick else c['experiment']['num_runs']} | Evaluation budget: {2000 if a.quick else c['experiment']['max_fitness_evaluations']}"); print("-"*52)
    for r in summary: print(f"{r['Algorithm']}: Mean={r['Mean']:.2f}, Std={r['Std']:.2f}, Best={r['Min']:.2f}")
    print("-"*52); print(f"Best mean performance: {best}\nMost robust: {robust}\nFastest convergence: {fastest}\nResults saved to: results/")
if __name__=="__main__": main()
