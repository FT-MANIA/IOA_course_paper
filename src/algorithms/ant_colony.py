"""Capacity-aware ant colony optimization for CVRP."""
import random
import numpy as np
from src.decoder import calculate_total_distance, validate_solution
from src.utils import OptimizationResult, timed, seed_everything

class AntColonyOptimizer:
    """ACO using feasible customer transitions and global/iteration-best reinforcement."""
    def __init__(self, instance, config: dict, seed: int = 0):
        self.instance, self.cfg, self.seed = instance, config, seed; seed_everything(seed); self.evaluations=0

    def _construct(self, tau, eta):
        unvisited=set(range(1, self.instance.num_customers+1)); routes=[]; route=[0]; load=0.0; current=0
        while unvisited:
            allowed=[j for j in unvisited if load + self.instance.demands[j] <= self.instance.capacity]
            if not allowed:
                route.append(0); routes.append(route); route=[0]; load=0.0; current=0; continue
            weights=np.array([(tau[current,j]**self.cfg["alpha"])*(eta[current,j]**self.cfg["beta"]) for j in allowed],float)
            probs=weights/weights.sum() if weights.sum() else np.ones(len(allowed))/len(allowed)
            nxt=int(np.random.choice(allowed,p=probs)); route.append(nxt); unvisited.remove(nxt); load+=self.instance.demands[nxt]; current=nxt
        route.append(0); routes.append(route); return routes

    def optimize(self, max_evaluations: int) -> OptimizationResult:
        """Run feasible ant construction under the objective budget."""
        start=timed(); d=self.instance.distance_matrix; n=len(d); eta=np.zeros_like(d); eta[d>0]=1/d[d>0]; tau=np.ones((n,n)); best=float("inf"); best_routes=[]; convergence=[]; history=[]; iteration=0
        while self.evaluations < max_evaluations:
            iteration+=1; colony=[]
            for _ in range(self.cfg["num_ants"]):
                if self.evaluations >= max_evaluations: break
                routes=self._construct(tau,eta); value=calculate_total_distance(routes,d); self.evaluations+=1; colony.append((value,routes))
            if not colony: break
            colony.sort(key=lambda x:x[0]); ib=colony[0]
            if ib[0]<best: best,best_routes=ib[0],ib[1]
            tau *= (1-self.cfg["rho"])
            for value,routes in colony:
                deposit=self.cfg["q"]/value
                for route in routes:
                    for a,b in zip(route,route[1:]): tau[a,b]+=deposit; tau[b,a]+=deposit
            for route in best_routes:
                for a,b in zip(route,route[1:]): tau[a,b]+=self.cfg["q"]/best; tau[b,a]+=self.cfg["q"]/best
            vals=np.array([x[0] for x in colony]); convergence.append(best); history.append({"iteration":iteration,"best_distance":best,"mean_distance":float(vals.mean()),"std_distance":float(vals.std()),"fitness_evaluations":self.evaluations})
        validate_solution(best_routes,self.instance.demands,self.instance.capacity,self.instance.num_customers)
        return OptimizationResult(best,best_routes,[c for r in best_routes for c in r if c],history,timed()-start,self.evaluations,convergence)
