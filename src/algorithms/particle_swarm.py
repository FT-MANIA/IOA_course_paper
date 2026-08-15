"""Random-key PSO for permutation-encoded CVRP."""
import numpy as np
from src.decoder import decode_permutation, calculate_total_distance
from src.utils import OptimizationResult, timed, seed_everything

class ParticleSwarmOptimizer:
    """Standard continuous PSO whose positions decode through argsort."""
    def __init__(self, instance, config: dict, seed: int = 0):
        self.instance,self.cfg,self.seed=instance,config,seed; seed_everything(seed); self.evaluations=0

    def _evaluate(self,x):
        permutation=(np.argsort(x)+1).tolist(); routes=decode_permutation(permutation,self.instance.demands,self.instance.capacity); value=calculate_total_distance(routes,self.instance.distance_matrix); self.evaluations+=1; return value,permutation,routes

    def optimize(self,max_evaluations:int)->OptimizationResult:
        """Run random-key PSO with linearly decreasing inertia."""
        start=timed(); s=self.cfg["swarm_size"]; n=self.instance.num_customers; x=np.random.random((s,n)); v=np.random.uniform(-.2,.2,(s,n)); pbest=x.copy(); pvals=[]; p_routes=[]; p_perms=[]
        for row in x:
            val,perm,routes=self._evaluate(row); pvals.append(val); p_perms.append(perm); p_routes.append(routes)
        pvals=np.array(pvals); gi=int(np.argmin(pvals)); gbest=x[gi].copy(); best=float(pvals[gi]); best_perm=p_perms[gi]; best_routes=p_routes[gi]; convergence=[best]; history=[]; it=0
        while self.evaluations < max_evaluations:
            it+=1; w=self.cfg["w_start"]-(self.cfg["w_start"]-self.cfg["w_end"])*min(1,it/max(1,max_evaluations//s)); vals=[]
            for i in range(s):
                if self.evaluations>=max_evaluations: break
                v[i]=w*v[i]+self.cfg["c1"]*np.random.random(n)*(pbest[i]-x[i])+self.cfg["c2"]*np.random.random(n)*(gbest-x[i]); x[i]=np.clip(x[i]+v[i],0,1)
                val,perm,routes=self._evaluate(x[i]); vals.append(val)
                if val<pvals[i]: pvals[i]=val; pbest[i]=x[i].copy(); p_perms[i]=perm; p_routes[i]=routes
                if val<best: best,best_perm,best_routes=val,perm,routes; gbest=x[i].copy()
            arr=np.asarray(vals); convergence.append(best); history.append({"iteration":it,"best_distance":best,"mean_distance":float(arr.mean()),"std_distance":float(arr.std()),"fitness_evaluations":self.evaluations})
        return OptimizationResult(best,best_routes,best_perm,history,timed()-start,self.evaluations,convergence)
