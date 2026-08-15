"""Publication-friendly matplotlib figures without seaborn."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from src.metrics import previous_best_interpolate

def _save(fig, path):
    """Save a figure in PNG and, when requested by caller, PDF."""
    fig.savefig(path, dpi=300, bbox_inches="tight"); fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)

def plot_instance(instance, path):
    """Plot depot and demand-sized customer points."""
    fig,ax=plt.subplots(figsize=(7,6)); d=instance.demands[1:]; ax.scatter(instance.coordinates[1:,0],instance.coordinates[1:,1],s=25+d*5,c=d,cmap="viridis",label="Customers"); ax.scatter(*instance.coordinates[0],marker="*",s=220,c="red",label="Depot");
    for i,(x,y) in enumerate(instance.coordinates[1:],1): ax.annotate(str(i),(x,y),fontsize=6,xytext=(2,2),textcoords="offset points")
    ax.set(xlabel="X coordinate",ylabel="Y coordinate",title="Fixed CVRP Instance"); ax.legend(); ax.grid(alpha=.25); _save(fig,path)

def plot_convergence(traces, path, max_eval):
    """Plot mean plus/minus standard deviation on a common evaluation grid."""
    grid=np.linspace(1,max_eval,200); fig,ax=plt.subplots(figsize=(8,5))
    for name,records in traces.items():
        curves=[]
        for rec in records:
            if rec and isinstance(rec[0], dict):
                ev=np.array([0]+[h["fitness_evaluations"] for h in rec]); vals=np.array([rec[0]["best_distance"]]+[h["best_distance"] for h in rec])
            else:
                ev=np.linspace(0,max_eval,len(rec)); vals=np.asarray(rec)
            curves.append(previous_best_interpolate(ev,vals,grid))
        arr=np.array(curves); m=arr.mean(0); sd=arr.std(0); ax.plot(grid,m,label=name); ax.fill_between(grid,m-sd,m+sd,alpha=.18)
    ax.set(xlabel="Fitness Evaluations",ylabel="Best-so-far Total Distance",title="Convergence Comparison"); ax.grid(alpha=.25); ax.legend(); _save(fig,path)

def plot_boxplot(values, path):
    """Plot final-distance boxplots with raw jitter points."""
    fig,ax=plt.subplots(figsize=(7,5)); names=list(values); data=[values[n] for n in names]; ax.boxplot(data,labels=names,showfliers=True)
    rng=np.random.default_rng(123); 
    for i,a in enumerate(data,1): ax.scatter(i+rng.uniform(-.08,.08,len(a)),a,s=12,alpha=.55)
    ax.set(xlabel="Algorithm",ylabel="Best Total Distance",title="Final Solution Quality"); ax.grid(axis="y",alpha=.25); _save(fig,path)

def plot_route(instance, routes, algorithm, distance, path):
    """Plot a solution with one color per vehicle route."""
    fig,ax=plt.subplots(figsize=(7,6)); colors=plt.cm.tab20(np.linspace(0,1,max(1,len(routes))))
    for color,route in zip(colors,routes):
        xy=instance.coordinates[route]; ax.plot(xy[:,0],xy[:,1],"-o",color=color,ms=3)
    ax.scatter(*instance.coordinates[0],marker="*",s=220,c="black",label="Depot"); ax.set(xlabel="X coordinate",ylabel="Y coordinate",title=f"{algorithm} Best Solution\nTotal Distance = {distance:.2f} | Vehicles = {len(routes)}"); ax.grid(alpha=.25); ax.legend(); _save(fig,path)

def plot_sensitivity(rows, algorithm, parameter, path):
    """Plot mean and standard deviation from one-factor sensitivity runs."""
    vals=sorted(set(float(r[parameter]) for r in rows)); means=[np.mean([r["BestDistance"] for r in rows if float(r[parameter])==v]) for v in vals]; sds=[np.std([r["BestDistance"] for r in rows if float(r[parameter])==v],ddof=1) for v in vals]; fig,ax=plt.subplots(figsize=(7,5)); ax.errorbar(vals,means,yerr=sds,fmt="-o",capsize=3); ax.set(xlabel=parameter,ylabel="Final Best Distance",title=f"{algorithm} Parameter Sensitivity"); ax.grid(alpha=.25); _save(fig,path)
