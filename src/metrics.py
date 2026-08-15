"""Statistics and convergence helpers."""
from statistics import mean, median
import numpy as np

def describe(values: list[float]) -> dict[str, float]:
    """Compute requested descriptive statistics."""
    a = np.asarray(values, dtype=float)
    return {"Mean": float(np.mean(a)), "Std": float(np.std(a, ddof=1)), "Median": float(np.median(a)), "Min": float(np.min(a)), "Max": float(np.max(a)), "IQR": float(np.percentile(a, 75) - np.percentile(a, 25))}

def previous_best_interpolate(evaluations: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Sample a step-wise best-so-far trace on a common evaluation grid."""
    out = np.empty_like(grid, dtype=float); idx = 0; current = values[0]
    for i, x in enumerate(grid):
        while idx + 1 < len(evaluations) and evaluations[idx + 1] <= x: idx += 1; current = values[idx]
        out[i] = current
    return out

def convergence90(trace: list[float]) -> int:
    """Return the first trace index reaching 90 percent of total improvement."""
    initial, final = trace[0], trace[-1]; target = initial - 0.9 * (initial - final)
    return next((i for i, v in enumerate(trace) if v <= target), len(trace) - 1)
