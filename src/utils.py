"""Configuration, reproducibility, and result data structures."""
from dataclasses import dataclass
from pathlib import Path
import json, random, time
import numpy as np

@dataclass
class OptimizationResult:
    """Common optimizer output."""
    best_distance: float; best_routes: list[list[int]]; best_solution: object; history: list[dict]; runtime: float; fitness_evaluations: int; convergence: list[float]

def seed_everything(seed: int) -> None:
    """Seed Python and NumPy random generators."""
    random.seed(seed); np.random.seed(seed)

def load_config(path: str | Path) -> dict:
    """Load JSON configuration."""
    return json.loads(Path(path).read_text(encoding="utf-8"))

def timed() -> float:
    """Return a monotonic wall-clock timestamp."""
    return time.perf_counter()
