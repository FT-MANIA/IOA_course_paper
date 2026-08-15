"""CVRP instance representation and reproducible CSV generation."""
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np

@dataclass(frozen=True)
class CVRPInstance:
    """A depot plus customer coordinates and integer demands."""
    coordinates: np.ndarray
    demands: np.ndarray
    capacity: float

    @property
    def num_customers(self) -> int:
        return len(self.demands) - 1

    @property
    def distance_matrix(self) -> np.ndarray:
        delta = self.coordinates[:, None, :] - self.coordinates[None, :, :]
        return np.sqrt((delta * delta).sum(axis=2))

def generate_instance(path: str | Path, num_customers: int, capacity: float, seed: int) -> CVRPInstance:
    """Generate and persist one deterministic clustered-but-spread CVRP instance."""
    rng = np.random.default_rng(seed)
    coords = np.vstack(([50.0, 50.0], rng.uniform(0, 100, size=(num_customers, 2))))
    demands = np.concatenate(([0.0], rng.integers(5, 26, size=num_customers).astype(float)));
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["node_id", "x", "y", "demand", "is_depot"])
        for i, (xy, d) in enumerate(zip(coords, demands)):
            writer.writerow([i, f"{xy[0]:.8f}", f"{xy[1]:.8f}", f"{d:.0f}", int(i == 0)])
    return CVRPInstance(coords, demands, capacity)

def load_instance(path: str | Path, capacity: float) -> CVRPInstance:
    """Load a previously generated instance CSV."""
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    rows.sort(key=lambda r: int(r["node_id"]))
    coords = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    demands = np.array([float(r["demand"]) for r in rows])
    if not rows or int(rows[0]["is_depot"]) != 1:
        raise ValueError("CSV must contain depot as node 0")
    return CVRPInstance(coords, demands, capacity)
