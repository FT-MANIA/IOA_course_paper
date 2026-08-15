"""Common split decoder and route distance functions."""
from typing import Iterable, Sequence
import numpy as np

def decode_permutation(permutation: Sequence[int], demands: Sequence[float], capacity: float) -> list[list[int]]:
    """Split a customer permutation into capacity-feasible depot-delimited routes."""
    routes: list[list[int]] = []; route = [0]; load = 0.0
    for customer in permutation:
        demand = float(demands[customer])
        if demand > capacity: raise ValueError(f"Demand of customer {customer} exceeds capacity")
        if load and load + demand > capacity:
            route.append(0); routes.append(route); route = [0]; load = 0.0
        route.append(int(customer)); load += demand
    if len(route) > 1: route.append(0); routes.append(route)
    return routes

def calculate_total_distance(routes: Iterable[Sequence[int]], distance_matrix: np.ndarray) -> float:
    """Return the sum of consecutive edge distances in all routes."""
    return float(sum(distance_matrix[a, b] for route in routes for a, b in zip(route, route[1:])))

def validate_solution(routes: Sequence[Sequence[int]], demands: Sequence[float], capacity: float, num_customers: int) -> None:
    """Raise ValueError unless routes form a valid complete CVRP solution."""
    seen: list[int] = []
    for route in routes:
        if len(route) < 3 or route[0] != 0 or route[-1] != 0: raise ValueError("Each route must start/end at depot")
        customers = list(route[1:-1]); load = sum(float(demands[c]) for c in customers)
        if load > capacity + 1e-9: raise ValueError("Route capacity exceeded")
        seen.extend(customers)
    expected = list(range(1, num_customers + 1))
    if sorted(seen) != expected: raise ValueError("Customers are missing or duplicated")
