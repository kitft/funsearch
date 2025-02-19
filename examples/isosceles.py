"""Finds a large isosceles-free subset of an n by n integer lattice
"""

import itertools
import numpy as np
import funsearch

@funsearch.run 
def evaluate(n: int) -> int:
    """Returns the size of an isosceles-free subset of an n by n integer lattice."""
    subset = solve(n)
    return len(subset)

def solve(n: int) -> list[tuple[int, int]]:
    """Returns a large isosceles-free subset of an n by n integer lattice."""
    # Generate all possible points in the n x n lattice
    all_points = list(itertools.product(range(n), repeat=2))  # List of tuples (x, y)
    
    # Precompute priorities for all points
    priorities = np.array([priority(point, n) for point in all_points], dtype=float)
    
    # Initialize the isosceles-free subset
    subset = []
    
    # Add points to the subset in order of their priority
    while np.any(priorities != -np.inf):
        # Find the point with the highest priority
        max_index = np.argmax(priorities)
        point = all_points[max_index]
        
        # Check if adding this point creates an isosceles triangle
        if not forms_isosceles_triangle(subset, point):
            subset.append(point)
        
        # Mark this point as processed
        priorities[max_index] = -np.inf
    
    return subset

def forms_isosceles_triangle(subset: list[tuple[int, int]], new_point: tuple[int, int]) -> bool:
    """
    Checks if adding `new_point` to `subset` creates an isosceles triangle.
    An isosceles triangle is a set of three points where at least two pairs of points
    have the same Euclidean distance.
    """
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            # Calculate distances between all pairs of points
            d1 = distance(subset[i], subset[j])
            d2 = distance(subset[i], new_point)
            d3 = distance(subset[j], new_point)
            
            # Check if any two distances are equal (isosceles condition)
            if d1 == d2 or d1 == d3 or d2 == d3:
                return True
    return False

def distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Returns the Euclidean distance between two points."""
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
            
@funsearch.evolve 
def priority(v: tuple[int,...], n: int) -> float: 
    """Returns the priority, as a floating number, of the vector v denoting the coordinates of a point in the n by n integer lattice. 
    The priority function will be used to construct an isosceles-free subset of the lattice.
    """
    return 0.0


