"""Finds subsets of an nxn grid that contains 3 points in a line

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""
import itertools
import numpy as np
import funsearch
import itertools


def are_collinear(p1, p2, p3):
  """Returns True if the three points are collinear."""
  x1, y1 = p1
  x2, y2 = p2
  x3, y3 = p3
  return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)


@funsearch.run
def evaluate(n: int) -> int:
  """Returns the size of a point set on the nxn grid that does not contain 3 points in a line."""
  capset = solve(n)
  return len(capset),capset


def solve(n: int) -> np.ndarray:
  """Returns a large non-3-points-in-line subset on the nxn grid."""
  all_points = np.array(list(itertools.product(np.arange(n), repeat=2)), dtype=np.int32)
  #all_integers = np.arange(n)

  
  # Precompute all priorities.
  priorities = np.array([priority(tuple(point), n) for point in all_points])


  # Build `capset` greedily, using priorities for prioritization.
  capset = np.empty(shape=(0,2), dtype=np.int32)
  while np.any(priorities != -np.inf):
    # Add a vector with maximum priority to `capset`, and set priorities of
    # invalidated vectors to `-inf`, so that they never get selected.
    max_index = np.argmax(priorities)
    new_points = all_points[None, max_index]  # [1, n]
    new_point = new_points[0]
    priorities[max_index] = -np.inf

    # Block those points in all_points which lie on a line spanned by a point in capset and new_point

    for index in range(len(all_points)):
      if capset.size == 0:
          continue
      elif any(are_collinear(new_point, all_points[index], cap) for cap in capset):
        priorities[index] = -np.inf

    capset = np.concatenate([capset, new_point.reshape(1, -1)], axis=0)
    

  return capset


@funsearch.evolve
def priority(el: tuple[int, int], n: int) -> float:
  """Returns the priority with which we want to add `element` to the cap set."""
  return 0.0
     