"""Finds large SS progressions 

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""
import itertools

import numpy as np

import funsearch


@funsearch.run
def evaluate(n: int) -> int:
  """Returns the size of a salem-spencer set for n variables"""
  capset = solve(n)
  return len(capset)


def solve(n: int) -> np.ndarray:
  """Returns a salem-spencer set for n variables"""
  all_integers = np.arange(n)

  # Precompute all priorities.
  priorities = np.array([priority(int, n) for int in all_integers])

  # Build `set` greedily, using priorities for prioritization.
  capset = np.empty(shape=(0,), dtype=np.int32)
  while np.any(priorities != -np.inf):
    # Add a integer with maximum priority to `set`, and set priorities of
    # invalidated integers to `-inf`, so that they never get selected.
    max_index = np.argmax(priorities)
    pick_new_int = all_integers[None, max_index]  # [1, n]

    #identify the elements which would form part of an arithmetic progression
    blocking = 2*pick_new_int - capset
    blocking2= 2*capset - pick_new_int
    blocking3 = (pick_new_int + capset) / 2
    blocking3 = np.round(blocking3[np.isclose(blocking3, np.round(blocking3))]).astype(int)

    #remove those elements from the priority list
    blocking_all = np.concatenate([blocking, blocking2,blocking3], axis=0)
    blocking_all = blocking_all[(blocking_all < n) & (blocking_all >= 0)]
    priorities[blocking_all] = -np.inf
    priorities[max_index] = -np.inf
    capset = np.concatenate([capset, pick_new_int], axis=0)

  return capset


def priority(k: int, n: int) -> float:
  """Returns the priority with which we want to add `element` to the salem-spencer set.
  n is the number of possible integers, and k is the integer we want to determine priority for. 
  """
  """Improved version of `priority_v3`.

  This version takes into account the order in which elements are added to the Salem-Spencer set.
  The earlier an element is added, the higher its priority.

  Args:
    k: The integer for which we want to determine priority.
    n: The total number of possible integers.

  Returns:
    The priority for adding `k` to the Salem-Spencer set.
  """
  # Get the current Salem-Spencer set.
  ss_set = funsearch.get_salem_spencer_set()

  # Calculate the average gap between elements in the set.
  if len(ss_set) < 2:
    avg_gap = 1.0
  else:
    avg_gap = np.mean(np.diff(sorted(ss_set)))

  # Calculate the variance of the gaps between elements in the set.
  if len(ss_set) < 3:
    gap_variance = 0.0
  else:
    gap_variance = np.var(np.diff(sorted(ss_set)))

  # Calculate the order in which elements are added to the set.
  order = len(ss_set) + 1

  # Calculate the priority as a combination of the integer value, the set size, the average gap, the gap variance, and the order.
  return (k / n) * (len(ss_set) + 1) * (1.0 / avg_gap) * (1.0 + gap_variance) * (1.0 / order)

