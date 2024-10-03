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
  priorities = np.array([priority(tuple(all_integers), n) for int in all_integers])

  # Build `set` greedily, using priorities for prioritization.
  capset = np.empty(shape=(0,), dtype=np.int32)
  while np.any(priorities != -np.inf):
    # Add a integer with maximum priority to `set`, and set priorities of
    # invalidated integers to `-inf`, so that they never get selected.
    max_index = np.argmax(priorities)
    pick_new_int = all_integers[None, max_index]  # [1, n]

    blocking = 2*pick_new_int - capset#(pick_new_int - capset) + pick_new_int
    blocking2= 2*capset - pick_new_int#pick_new_int - (pick_new_int - capset)
    blocking_all = np.concatenate([blocking, blocking2], axis=0)
    blocking_all = blocking_all[(blocking_all < n) & (blocking_all >= 0)]
    priorities[blocking_all] = -np.inf
    priorities[max_index] = -np.inf
    capset = np.concatenate([capset, pick_new_int], axis=0)

  return capset


def priority(k: int, n: int) -> float:
  """Returns the priority with which we want to add `element` to the salem-spencer set.
  n is the number of possible integers, and k is the integer we want to determine priority for. 
  """
  """Improved version of `priority_v0`.

  Returns a higher priority for `k` if it creates a larger gap with its neighbors in the Salem-Spencer set.
  """
  # Calculate the gaps between k and its neighbors (k-1 and k+1)
  gap_left = abs(k - (k - 1)) if k > 0 else 0
  gap_right = abs(k - (k + 1)) if k < n - 1 else 0

  # Prioritize larger gaps
  return gap_left + gap_right

