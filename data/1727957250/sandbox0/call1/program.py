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
  """Returns the size of an `n`-dimensional cap set."""
  capset = solve(n)
  return len(capset)


def solve(n: int) -> np.ndarray:
  """Returns a salem-spencer set for n variables"""
  all_integers = np.arange(n)

  # Powers in decreasing order for compatibility with `itertools.product`, so
  # that the relationship `i = all_vectors[i] @ powers` holds for all `i`.
  #powers = 3 ** np.arange(n - 1, -1, -1)

  # Precompute all priorities.
  priorities = np.array([priority(tuple(all_integers), n) for int in all_integers])

  # Build `capset` greedily, using priorities for prioritization.
  capset = np.empty(shape=(0,), dtype=np.int32)
  while np.any(priorities != -np.inf):
    # Add a vector with maximum priority to `capset`, and set priorities of
    # invalidated vectors to `-inf`, so that they never get selected.
    max_index = np.argmax(priorities)
    pick_new_int = all_integers[None, max_index]  # [1, n]
    #blocking = np.einsum('cn,n->c', (- capset - vector) % 3, powers)  # [C]
    blocking = 2*pick_new_int - capset#(pick_new_int - capset) + pick_new_int
    blocking2= 2*capset - pick_new_int#pick_new_int - (pick_new_int - capset)
    blocking_all = np.concatenate([blocking, blocking2], axis=0)
    blocking_all = blocking_all[(blocking_all < n) & (blocking_all >= 0)]
    priorities[blocking_all] = -np.inf
    priorities[max_index] = -np.inf
    capset = np.concatenate([capset, pick_new_int], axis=0)

  return capset


def priority(k: int, n: int) -> float:
  """Returns the priority with which we want to add `element` to the cap set.
  el is a tuple of length n with values 0-2.
  """
  """Returns the priority with which we want to add `el` to the cap set.
  el is a tuple of length n with values 0-2.
  """
  return sum(el) + len(set(el))

