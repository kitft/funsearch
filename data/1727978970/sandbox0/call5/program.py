"""Finds large cap sets.

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
  """Returns a large cap set in `n` dimensions."""
  all_vectors = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)

  # Powers in decreasing order for compatibility with `itertools.product`, so
  # that the relationship `i = all_vectors[i] @ powers` holds for all `i`.
  powers = 3 ** np.arange(n - 1, -1, -1)

  # Precompute all priorities.
  priorities = np.array([priority(tuple(vector), n) for vector in all_vectors])

  # Build `capset` greedily, using priorities for prioritization.
  capset = np.empty(shape=(0, n), dtype=np.int32)
  while np.any(priorities != -np.inf):
    # Add a vector with maximum priority to `capset`, and set priorities of
    # invalidated vectors to `-inf`, so that they never get selected.
    max_index = np.argmax(priorities)
    vector = all_vectors[None, max_index]  # [1, n]
    blocking = np.einsum('cn,n->c', (- capset - vector) % 3, powers)  # [C]
    priorities[blocking] = -np.inf
    priorities[max_index] = -np.inf
    capset = np.concatenate([capset, vector], axis=0)

  return capset


def priority(el: tuple[int, ...], n: int) -> float:
  """Returns the priority with which we want to add `element` to the cap set.
  el is a tuple of length n with values 0-2.
  """
  """Improved version of `priority_v8`."""
  # Prioritize elements with a higher ratio of 2's to total elements, a lower ratio of 1's to total elements, a higher number of consecutive 2's, a lower number of consecutive 1's, and a higher number of 0's at the ends of the element
  return el.count(2) / n - el.count(1) / n + el.count(0) / n + max(sum(1 for _ in group) for key, group in itertools.groupby(el) if key == 2) - max(sum(1 for _ in group) for key, group in itertools.groupby(el) if key == 1) + el.count(0, el.index(1 if 1 in el else n)) / n + el.count(0, el.rindex(1 if 1 in el else -1)) / n

