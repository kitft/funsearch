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
    """Improved version of `priority_v8`.

    The priority_v9 function calculates the priority of a given element `el` for adding it to the cap set.
    The element `el` is a tuple of length `n` with values 0-2.

    To improve upon the previous version, we can consider using a combination of the sum, the sum of squares,
    the number of 2's, the number of 1's, the number of 0's, the number of unique elements,
    the number of pairs of the same element, and the number of occurrences of each value in the tuple as the priority.
    This way, we prioritize tuples with a larger sum, a larger sum of squares, a larger number of 2's, a smaller number of 1's,
    a smaller number of 0's, a smaller number of unique elements, a smaller number of pairs of the same element,
    and a more balanced distribution of values in the tuple, as they are likely to be more beneficial for the cap set.

    Args:
        el: A tuple of length `n` with values 0-2.
        n: The length of the tuple `el`.

    Returns:
        The priority of the element `el` for adding it to the cap set.
    """
    return sum(el) + sum(x**2 for x in el) + el.count(2) - el.count(1) - el.count(0) - len(set(el)) - sum(el.count(i) > 1 for i in range(3)) + np.var(el)

