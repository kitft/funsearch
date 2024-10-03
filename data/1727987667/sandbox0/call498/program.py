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
  """Improved version of `priority_v1`.

  The priority function is now based on the size of the gap between consecutive numbers in the Salem-Spencer sequence.
  We want to maximize this gap, as it increases the "spread" of the sequence and makes it less likely that new numbers
  will be close to any existing numbers in the sequence.

  The priority of a number is calculated as the product of two factors:
  1. The size of the gap between the number and the previous number, normalized by the total number of possible integers.
  2. The size of the gap between the number and the next number, normalized by the total number of possible integers.

  The product of these two factors gives a measure of the "desirability" of the number as the next element in the sequence.
  """
  # Get the current Salem-Spencer sequence
  ss_sequence = funsearch.get_current_sequence()

  # Find the previous and next numbers in the sequence
  prev_num = max([num for num in ss_sequence if num < k], default=0)
  next_num = min([num for num in ss_sequence if num > k], default=n)

  # Calculate the size of the gaps between the number and its neighbors
  gap_before = (k - prev_num) / n
  gap_after = (next_num - k) / n

  # Return the product of the gap sizes as the priority
  return gap_before * gap_after

