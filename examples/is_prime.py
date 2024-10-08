"""


On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""
import itertools

import numpy as np

import funsearch

import math

def is_prime(n):
    n=n-2
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


@funsearch.run
def evaluate(n: int) -> float:
  """Returns the size of a certain `n`-dimensional cap set."""
  #capset = solve(n)
  result = np.mean(np.equal(np.vectorize(is_prime)(np.arange(5, n)), np.vectorize(priority)(np.arange(5, n))))
  return result


@funsearch.evolve
def priority(m: int) -> bool:
  """ Return whether or not m is an element of the set
  """
  return False