"""
On every iteration, improve priority_vX+1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""
import itertools

import numpy as np

import funsearch

import math


@funsearch.run
def evaluate(n: int) -> int:
    """Returns the size of a certain `n`-dimensional cap set."""
    #capset = solve(n)
    #result = np.mean(np.equal(np.vectorize(solve)(np.arange(2, n)), np.vectorize(priority)(np.arange(2, n))))
    result=0
    return result


def solve(n):
    # if n <= 3:
    #     return False
    
    # def is_prime(n):
    #     if n <= 1:
    #         return False
    #     for i in range(2, int(math.sqrt(n)) + 1):
    #         if n % i == 0:
    #             return False
    #     return True
    return False#is_prime(n - 2)



@funsearch.evolve
def priority(m: int) -> bool:
  """ Return whether or not m is an element of the set
  """
  return False