"""Finds sets

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""
import itertools
import numpy as np
import funsearch
import math

@funsearch.run
def evaluate(n: int) -> int:
  return solve(n)

def solve(n: int) -> int:
  """Returns the count of numbers between 5 and n-1 where is_prime(m) matches (m)."""
  final_count = 0
  for m in range(5, n):
      is_prime = True
      if m <= 1:
          is_prime = False
      else:
          for i in range(2, int(math.sqrt(m)) + 1):
              if m % i == 0:
                  is_prime = False
                  break
      pr = priority(m)>0.5
      if is_prime == pr:
          final_count = final_count +   1
  return final_count
  

@funsearch.evolve
def priority(n: int) -> float:
  """Returns 1 if add to set, 0 otherwise.
  n is an int.
  """
  return 0.0