"""Finds narrow admissible tuples (lists of integers that do not fill every residue class modulo any prime)"""

import funsearch, functools

@funsearch.run
def evaluate(n: int) -> int:
  """Returns the size of an `n`-dimensional cap set."""
  tuple = solve(n)
  return len(tuple)

@functools.cache
def primes(n):
    out = list()
    sieve = [True] * (n+1)
    for p in range(2, n+1):
        if (sieve[p] and sieve[p]%2==1):
            out.append(p)
            for i in range(p, n+1, p):
                sieve[i] = False
    return out

def solve(n: int) -> list:
  """Returns an admissible tuple of width (at most) n."""
  map = [True] * (n+1)
  P = primes(n)
  for p in P:
    rs = {i % p for i in range(n+1) if map[i]}
    if len(rs) < p:
      continue
    r = priority(p,n) % p
    for i in range(r,n+1,p):
      map[i] = False   
  return [i for i in range(n+1) if map[i]]

@funsearch.evolve
def priority(p: int, n: int) -> int:
  """
  Returns the nonzero residue class to avoid modulo p when constructing an admissible tuple of width n.
  """
  return 1

