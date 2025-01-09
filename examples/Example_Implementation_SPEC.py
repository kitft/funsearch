"""<<<E.g. Finds EXAMPLE_PROBLEMS. A brief description of the problem you're solving might be a 
good idea to include here, as it will go in to the prompt>>>

<<<THE BELOW SHOULD ALWAYS BE INCLUDED>>>
On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""
import itertools ####<<<IF NECESSARY>>>>
import numpy as np
import funsearch

@funsearch.run #####<<<< THIS IS THE ENTRY POINT for funsearchy, and returns the score!>>>######
def evaluate(n: int) -> int:
    """Returns the size of an `n`-dimensional EXAMPLE_PROBLEM"""
    example_output = solve(n)
    return len(example_output)#### <<<<<THIS OUTPUT BECOMES THE SCORE>>>>>>

def solve(n: int) -> np.ndarray:
    """Returns an EXAMPLE in `n` dimensions."""
    ####<<<AT SOME POINT YOU SHOULD CALL "priority(<whatever type you've specified below>)">>>
    pass  # TODO: Implement the solve function

@funsearch.evolve ####<<<< THIS TELLS FUNSEARCH WHAT TO EVOLVE>>>######
def priority(el: tuple[int, ...], n: int) -> float: ### <<<MODIFY THE TYPE SIGNATURE IF NECESSARY. 
    #### THE EVO algorithm doesn't care what the type sig is, as it simply calls evaluate() to get the score>>>
    #### HOWEVER, this function should still be called 'priority'
    """Returns the priority with which we want to add `element` to the EXAMPLE_PROBLEM. <<BRIEF DESCRIPTION OF THE SPECIFIC PRIORITY PROBLEM IS A GOOD IDEA HERE>>>
    el is a tuple of length n with values 0-2.
    """
    pass  # TODO: Implement the priority function



