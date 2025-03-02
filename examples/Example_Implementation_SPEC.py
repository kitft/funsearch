### SYSTEM PROMPT <<<< MODIFY everything contained in the triple quotes, OR SIMPLY DON'T INCLUDE IT TO USE THE DEFAULT SYSTEM PROMPT FROM CONFIG.PY. THE SYSTEM PROMPT ALSO LISTS PACKAGES THAT CAN BE USED >>>>>>
"""You are a state-of-the-art python code completion system that will be used as part of a genetic algorithm to evolve cap sets.
You will be given a list of functions, and you should improve the incomplete last function in the list.
1. Make only small changes but be sure to make some change.
2. Try to keep the code short and any comments concise.
3. Your response should be an implementation of the function priority_v# (where # is the current iteration number); do not include any examples or extraneous functions.
4. You may use numpy and itertools.
The code you generate will be appended to the user prompt and run as a python program."""
### END SYSTEM PROMPT
### <<<< user prompt: will be added to the final prompt>>>>>
"""<<<E.g. Finds EXAMPLE_PROBLEMS. A brief description of the problem you're solving might be a 
good idea to include here, as it will go in to the prompt>>>
"""
import itertools ####<<<IF NECESSARY. DO NOT IMPORT ANY POTENTIALLY DANGEROUS PACKAGES HERE>>>>. 
import numpy as np
import funsearch

@funsearch.run #####<<<< THIS IS THE ENTRY POINT for funsearchy, and returns the score!>>>######
def evaluate(n: int) -> int:
    """Returns the size of an `n`-dimensional EXAMPLE_PROBLEM"""
    import pickle #### <<<<IF NECESSARY. IMPORT ANY POTENTIALLY DANGEROUS IMPORTS HERE, SO THAT THE PRIORITY FUNCTION HAS NO ACCESS>>>>>
    example_output = solve(n)
    return len(example_output)#### <<<<<THIS OUTPUT BECOMES THE SCORE>>>>>>

#### If you define additonal functions, ensure they have a docstring! This is to make the parser happy. 
#### Again, you will need to import any necessary packages inside their namespace.

def solve(n: int) -> np.ndarray:
    import pickle #### <<<<IF NECESSARY. IMPORT ANY POTENTIALLY DANGEROUS IMPORTS HERE, SO THAT THE PRIORITY FUNCTION HAS NO ACCESS>>>>>
    """Returns an EXAMPLE in `n` dimensions."""
    ####<<<AT SOME POINT YOU SHOULD CALL "priority(<whatever type you've specified below>)">>>
    pass  # TODO: Implement the solve function

@funsearch.evolve ####<<<< THIS TELLS FUNSEARCH WHAT TO EVOLVE>>>######
def priority(el: tuple[int, ...], n: int) -> float: ### <<<MODIFY THE TYPE SIGNATURE IF NECESSARY. 
    #### THE EVO algorithm doesn't care what the type sig is, as it simply calls evaluate() to get the score>>>
    #### HOWEVER, this function should still be called 'priority'
    """Returns the priority with which we want to add `element` to the EXAMPLE_PROBLEM, as a floating point number. <<BRIEF DESCRIPTION OF THE SPECIFIC PRIORITY PROBLEM IS A GOOD IDEA HERE>>>
    el is a tuple of length n with values 0-2.
    """
    pass  # TODO: Implement the priority function

##### SYSTEM PROMPT - modify in CONFIG.PY, or at the top of the spec file
##### THE USER PROMPT WILL BE FORMED AS FOLLOWS:
# === PROMPT ===
# """Finds large cap sets (sets of n-dimensional vectors over F_3 that do not contain 3 points on a line)."""
#
# import itertools
# import numpy as np
# import funsearch
#
# @funsearch.run
# def priority_v0(v: tuple[int, ...], n: int) -> float:
#   """Returns the priority, as a floating point number, of the vector `v` of length `n`. The vector 'v' is a tuple of values in {0,1,2}.
#       The cap set will be constructed by adding vectors that do not create a line in order by priority.
#   """
#   return 0
#
# def priority_v1(v: tuple[int, ...], n: int) -> float:
#   """Improved version of `priority_v0`.
#   """
#   return 1
#
# def priority_v2(v: tuple[int, ...], n: int) -> float:
#   """Improved version of `priority_v1`.
#   """
#   return 2
# === END PROMPT ===



