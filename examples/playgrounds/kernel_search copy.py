### SYSTEM PROMPT
"""You are a state-of-the-art python code completion system that will be used as part of a genetic algorithm to optimize matrix multiplication algorithms.
You will be given a list of functions, and you should improve the incomplete last function in the list.
1. Make only small changes but be sure to make some change.
2. Try to keep the code short and any comments concise.
3. Your response should be an implementation of the function priority_v# (where # is the current iteration number); do not include any examples or extraneous functions.
4. You may use itertools, but not numpy or other external libraries.
The code you generate will be appended to the user prompt and run as a python program."""
### END SYSTEM PROMPT

### user prompt
"""Finds efficient algorithms for multiplying a 4x5 integer matrix by a 5x5 integer matrix.
The goal is to MINIMIZE the number of scalar multiplications while maintaining correctness. `priority` is the matmul function.
You should only use lists and loops. You should try to use AS FEW SCALAR MULTIPLICATIONS AS POSSIBLE, using strategies like the Strassen algorithm. Be innovative.
The matrices are always 4x5 and 5x5, and are unpacked into variables first in the same way, by every program. A and B should *not* be accessed after they are unpacked!
"""
import itertools

@funsearch.run
def evaluate(test_cases: int = 100) -> float:
    """Evaluates the performance of the matrix multiplication algorithm.
    Returns a score based on correctness and efficiency."""
    import random
    import time
    import types
    import functools
    
    # Generate test cases
    import numpy as np
    rng = np.random.default_rng()
    A_matrices = rng.integers(-100, 101, size=(test_cases, 4, 5)).tolist()
    B_matrices = rng.integers(-100, 101, size=(test_cases, 5, 5)).tolist()
    test_matrices = list(zip(A_matrices, B_matrices))
    
    # Verify correctness against naive implementation
    def naive_multiply(A, B):
        import numpy as np
        A_np = np.array(A)
        B_np = np.array(B)
        C_np = np.matmul(A_np, B_np)
        C = C_np.tolist()
        return C
    # Test the evolved algorithm
    start_time = time.time()
    all_correct = True
    for A, B in test_matrices:
        result = priority(A, B)
        expected = naive_multiply(A, B)
        if result != expected:
            all_correct = False
            break
    
    execution_time = (time.time() - start_time)/test_cases
    result, score = count_multiplications_recursive(priority,A_matrices[0],B_matrices[0])
    
    # Return score (higher is better)
    if not all_correct:
        return 0.0  # Incorrect implementation gets zero score
    
    # Score is inversely proportional to execution time
    return score 


# For more complex functions with inner functions
def count_multiplications_recursive(func, *args, **kwargs):
    """Count multiplications in a function and all its inner functions."""
    import dis, types
    total_mults = 0
    
    # Count in the main function
    bytecode = dis.Bytecode(func)
    for instr in bytecode:
        if instr.opname == 'BINARY_MULTIPLY' or (instr.opname == 'BINARY_OP' and instr.arg == 5):
            total_mults += 1
    
    # Find and count in inner functions
    for const in func.__code__.co_consts:
        if isinstance(const, types.CodeType):
            # This is an inner function
            inner_func = types.FunctionType(const, func.__globals__)
            inner_mults = count_multiplications_recursive(inner_func)
            total_mults += inner_mults
    
    # If this is the top-level call, execute the function
    if args or kwargs:
        result = func(*args, **kwargs)
        return result, total_mults
    
    return total_mults

def count_operations(A: list[list[int]], B: list[list[int]]) -> int:
    """Counts the number of arithmetic operations in the multiplication."""
    # This is a simplified model - in reality we'd instrument the code
    ops = 0
    for i in range(4):
        for j in range(5):
            for k in range(5):
                ops += 2  # One multiplication and one addition per inner loop
    return ops

@funsearch.evolve
def priority(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Multiplies a 4x5 matrix A by a 5x5 matrix B efficiently, using as few scalar multiplications as possible.
    Returns the resulting 4x5 matrix C. It only use lists, loops, and basic arithmetic operations, and do not use any external libraries.
    It first unpacks the matrices into variables, and then intitialises the result matrix. This part of the program is always the same, and A and B should not be accessed past this point.
    Minimising the number of scalar multiplications is the ultimate goal. The naive implementation, below, uses 100 multiplications.
    """
    # Unpack the matrices into variables
    a11,a12,a13,a14,a15 = A[0]
    a21,a22,a23,a24,a25 = A[1]
    a31,a32,a33,a34,a35 = A[2]
    a41,a42,a43,a44,a45 = A[3]
    b11,b12,b13,b14,b15 = B[0]
    b21,b22,b23,b24,b25 = B[1]
    b31,b32,b33,b34,b35 = B[2]
    b41,b42,b43,b44,b45 = B[3]
    b51,b52,b53,b54,b55 = B[4]  
    # Initialize result matrix
    C = [[0 for _ in range(5)] for _ in range(4)]

    #ONLY THIS BIT SHOULD BE MODIFIED. A and B are NOT accessed past this point.
    # Compute each element directly using unpacked variables
    C[0][0] = a11*b11 + a12*b21 + a13*b31 + a14*b41 + a15*b51
    C[0][1] = a11*b12 + a12*b22 + a13*b32 + a14*b42 + a15*b52
    C[0][2] = a11*b13 + a12*b23 + a13*b33 + a14*b43 + a15*b53
    C[0][3] = a11*b14 + a12*b24 + a13*b34 + a14*b44 + a15*b54
    C[0][4] = a11*b15 + a12*b25 + a13*b35 + a14*b45 + a15*b55
    
    C[1][0] = a21*b11 + a22*b21 + a23*b31 + a24*b41 + a25*b51
    C[1][1] = a21*b12 + a22*b22 + a23*b32 + a24*b42 + a25*b52
    C[1][2] = a21*b13 + a22*b23 + a23*b33 + a24*b43 + a25*b53
    C[1][3] = a21*b14 + a22*b24 + a23*b34 + a24*b44 + a25*b54
    C[1][4] = a21*b15 + a22*b25 + a23*b35 + a24*b45 + a25*b55
    
    C[2][0] = a31*b11 + a32*b21 + a33*b31 + a34*b41 + a35*b51
    C[2][1] = a31*b12 + a32*b22 + a33*b32 + a34*b42 + a35*b52
    C[2][2] = a31*b13 + a32*b23 + a33*b33 + a34*b43 + a35*b53
    C[2][3] = a31*b14 + a32*b24 + a33*b34 + a34*b44 + a35*b54
    C[2][4] = a31*b15 + a32*b25 + a33*b35 + a34*b45 + a35*b55
    
    C[3][0] = a41*b11 + a42*b21 + a43*b31 + a44*b41 + a45*b51
    C[3][1] = a41*b12 + a42*b22 + a43*b32 + a44*b42 + a45*b52
    C[3][2] = a41*b13 + a42*b23 + a43*b33 + a44*b43 + a45*b53
    C[3][3] = a41*b14 + a42*b24 + a43*b34 + a44*b44 + a45*b54
    C[3][4] = a41*b15 + a42*b25 + a43*b35 + a44*b45 + a45*b55
    
    return C



def count_multiplications(func, *args, **kwargs):
    """Count the number of multiplication operations in a function using bytecode analysis."""
    import dis, types
    # Counter for multiplication operations
    mult_count = 0
    
    # Get the bytecode
    bytecode = dis.Bytecode(func)
    # Print the bytecode for inspection
    #for instr in bytecode:
    #    print(f"{instr.offset} {instr.opname} {instr.arg if instr.arg is not None else ''}")
    
    # Count multiplication instructions (BINARY_MULTIPLY or BINARY_OP 5)
    for instr in bytecode:
        if instr.opname == 'BINARY_MULTIPLY' or (instr.opname == 'BINARY_OP' and instr.arg == 5):
            mult_count += 1
    
    # Execute the function
    result = func(*args, **kwargs)
    
    return result, mult_count

print(count_multiplications_recursive(priority, [[1,2,3,4,5] for _ in range(4)], [[7,8,9,10,11] for _ in range(5)]))
print(count_multiplications(priority, [[1,2,3,4,5] for _ in range(4)], [[7,8,9,10,11] for _ in range(5)]))
print(evaluate(10))