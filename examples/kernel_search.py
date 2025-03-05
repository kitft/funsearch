### SYSTEM PROMPT
"""You are a state-of-the-art python code completion system that will be used as part of a genetic algorithm to optimize matrix multiplication algorithms.
You will be given a list of functions, and you should improve the incomplete last function in the list.
1. Make only small changes but be sure to make some change.
2. Your response should be an implementation of the function matmul_v# (where # is the current iteration number); do not include any examples or extraneous functions.
3. You may use itertools, but not numpy or other external libraries.
4. Think step by step to outline an algorithm, and then implement it.
The code you generate will be appended to the user prompt and run as a python program."""
### END SYSTEM PROMPT
### user prompt
"""Finds efficient algorithms for multiplying two 3x3 matrices.
The goal is to MINIMIZE the number of scalar multiplications while maintaining correctness. `matmul` is the matmul function.
You should only use lists and loops. You should try to use AS FEW SCALAR MULTIPLICATIONS AS POSSIBLE, using strategies like the Strassen algorithm. Be innovative.
The matrices are always 3x3 and 3x3, and are unpacked into variables first in the same way, by every program. A and B should *not* be accessed after they are unpacked!
The naive implementation of matrix multiplication uses 27 multiplications. We want to reduce this number.
"""
import itertools
import funsearch

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
    A_matrices = rng.integers(-100, 101, size=(test_cases, 3, 3)).tolist()
    B_matrices = rng.integers(-100, 101, size=(test_cases, 3, 3)).tolist()
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
    #start_time = time.time()
    all_correct = True
    for A, B in test_matrices:
        result = matmul(A, B)
        expected = naive_multiply(A, B)
        if result != expected:
            all_correct = False
            break
    
    #execution_time = (time.time() - start_time)/test_cases
    result, score = count_multiplications_recursive(matmul,A_matrices[0],B_matrices[0])
    
    # Return score (higher is better)
    if not all_correct:
        raise Exception("Incorrect implementation")
        #return -10000  # Incorrect implementation gets a large negative penalty
    
    return -score 


# For more complex functions with inner functions
def count_multiplications_recursive(func, *args, **kwargs):
    """Count multiplications in a function and all its inner functions.
    We also disallow loops, comprehensions, while loops, conditional jumps, generator expressions, and calls to functions that could implement loops.
    This is a bit of a hack, but it was a good enough test."""
    import dis, types
    total_mults = 0

    
    # Count in the main function
    bytecode = dis.Bytecode(func)
    for instr in bytecode:
        # Check for various loop constructs in bytecode
        if instr.opname in ['FOR_ITER', 'SETUP_LOOP', 'GET_ITER', 'GET_ANEXT', 'SETUP_ASYNC_WITH']:
            raise Exception(f"Loop detected in the implementation {instr.opname}")
            
        # Check for list/set/dict comprehensions
        if instr.opname in ['LIST_APPEND', 'SET_ADD', 'MAP_ADD']:
            raise Exception(f"Comprehension detected in the implementation {instr.opname}")
            
        # Check for while loops and conditional jumps
        if instr.opname in ['JUMP_ABSOLUTE', 'POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE', 
                           'JUMP_IF_FALSE_OR_POP', 'JUMP_IF_TRUE_OR_POP']:
            raise Exception(f"Potential while loop or conditional jump detected in the implementation {instr.opname}")
            
        # Check for generator expressions
        if instr.opname in ['YIELD_VALUE', 'YIELD_FROM']:
            raise Exception(f"Generator expression detected in the implementation {instr.opname}    ")

        # Check for calls to functions that could implement loops
        if instr.opname in ['CALL_FUNCTION', 'CALL_METHOD', 'CALL_FUNCTION_KW', 'CALL_FUNCTION_EX']:
            func_name = None
            if hasattr(instr, 'argval') and isinstance(instr.argval, str):
                func_name = instr.argval
            elif hasattr(func, '__code__') and hasattr(func.__code__, 'co_names'):
                # Try to get the name from the previous instruction
                idx = list(bytecode).index(instr)
                if idx > 0:
                    prev = list(bytecode)[idx-1]
                    if hasattr(prev, 'argval') and isinstance(prev.argval, str):
                        func_name = prev.argval
            
            loop_funcs = ['map', 'filter', 'reduce', 'sum', 'any', 'all', 'enumerate', 'zip', 
                          'sorted', 'min', 'max', 'reversed', 'join', 'split', 'replace']
            if func_name and any(loop_func == func_name for loop_func in loop_funcs):
                raise Exception(f"Hidden loop construct detected: {func_name}")
                
        # Check for string operations that might contain implicit loops
        if instr.opname == 'LOAD_ATTR':
            string_loop_methods = ['join', 'split', 'replace', 'strip', 'find', 'index', 'count']
            if instr.argval in string_loop_methods:
                raise Exception(f"String method with implicit loop detected: {instr.argval}")
                
        # Check for recursive calls (potential tail recursion)
        if instr.opname == 'LOAD_GLOBAL' and instr.argval == func.__name__:
            next_instrs = list(bytecode)[list(bytecode).index(instr):]
            for next_instr in next_instrs[:5]:  # Check next few instructions
                if next_instr.opname in ['CALL_FUNCTION', 'CALL_METHOD']:
                    raise Exception("Recursive call detected (implicit loop)")
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

@funsearch.evolve
def matmul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Multiplies a 3x3 matrix A by a 3x3 matrix B efficiently, using as few scalar multiplications as possible.
    Returns the resulting 3x3 matrix C. It only use lists, loops, and basic arithmetic operations, and do not use any external libraries.
    It first unpacks the matrices into variables, and then intitialises the result matrix. This part of the program is always the same, and A and B should not be accessed past this point.
    Minimising the number of scalar multiplications is the ultimate goal. The naive implementation, below, uses 27 multiplications.
    """
    # Unpack the matrices into variables - DO NOT MODIFY THESE LINES
    a11,a12,a13 = A[0]
    a21,a22,a23 = A[1]
    a31,a32,a33 = A[2]
    b11,b12,b13 = B[0]
    b21,b22,b23 = B[1]
    b31,b32,b33 = B[2]
    # Initialize result matrix - DO NOT MODIFY THIS LINE
    C = [[0,0,0],[0,0,0],[0,0,0]]

    #ONLY THE CODE BELOW SHOULD BE MODIFIED. A and B are NOT accessed past this point.
    # Compute each element directly using unpacked variables
    C[0][0] = a11*b11 + a12*b21 + a13*b31 
    C[0][1] = a11*b12 + a12*b22 + a13*b32 
    C[0][2] = a11*b13 + a12*b23 + a13*b33 

    C[1][0] = a21*b11 + a22*b21 + a23*b31 
    C[1][1] = a21*b12 + a22*b22 + a23*b32 
    C[1][2] = a21*b13 + a22*b23 + a23*b33 

    
    C[2][0] = a31*b11 + a32*b21 + a33*b31 
    C[2][1] = a31*b12 + a32*b22 + a33*b32 
    C[2][2] = a31*b13 + a32*b23 + a33*b33 
            
    return C