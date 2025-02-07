import pytest
from funsearch.test_security_2 import is_function_safe

def test_security_validation():
    # Test safe function
    safe_code = """
def priority_v2(v: tuple[int, ...], n: int) -> float:
    unique_values = np.unique(v)
    counts = [list(v).count(val) for val in unique_values]
    if 1 in counts:
        priority = np.mean([1/c for c in counts if c != 1])
    else:
        priority = 1.0
    return priority
"""
    assert is_function_safe(safe_code)

    # Test unsafe function
    unsafe_code = """
def priority_v2(v: tuple[int, ...], n: int) -> float:
    exec("print('hello')")
    return 1.0
"""
    assert not is_function_safe(unsafe_code) 