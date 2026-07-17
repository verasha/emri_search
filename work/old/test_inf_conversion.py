import cupy as cp
import numpy as np

# Test different infinite values on GPU and their conversion to CPU
print("=== Testing GPU to CPU conversion of infinite values ===")

# Create different infinite values on GPU
gpu_pos_inf = cp.array(np.inf)
gpu_neg_inf = cp.array(-np.inf)
gpu_nan = cp.array(np.nan)
gpu_large_pos = cp.array(1e308)
gpu_large_neg = cp.array(-1e308)

print(f"GPU positive infinity: {gpu_pos_inf}")
print(f"GPU negative infinity: {gpu_neg_inf}")
print(f"GPU NaN: {gpu_nan}")
print(f"GPU large positive: {gpu_large_pos}")
print(f"GPU large negative: {gpu_large_neg}")

print("\n=== Converting to CPU with .get() ===")
cpu_pos_inf = gpu_pos_inf.get()
cpu_neg_inf = gpu_neg_inf.get()
cpu_nan = gpu_nan.get()
cpu_large_pos = gpu_large_pos.get()
cpu_large_neg = gpu_large_neg.get()

print(f"CPU positive infinity: {cpu_pos_inf}, type: {type(cpu_pos_inf)}")
print(f"CPU negative infinity: {cpu_neg_inf}, type: {type(cpu_neg_inf)}")
print(f"CPU NaN: {cpu_nan}, type: {type(cpu_nan)}")
print(f"CPU large positive: {cpu_large_pos}, type: {type(cpu_large_pos)}")
print(f"CPU large negative: {cpu_large_neg}, type: {type(cpu_large_neg)}")

print("\n=== Converting to Python float ===")
try:
    py_pos_inf = float(cpu_pos_inf)
    print(f"Python positive infinity: {py_pos_inf}, type: {type(py_pos_inf)}")
except Exception as e:
    print(f"Error converting pos_inf to float: {e}")

try:
    py_neg_inf = float(cpu_neg_inf)
    print(f"Python negative infinity: {py_neg_inf}, type: {type(py_neg_inf)}")
except Exception as e:
    print(f"Error converting neg_inf to float: {e}")

try:
    py_nan = float(cpu_nan)
    print(f"Python NaN: {py_nan}, type: {type(py_nan)}")
except Exception as e:
    print(f"Error converting nan to float: {e}")

print("\n=== Testing finite checks ===")
print(f"np.isfinite(py_pos_inf): {np.isfinite(py_pos_inf)}")
print(f"np.isfinite(py_neg_inf): {np.isfinite(py_neg_inf)}")
print(f"np.isfinite(py_nan): {np.isfinite(py_nan)}")
print(f"np.isinf(py_pos_inf): {np.isinf(py_pos_inf)}")
print(f"np.isinf(py_neg_inf): {np.isinf(py_neg_inf)}")
print(f"np.isnan(py_nan): {np.isnan(py_nan)}")

print("\n=== Testing dynesty-style array creation ===")
# Simulate what dynesty does
test_values = [py_pos_inf, py_neg_inf, py_nan, 1.5, -2.3]
try:
    np_array = np.array(test_values)
    print(f"NumPy array creation successful: {np_array}")
except Exception as e:
    print(f"Error creating NumPy array: {e}")

print("\n=== Test what happens when we try implicit conversion ===")
# This is what causes the original error
try:
    # This simulates what dynesty tries to do internally
    implicit_array = np.array([gpu_pos_inf])  # CuPy array in NumPy array
    print(f"Implicit conversion successful: {implicit_array}")
except Exception as e:
    print(f"Implicit conversion error: {e}")
    print(f"Error type: {type(e)}")