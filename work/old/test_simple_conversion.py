import numpy as np

# Test the core conversion issue without CuPy
print("=== Testing infinite value handling ===")

# Create different infinite values
pos_inf = np.inf
neg_inf = -np.inf
nan_val = np.nan
large_pos = 1e308
large_neg = -1e308

print(f"Positive infinity: {pos_inf}")
print(f"Negative infinity: {neg_inf}")
print(f"NaN: {nan_val}")
print(f"Large positive: {large_pos}")
print(f"Large negative: {large_neg}")

print("\n=== Converting to Python float ===")
py_pos_inf = float(pos_inf)
py_neg_inf = float(neg_inf)
py_nan = float(nan_val)

print(f"Python positive infinity: {py_pos_inf}, type: {type(py_pos_inf)}")
print(f"Python negative infinity: {py_neg_inf}, type: {type(py_neg_inf)}")
print(f"Python NaN: {py_nan}, type: {type(py_nan)}")

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

print("\n=== Testing clamping strategy ===")
def clamp_for_dynesty(value):
    if np.isposinf(value):
        return 700.0
    elif np.isneginf(value):
        return -700.0
    elif not np.isfinite(value):
        return -700.0
    else:
        return value

clamped_values = [clamp_for_dynesty(v) for v in test_values]
print(f"Original values: {test_values}")
print(f"Clamped values: {clamped_values}")

clamped_array = np.array(clamped_values)
print(f"Clamped array: {clamped_array}")

print("\n=== Testing extreme values ===")
extreme_values = [1e100, -1e100, 1e200, -1e200, 1e308, -1e308]
for val in extreme_values:
    clamped = clamp_for_dynesty(val)
    print(f"Original: {val:e}, Clamped: {clamped}, Finite: {np.isfinite(val)}")