# NumPy Core Learning Notes
![Repository Views](https://komarev.com/ghpvc/?username=jashinspires&repo=Numpyknowledge&label=Views&color=007ec6&style=flat-square)

> Personal revision tracker of my NumPy learning journey — concepts I've written code for, understood, and found worth noting. New sections get appended as I go deeper.

---

## Table of Contents

1. [Introduction to NumPy](#1-introduction-to-numpy)
2. [NumPy Arrays vs Python Sequences](#2-numpy-arrays-vs-python-sequences)
3. [Creating Arrays & Structural Generators](#3-creating-arrays--structural-generators)
4. [Array Attributes](#4-array-attributes)
5. [Changing Data Type (`astype`)](#5-changing-data-type-astype)
6. [Array Operations & Vectorization](#6-array-operations--vectorization)
7. [Array Functions (Aggregations & Axis Mechanics)](#7-array-functions-aggregations--axis-mechanics)
8. [Dot Product & Matrix Multiplication](#8-dot-product--matrix-multiplication)
9. [Indexing and Slicing (1D, 2D, 3D)](#9-indexing-and-slicing-1d-2d-3d)
10. [Iterating Over Arrays (`np.nditer`)](#10-iterating-over-arrays-npnditer)
11. [Reshaping & Layout Utilities](#11-reshaping--layout-utilities)
12. [Stacking and Splitting](#12-stacking-and-splitting)
13. [Quick Tricks & Gotchas](#13-quick-tricks--gotchas)
14. [Why NumPy Over Python Lists (Deep Dive)](#14-why-numpy-over-python-lists-deep-dive)
15. [Advanced Indexing](#15-advanced-indexing)
16. [Broadcasting](#16-broadcasting)
17. [Broadcasting Rules](#17-broadcasting-rules)
18. [Mental Models & Gotchas (Advanced)](#18-mental-models--gotchas-advanced)
19. [Mathematical Formulas with NumPy](#19-mathematical-formulas-with-numpy)
20. [Working with Missing Values](#20-working-with-missing-values)
21. [Plotting with Matplotlib](#21-plotting-with-matplotlib)
22. [Roadmap](#22-roadmap)

---

## 1. Introduction to NumPy

**NumPy** (Numerical Python) is the foundational package for scientific computing in Python.

- Provides a **multidimensional array object** (`ndarray`)
- Provides **derived objects** — masked arrays, matrices
- Provides fast routines for mathematical, logical, and shape-manipulation operations:
  - Sorting, selecting, I/O
  - Discrete Fourier Transforms
  - Basic linear algebra and statistics
  - Random simulation

> **Core object:** `ndarray` — an n-dimensional array of homogeneous data.

```python
import numpy as np
```

---

## 2. NumPy Arrays vs Python Sequences

| Feature | NumPy Array | Python List |
|---|---|---|
| **Size** | Fixed at creation. Changing size creates a new array in memory. | Dynamic — resizes freely. |
| **Data Type** | Homogeneous — every element must share the same dtype. | Heterogeneous — can mix ints, strings, objects. |
| **Memory Layout** | Contiguous block allocation. | Array of scattered pointers. |
| **Speed** | Vectorized — executes in compiled C. | Slow element-by-element Python loops. |
| **Math Ops** | Element-wise by default. | Requires explicit loops or comprehensions. |

> **Under-the-hood trap:** Because `ndarray` has a fixed memory footprint, appending inside a loop causes NumPy to silently duplicate the entire array, fill it, and discard the old one. For large datasets, this completely tanks performance.

---

## 3. Creating Arrays & Structural Generators

### From Python Lists

```python
a = np.array([1, 3, 4, 5])                          # 1D — Vector
b = np.array([[1, 2, 3], [4, 5, 6]])                 # 2D — Matrix
c = np.array([[[1,2,3],[4,5,6]], [[3,45,8],[21,32,1]]])  # 3D — Tensor
```

### Specifying dtype at Creation

```python
k = np.array([1, 2, 3], dtype=float)   # → array([1., 2., 3.])
```

### Built-in Generators

| Function | Purpose | Notes |
|---|---|---|
| `np.arange(start, stop, step)` | Range-like, returns array | Excludes `stop`. Controls step size. |
| `np.ones(shape)` | All-ones array | Default dtype is `float64`. |
| `np.zeros(shape)` | All-zeros array | Ideal for pre-allocated buffers. |
| `np.random.random(shape)` | Uniform random floats `[0.0, 1.0)` | Fills the given shape. |
| `np.linspace(start, stop, num)` | Evenly spaced values | Includes both endpoints. Controls point count. |
| `np.identity(n)` | Square identity matrix | `n × n`, 1s on main diagonal. |

> **`linspace` edge case:** Forcing `dtype=int` truncates fractional steps rather than rounding them. The result is slightly asymmetric — e.g., `np.linspace(-10, 10, 10, dtype=int)` → `[-10, -8, -6, -4, -2, 1, 3, 5, 7, 10]`.

> **Quick trick:** Chain `arange` with `reshape` to build test matrices instantly:
> ```python
> np.arange(1, 13).reshape(3, 4)
> ```

---

## 4. Array Attributes

```python
arr1 = np.arange(8)
arr2 = np.arange(8, dtype=float).reshape(2, 4)
arr3 = np.arange(8).reshape(2, 2, 2)
```

| Attribute | Description | Example |
|---|---|---|
| `.ndim` | Number of dimensions | `arr3.ndim` → `3` |
| `.shape` | Tuple of axis sizes | `arr3.shape` → `(2, 2, 2)` |
| `.size` | Total element count | `arr3.size` → `8` |
| `.itemsize` | Bytes per element | `8` for `int64` / `float64` |
| `.dtype` | Internal numeric type | `int64` |
| `.nbytes` | Total bytes consumed by data | Use this, not `sys.getsizeof()` |

> **3D shape mental model for `(2, 2, 2)`:**
> - 1st value → number of 2D planes (blocks)
> - 2nd value → rows per plane
> - 3rd value → columns per plane

---

## 5. Changing Data Type (`astype`)

```python
lighter_arr = arr3.astype(np.int32)
```

Always returns a **new array** — the original is unchanged. Useful for reducing memory overhead, e.g., downcasting `int64` → `int32` or `float64` → `float32` on large datasets.

---

## 6. Array Operations & Vectorization

```python
a1 = np.arange(12).reshape(3, 4)
a2 = np.arange(12, 24).reshape(3, 4)
```

### Scalar Operations (Broadcasting Against a Single Value)

```python
a1 * 2     # multiply every element by 2
a1 + 10    # add 10 to every element
a1 ** 2    # square every element
```

### Relational / Boolean Operations

```python
a2 == 15   # returns a boolean array of identical shape
a1 > 5
```

Boolean arrays are the foundation of **masking and filtering** — covered in §15.

### Element-wise Array Operations

```python
a1 * a2    # element-wise product
a1 + a2
a1 / a2
```

> Arrays must have **compatible shapes** — or follow NumPy's broadcasting rules (§16–17).

---

## 7. Array Functions (Aggregations & Axis Mechanics)

### Min / Max / Sum / Product

```python
np.prod(k1)           # product of all elements → scalar
np.prod(k1, axis=0)   # column-wise product
np.prod(k2, axis=1)   # row-wise product
```

> **Axis rule of thumb (2D):**
> - `axis=0` → collapses rows → result is a 1D row (column-wise operation)
> - `axis=1` → collapses columns → result is a 1D column (row-wise operation)
> - No axis → flattens completely, returns a scalar

### Statistical Functions

```python
np.mean(k1, axis=1)   # row-wise mean
np.median(k1)         # median of the full array
np.std(k1)            # standard deviation
np.var(k1)            # variance
```

---

## 8. Dot Product & Matrix Multiplication

```python
d1 = np.arange(12).reshape(3, 4)
d2 = np.arange(13, 25).reshape(4, 3)
np.dot(d1, d2)   # → shape (3, 3)
```

> **Inner dimension rule:**
>
> $$(\mathbf{M \times N}) \cdot (\mathbf{N \times P}) \rightarrow (\mathbf{M \times P})$$
>
> The inner dimensions must match exactly. Everything else is a `ValueError`.

---

## 9. Indexing and Slicing (1D, 2D, 3D)

```python
I1 = np.arange(12)
I2 = np.arange(12).reshape(3, 4)
I3 = np.arange(27).reshape(3, 3, 3)
```

### 1D

$$\text{[start : stop : step]}$$

```python
I1[2:5:2]    # elements at index 2, 4
```

### 2D — `[rows, cols]`

Always use comma-separated indices in a single bracket — not nested brackets — to avoid creating intermediate Python objects.

```python
I2[0, :2]      # row 0, first 2 columns
I2[:, 2]       # all rows, column 2
I2[1:3, 1:3]   # sub-matrix
I2[::2, ::3]   # every 2nd row, every 3rd column
I2[::2, 1::2]  # alternating row/col pattern
I2[1, ::3]     # row 1, every 3rd element
```

### 3D — `[block, row, col]`

```python
I3[1, 1, 0]        # single element
I3[1, :, :]        # entire second block
I3[0, 1, :]        # row 1 of block 0
I3[1, :, 1]        # column 1 of block 1
I3[2, 1:, 1:]      # bottom-right sub-matrix of block 2
I3[0::2, 0, 0::2]  # multi-axis skip slice
```

> **Views vs. copies:** NumPy slices return a **view**, not a copy. Modifying a slice modifies the original array. Use `.copy()` to avoid accidental mutations.

---

## 10. Iterating Over Arrays (`np.nditer`)

```python
for i in np.nditer(I3):
    print(i)
```

`np.nditer()` iterates over every individual element regardless of dimensionality. A plain `for i in arr` loop only iterates the first axis.

---

## 11. Reshaping & Layout Utilities

### Transpose

```python
np.transpose(I2)   # equivalent to I2.T
```

Flips row and column coordinates.

### Ravel (Flatten to 1D)

```python
I3.ravel()     # returns a flat 1D view when possible
I3.flatten()   # always returns a full copy
```

> `ravel()` is faster and memory-efficient (returns a view). `flatten()` allocates new memory — use it when you specifically need an independent copy.

---

## 12. Stacking and Splitting

### Stacking

```python
np.hstack((I2, I2))   # horizontal — appends columns
np.vstack((I2, I2))   # vertical — appends rows
```

### Splitting

```python
np.hsplit(I2, 4)   # 4 vertical slices (by columns)
np.vsplit(I2, 3)   # 3 horizontal slices (by rows)
```

> The split parameter must divide the target axis evenly, otherwise NumPy raises a `ValueError`.

---

## 13. Quick Tricks & Gotchas

| # | Tip / Trap | Why It Matters |
|---|---|---|
| 1 | `np.arange(...).reshape(...)` | Fastest way to build test matrices. |
| 2 | Axis orientation | `axis=0` → vertical (column-wise); `axis=1` → horizontal (row-wise). |
| 3 | Slices are views | Modifying a slice mutates the source array. Use `.copy()`. |
| 4 | `astype()` for memory | Downcast `int64→int32` or `float64→float32` to cut memory in half. |
| 5 | `linspace` for plotting | Gives clean, even x-axis values — `arange` can miss the endpoint. |
| 6 | `arr.T` over `np.transpose(arr)` | Cleaner syntax, same result. |
| 7 | Boolean masks | `arr == val` gives a filter mask. Foundation of all conditional selection. |
| 8 | `np.nditer` | Handles multi-dimensional iteration without nested loops. |
| 9 | Vectorize first | Always prefer vectorized expressions over Python loops. Order-of-magnitude speed difference. |
| 10 | `np.identity` vs `np.eye` | `identity(n)` → square only; `eye(n, m)` → rectangular layouts. |

---

## 14. Why NumPy Over Python Lists (Deep Dive)

This is a **speed** comparison, not a time-complexity one. Both are O(n) for element-wise ops — but the constant factor is brutal.

### Core Difference

| Python List | NumPy Array |
|---|---|
| **Referential** — stores pointers to scattered objects | **Contiguous** — raw values packed into a single memory block |
| Dynamic size, mixed types | Fixed size, fixed dtype |
| Loop runs in Python (slow) | Loop runs in C (fast, SIMD-vectorized) |
| Large per-element overhead | Compact, cache-friendly layout |

### Benchmark

```python
# List approach
a = [i for i in range(10_000_000)]
b = [i for i in range(10_000_000, 20_000_000)]

import time
start = time.time()
c = [a[i] + b[i] for i in range(len(a))]
print(time.time() - start)   # seconds

# NumPy approach — typically 50–100x faster
a1 = np.arange(10_000_000)
b1 = np.arange(10_000_000, 20_000_000)

start = time.time()
c1 = a1 + b1   # no Python loop, pure C
print(time.time() - start)
```

### Memory Check

```python
import sys
sys.getsizeof([i for i in range(10_000_000)])   # large
sys.getsizeof(np.arange(10_000_000))            # much smaller
```

> **Why NumPy is fast:** Not because the algorithm is better — both are O(n). It's the data layout (contiguous + typed) that enables CPU-level optimizations: SIMD instructions, cache locality, no pointer chasing.

> **`sys.getsizeof()` lies on NumPy arrays** — it reports Python object overhead, not actual data. Use `arr.nbytes` for the real number.

---

## 15. Advanced Indexing

### 15.1 Fancy Indexing

Pass a list or array of indices to grab arbitrary rows or columns.

```python
a = np.arange(12).reshape(4, 3)

a[:, [0, 2]]    # all rows, only columns 0 and 2
a[[0, 2], :]    # only rows 0 and 2
```

> Fancy indexing always returns a **copy**, never a view. Contrast with regular slicing (`a[:, 0:2]`), which gives a view.

### 15.2 Boolean Indexing & Masking

```python
a = np.random.randint(1, 100, 24).reshape(6, 4)

mask = a > 50     # boolean array, same shape as a
a[mask]           # 1D array of matching elements
```

### 15.3 Combining Conditions

Use `&`, `|`, `~` — **not** Python's `and`, `or`, `not`. Each condition must be wrapped in parentheses due to operator precedence.

```python
a[(a > 50) & (a % 2 == 0)]    # > 50 AND even
a[(a > 50) | (a < 10)]        # > 50 OR < 10
a[~(a > 50)]                  # NOT > 50
```

> **Common mistake:** `a > 50 and a % 2 == 0` raises `ValueError: truth value of array is ambiguous`. Python's `and` expects scalars. Always use `&` for element-wise logic.

> **Parentheses matter:** `&` binds tighter than `>`, so `a > 50 & a % 2 == 0` parses incorrectly. Always wrap each condition.

---

## 16. Broadcasting

Broadcasting is the mechanism NumPy uses to perform arithmetic on arrays of **different shapes** without explicit loops or copies. The smaller array is *virtually* stretched to match the larger one — no memory is actually allocated for the stretched version.

```python
a = np.arange(6).reshape(2, 3)
b = np.arange(6, 12).reshape(2, 3)
print(a + b)    # same shape — straightforward
```

The real power is when shapes differ:

```python
a = np.arange(6).reshape(2, 3)   # shape (2, 3)
b = np.array([10, 20, 30])       # shape (3,)
a + b                            # b is broadcast across each row of a
```

---

## 17. Broadcasting Rules

Two rules, applied in order:

### Rule 1 — Align Dimensions

If arrays have different numbers of dimensions, **prepend 1s** to the shape of the smaller one until both have equal rank.

```
A: (    3, 4)
B: (2, 3, 4)
→ A becomes (1, 3, 4)
```

### Rule 2 — Stretch Size-1 Dimensions

For each dimension, sizes must either be equal or one of them must be `1` (it stretches to match). If neither holds → `ValueError`.

```
(4, 3)  +  (3,)     ✅  →  (3,) becomes (1, 3) → stretches to (4, 3)
(4, 3)  +  (4, 1)   ✅  →  stretches column-wise
(4, 3)  +  (4,)     ❌  →  trailing dims (3 vs 4) mismatch
```

### Quick Mental Check

Align shapes **from the right**, then verify each pair is equal or has a `1`:

```
       (2, 3, 4)
          (3, 4)   →  prepend 1 → (1, 3, 4) ✅
```

> **`np.newaxis` trick:** Insert a size-1 dimension to force broadcasting in a specific direction.
>
> ```python
> col = np.array([1, 2, 3])          # shape (3,)
> col[:, np.newaxis]                 # shape (3, 1) — now a column vector
> a + col[:, np.newaxis]             # broadcasts down the rows
> ```

> **Memory trap:** Broadcasting can silently allocate huge intermediate arrays. `(10000, 1) + (1, 10000)` produces a `10000 × 10000` result. Watch memory usage.

---

## 18. Mental Models & Gotchas (Advanced)

| Topic | Remember This |
|---|---|
| List vs Array speed | It's the memory layout, not the algorithm |
| Fancy indexing | Returns a **copy** |
| Slicing | Returns a **view** — modifying it modifies the original |
| Boolean masks | Must match the shape of the indexed array |
| `and` / `or` | ❌ on arrays — use `&` / `|` with parentheses |
| Broadcasting | Right-aligned shape comparison, then check each axis |
| `np.newaxis` | Primary tool for shape engineering |
| Memory | Use `arr.nbytes`, not `sys.getsizeof()` |

**Notes to future me:**
- When stuck, print `.shape` — 90% of NumPy bugs are shape mismatches.
- Vectorize first, optimize later. Always measure before assuming.
- The mental model that unlocks NumPy: **think in arrays, not in loops.**

---

## 19. Mathematical Formulas with NumPy

NumPy makes implementing ML math clean and readable — no nested loops, just vectorized expressions.

### Element-wise Trigonometry

```python
a = np.arange(10)
np.sin(a)
```

### Sigmoid Function

```python
def sigmoid(array):
    return 1 / (1 + np.exp(-array))

print(sigmoid(np.arange(10)))
```

The key here is `np.exp()`, which applies `e^x` element-wise. This is the direct vectorized equivalent of calling `math.exp()` in a loop — but orders of magnitude faster.

### Mean Squared Error (MSE)

```python
actual    = np.random.randint(1, 50, 25)
predicted = np.random.randint(1, 50, 25)

def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)

mse(actual, predicted)
```

The expression `(actual - predicted) ** 2` works because both arrays have the same shape — element-wise subtraction followed by element-wise squaring. `np.mean()` then reduces the result to a scalar.

### Binary Cross-Entropy (BCE)

```python
def bce(y_true, y_pred, epsilon=1e-15):
    # clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

`np.clip()` prevents numerical instability — log of 0 is undefined, so predictions are bounded away from 0 and 1.

---

## 20. Working with Missing Values

NumPy represents missing or undefined numerical data using `np.nan` (Not a Number), which is a special IEEE 754 float.

```python
a = np.array([1, 2, 3, 4, np.nan, 6])
a
# → array([ 1.,  2.,  3.,  4., nan,  6.])
```

### Detecting NaN

```python
np.isnan(a)
# → array([False, False, False, False,  True, False])
```

### Filtering Out NaN

```python
a[~np.isnan(a)]
# → array([1., 2., 3., 4., 6.])
```

The `~` operator inverts the boolean mask — `~np.isnan(a)` gives `True` wherever the value is *not* NaN, so indexing with it drops all missing values.

> **dtype note:** Arrays containing `np.nan` are automatically cast to `float64`, since NaN is a float concept. Integer arrays can't hold NaN natively.

> **Aggregation gotcha:** Most NumPy aggregation functions (`np.mean`, `np.sum`, etc.) propagate NaN — if any element is NaN, the result is NaN. Use `np.nanmean`, `np.nansum`, etc. to ignore them:
>
> ```python
> np.mean(a)      # → nan
> np.nanmean(a)   # → 3.2  (ignores NaN)
> ```

---

## 21. Plotting with Matplotlib

`np.linspace` is the standard way to generate smooth, continuous x-axis values for plotting. It guarantees a fixed number of evenly spaced points between two bounds (both inclusive) — which `arange` can't reliably do with floating-point steps.

```python
import matplotlib.pyplot as plt
```

### Linear — y = x

```python
x = np.linspace(-10, 10, 100)
y = x
plt.plot(x, y)
```

### Quadratic — y = x²

```python
x = np.linspace(-10, 10, 100)
y = x ** 2
plt.plot(x, y)
```

### Sine Wave

```python
x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y)
```

### Logarithmic — y = x · ln(x)

```python
x = np.linspace(-10, 10, 100)
y = x * np.log(x)
plt.plot(x, y)
```

> Note: this will produce NaN for x ≤ 0 since `log` is undefined there. NumPy will compute it anyway and matplotlib will simply skip those points. Use `x = np.linspace(0.01, 10, 100)` for a clean plot.

### Sigmoid

```python
x = np.linspace(-10, 10, 100)
y = 1 / (1 + np.exp(-x))
plt.plot(x, y)
```

The sigmoid naturally squishes any real-valued input into `(0, 1)` — visible here as an S-curve. This is exactly what's used as a neuron activation in logistic regression and early neural networks.

---

## 22. Roadmap

- [x] Broadcasting Rules (Deep Dive)
- [x] Fancy Indexing & Boolean Mask Filtering
- [x] Mathematical Formulas (`sigmoid`, `mse`, `bce`)
- [x] Working with Missing Values (`np.nan`, `np.isnan`, `np.nanmean`)
- [x] Plotting with Matplotlib (`linspace` + common functions)
- [ ] Conditional Selection (`np.where`, `np.argmax`, `np.argmin`, `np.argsort`)
- [ ] Set Operations (`np.unique`, `np.intersect1d`, etc.)
- [ ] Structural Axis Manipulation (`expand_dims`, `squeeze`, `newaxis`)
- [ ] Mathematical & Trigonometric Engines
- [ ] Linear Algebra Submodule (`np.linalg`)
- [ ] Random Distributions & Seeds (`np.random`)
- [ ] Persistent Storage I/O (`np.save`, `np.load`, `np.loadtxt`)
- [ ] Memory Benchmarking: C-contiguous vs. F-contiguous views
- [ ] Coordinate Meshgrids (`np.meshgrid`)

---

*Last updated: after completing broadcasting, advanced indexing, mathematical formulas, missing values, and matplotlib plotting.* 🚀
