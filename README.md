# 📘 NumPy Core Learning Notes
![Repository Views](https://komarev.com/ghpvc/?username=jashinspires&repo=Numpyknowledge&label=Views&color=007ec6&style=flat-square)

> This README is a **personal revision tracker** of my NumPy learning journey. It covers concepts I've actually written code for and understood. New sections will be appended as I progress through advanced material.

---

## 📑 Table of Contents
1. [Introduction to NumPy](#1-introduction-to-numpy)
2. [NumPy Arrays vs Python Sequences](#2-numpy-arrays-vs-python-sequences)
3. [Creating Arrays & Structural Generators](#3-creating-arrays--structural-generators)
4. [Array Attributes (Memory & Shape Inspection)](#4-array-attributes-memory--shape-inspection)
5. [Changing Data Type (`astype`)](#5-changing-data-type-astype)
6. [Array Operations & Vectorization](#6-array-operations--vectorization)
7. [Array Functions (Aggregations & Axis Mechanics)](#7-array-functions-aggregations--axis-mechanics)
8. [Dot Product Matrix Multiplication](#8-dot-product-matrix-multiplication)
9. [Indexing and Slicing (1D, 2D, 3D)](#9-indexing-and-slicing-1d-2d-3d)
10. [Iterating Over Arrays (`np.nditer`)](#10-iterating-over-arrays-npnditer)
11. [Reshaping & Layout Utilities](#11-reshaping--layout-utilities)
12. [Stacking and Splitting](#12-stacking-and-splitting)
13. [⚡ Quick Tricks & Gotchas Matrix](#-quick-tricks--gotchas-matrix)
14. [🚀 Roadmap (Future Additions)](#-roadmap-future-additions)

---

## 1. Introduction to NumPy

**NumPy** (Numerical Python) is the fundamental package for scientific computing in Python.

- Provides a **multidimensional array object** (`ndarray`)
- Provides **derived objects** (masked arrays, matrices)
- Provides routines for **fast operations** on arrays:
  - Mathematical, logical, shape manipulation
  - Sorting, selecting, I/O
  - Discrete Fourier Transforms
  - Basic linear algebra & statistics
  - Random simulation

> 💡 **Core Object** → `ndarray` = an n-dimensional array of homogeneous data types.

```python
import numpy as np

```

---

## 2. NumPy Arrays vs Python Sequences

| Feature | NumPy Array | Python List |
| --- | --- | --- |
| **Size** | **Fixed at creation**. Modifying size creates a new array in memory. | Dynamic resizing. |
| **Data Type** | **Homogeneous** (every single element must be identical). | Heterogeneous (can mix strings, ints, objects). |
| **Memory** | Contiguous & compact block allocation. | Array of pointers scattered in memory. |
| **Speed** | Vectorized performance executing directly in compiled C. | Slow element-by-element Python loops. |
| **Math Ops** | Element-wise by default. | Requires explicit loops or list comprehensions. |

> ⚠️ **Critical Under-the-Hood Trap:** Because an `ndarray` has a fixed size in memory, trying to append items or change its size inside a loop causes NumPy to silently duplicate the array, fill it, and delete the old one. For large datasets, this completely destroys execution performance.

---

## 3. Creating Arrays & Structural Generators

### From Python Lists

```python
a = np.array([1, 3, 4, 5])            # 1D Array (Vector)
b = np.array([[1, 2, 3], [4, 5, 6]])  # 2D Array (Matrix)
c = np.array([[[1,2,3],[4,5,6]],
              [[3,45,8],[21,32,1]]])  # 3D Array (Tensor)

```

### Specifying dtype at Creation

```python
k = np.array([1, 2, 3], dtype=float)   # Returns: array([1., 2., 3.])

```

### Built-in Array Generators

| Function | Purpose / Signature | Key Mechanical Behavior |
| --- | --- | --- |
| `np.arange(start, stop, step)` | Like `range()` but returns array | Generates sequences up to (but excluding) `stop`. Controls **step size**. |
| `np.ones(shape)` | Array of all 1s | Allocates a matrix populated entirely by `1.0` (default float). |
| `np.zeros(shape)` | Array of all 0s | Allocates a matrix populated entirely by `0.0`. Ideal for pre-allocating memory matrices. |
| `np.random.random(shape)` | Random floats in `[0.0, 1.0)` | Fills a given shape configuration with uniform random values. |
| `np.linspace(start, stop, num)` | Linear split space | Generates exactly `num` **evenly spaced** elements between `start` and `stop` **inclusive**. Controls **point count**. |
| `np.identity(n)` | Identity Matrix | Generates a square $n \times n$ identity matrix filled with `1.0` across its main diagonal. |

> ✨ **Pro-Tip (Anomalous Integer linspace Behavior):** When forcing an integer data type onto `np.linspace` (e.g., `np.linspace(-10, 10, 10, dtype=int)`), NumPy truncates the calculated fractional steps down to their nearest integers. This can result in slightly asymmetric integer jump patterns (`[-10, -8, -6, -4, -2, 1, 3, 5, 7, 10]`) compared to standard floating-point output.

> 💡 **Trick:** Chain `.reshape()` with `arange` to instantly build structured test matrices:
> ```python
> np.arange(1, 13).reshape(3, 4)
> 
> ```
> 
> 

---

## 4. Array Attributes (Memory & Shape Inspection)

```python
arr1 = np.arange(8)
arr2 = np.arange(8, dtype=float).reshape(2, 4)
arr3 = np.arange(8).reshape(2, 2, 2)

```

| Attribute | Description | Example Output |
| --- | --- | --- |
| `.ndim` | Number of dimensions | `arr3.ndim` → `3` |
| `.shape` | Tuple of dimensions | `arr3.shape` → `(2, 2, 2)` |
| `.size` | Total number of individual elements | `arr3.size` → `8` (Product of all values in `.shape`) |
| `.itemsize` | **The exact memory footprint of a single element in bytes** | `8` (An array with `int64` or `float64` takes 8 bytes = 64 bits per item) |
| `.dtype` | Assignment of internal numeric storage type | `int64` |

> 💡 **Mental Model for 3D Array Shapes `(2, 2, 2)`:**
> * 1st Parameter (`2`) → Number of 2D matrices (blocks/planes)
> * 2nd Parameter (`2`) → Number of rows inside each matrix
> * 3rd Parameter (`2`) → Number of columns inside each matrix
> 
> 

---

## 5. Changing Data Type (`astype`)

```python
lighter_arr = arr3.astype(np.int32)

```

* **Crucial Note:** Returns a **brand-new array** with the converted dtype; it does not modify the original array in place.
* Highly useful to **save memory overhead** when handling large datasets (e.g., down-casting a huge `int64` dataset down to a lighter `int32` or `float64` to `float32`).

---

## 6. Array Operations & Vectorization

```python
a1 = np.arange(12).reshape(3, 4)
a2 = np.arange(12, 24).reshape(3, 4)

```

### Scalar Operations (Element-Wise Broadcasting)

Applying a mathematical operation or relational comparison against a scalar constant broadcasts that instruction across every item:

```python
a1 * 2      # Multiplies every element by 2
a1 + 10     # Adds 10 to every element
a1 ** 2     # Squares every element

```

### Relational (Boolean) Operations

```python
a2 == 15    # Returns a Boolean matrix of identical shape containing True/False
a1 > 5

```

> 💡 Boolean array outputs are foundational for **masking and filtering arrays** (advanced topic to be added).

### Vector Operations (Array-on-Array)

When combining two arrays of identical shapes, operations apply to corresponding element pairs sequentially:

```python
a1 * a2       # Element-wise product matrix multiplication
a1 + a2
a1 / a2

```

> ⚠️ Arrays must have **compatible shapes** (or strictly follow NumPy's broadcasting rules).

---

## 7. Array Functions (Aggregations & Axis Mechanics)

### Min / Max / Sum / Product

```python
np.prod(k1)              # Product of ALL elements combined
np.prod(k1, axis=0)      # Column-wise product
np.prod(k2, axis=1)      # Row-wise product

```

> 🔑 **Axis Rule of Thumb (2D):**
> * `axis=0` → Collapses the matrix **downward along the rows**. The operation runs **column-wise**, returning a 1D row array.
> * `axis=1` → Collapses the matrix **sideways along the columns**. The operation runs **row-wise**, returning a 1D column array.
> * *No Axis Passed* → Flattens the array completely and calculates a single scalar answer for the entire dataset.
> 
> 

### Statistical Functions

```python
np.mean(k1, axis=1)   # Row-wise mean
np.median(k1)         # Median of full array
np.std(k1)            # Standard deviation
np.var(k1)            # Variance

```

---

## 8. Dot Product Matrix Multiplication

```python
d1 = np.arange(12).reshape(3, 4)
d2 = np.arange(13, 25).reshape(4, 3)
np.dot(d1, d2)        # Resulting shape → (3, 3)

```

> ⚠️ **The Inner Dimension Dimension Rule:** For mathematical matrix dot products (`np.dot(A, B)`), the inner dimensions must match exactly:
> 
> $$\mathbf{(M \times N) \cdot (N \times P) \rightarrow (M \times P)}$$
> 
> 

---

## 9. Indexing and Slicing (1D, 2D, 3D)

```python
I1 = np.arange(12)
I2 = np.arange(12).reshape(3, 4)
I3 = np.arange(27).reshape(3, 3, 3)

```

### 1D Extraction

$$\text{[start : stop : step]}$$

```python
I1[2:5:2]      # Simple slicing with step hopping

```

### 2D Extraction — `[rows, cols]`

Always separate dimension parameters inside a single set of brackets using commas. Do not use nested brackets like standard Python (`matrix[x][y]`), as this forces slow intermediate object construction.

```python
I2[0, :2]       # Row 0, first 2 columns
I2[:, 2]        # All rows, column 2
I2[1:3, 1:3]    # Sub-matrix extraction patch
I2[::2, ::3]    # Advanced skipping slice: every 2nd row, every 3rd column
I2[::2, 1::2]   # Alternating fancy row/col patterns
I2[1, ::3]      # Row 1, every 3rd element

```

### 3D Extraction — `[matrix_block, row, col]`

```python
I3[1, 1, 0]        # Single element extraction
I3[1, :, :]        # Entire 2nd matrix block plane
I3[0, 1, :]        # Row 1 of matrix block 0
I3[1, :, 1]        # Column 1 of matrix block 1
I3[2, 1:, 1:]      # Bottom-right sub-matrix slice of matrix block 2
I3[0::2, 0, 0::2]  # Advanced multi-axis skipping slice

```

> 💡 **Mental Model:** For a 3D array, always think through the hierarchy coordinates: `[which_matrix_block, which_row, which_col]`.

> ⚠️ **Critical Slicing Gotcha (Views vs. Copies):** NumPy slicing returns a **view** of the original array, not a copy! If you modify elements in a sliced sub-array, **the changes will permanently alter your original master array**. Use `.copy()` explicitly to avoid accidental edits.

---

## 10. Iterating Over Arrays (`np.nditer`)

```python
for i in np.nditer(I3):
    print(i)

```

* `np.nditer()` is an optimized internal engine that iterates over **every single individual element** sequentially regardless of how many dimensions the array has.
* A regular standard Python loop (`for i in arr`) only loops across the **first axis** (axis 0).

---

## 11. Reshaping & Layout Utilities

### Transpose

```python
np.transpose(I2)   # Or simply use the property shortcut: I2.T

```

* Flips rows and column coordinates completely.

### Ravel (Flatten)

```python
I3.ravel()         # Converts any multi-dimensional array into a flat 1D stream

```

> 💡 **Mechanics Difference (`ravel` vs `flatten`):**
> * `ravel()` → Returns a **view** whenever possible (Incredibly fast, zero extra memory footprint).
> * `flatten()` → Always allocates new memory and returns a physical **copy** of the data.
> 
> 

---

## 12. Stacking and Splitting

### Stacking (Stitching Arrays)

Group individual arrays cleanly inside a tuple wrapper passed directly to the stackers:

```python
np.hstack((I2, I2))   # Horizontal Stack → Chains items side-by-side (appends columns)
np.vstack((I2, I2))   # Vertical Stack → Chains items on top of each other (appends rows)

```

### Splitting (Chopping Arrays)

```python
np.hsplit(I2, 4)      # Split into 4 vertical pieces (by columns)
np.vsplit(I2, 3)      # Split into 3 horizontal pieces (by rows)

```

> ⚠️ **Shape Constraint:** The integer split parameter specified **must evenly divide** into the target matrix axis dimension length or NumPy will raise a `ValueError`.

---

## ⚡ Quick Tricks & Gotchas Matrix

| # | Optimization Tip / Trap | Key Mechanical Reason |
| --- | --- | --- |
| **1** | Build test matrices instantly. | Combine `np.arange(...).reshape(...)` to test functions rapidly. |
| **2** | Axis orientation rules. | `axis=0` runs down columns (vertical); `axis=1` runs across rows (horizontal). |
| **3** | Slices alter master arrays. | Slices are **views**, not copies. Modify a slice and you modify the master. Use `.copy()`. |
| **4** | Save memory profiles. | Convert data container types explicitly using `.astype()` to optimize physical footprint. |
| **5** | Continuous plot axes. | Use `np.linspace` when graphing functions to get perfectly predictable, smooth x-axis values. |
| **6** | Keep code clean. | Prefer property attribute calling `arr.T` over explicit method call `np.transpose(arr)`. |
| **7** | Masking foundation. | Relational calls like `arr == val` create Boolean masks for quick filtering (explored in next stage). |
| **8** | Bypass nested for-loops. | `np.nditer(arr)` handles multi-dimensional item iteration cleanly without nesting loop layers. |
| **9** | Vectorization supremacy. | Always prioritize vectorized expressions over manual Python loops—they run optimized C speed. |
| **10** | Square vs. Rectangular Identity. | Use `np.identity(n)` for square identity matrices; use `np.eye(n, m)` for rectangular layouts. |

---

## 🚀 Roadmap (Future Additions)

These advanced sections will be appended tracking progression order:

* [ ] Broadcasting Rules (Deep Dive Matrix Mechanics)
* [ ] Fancy Indexing & Boolean Mask Filtering Arrays
* [ ] Conditional Selection Tracking (`np.where`, `np.argmax`, `np.argmin`, `np.argsort`)
* [ ] Structured Vector Set Operations (`np.unique`, `np.intersect1d`, etc.)
* [ ] Advanced Structural Axis Manipulation (`expand_dims`, `squeeze`, `newaxis`)
* [ ] Mathematical & Trigonometric Processing Engines
* [ ] Linear Algebra Sub-module Execution (`np.linalg`)
* [ ] Random Generation Distributions & Seed Operations (`np.random`)
* [ ] Persistent Local Storage Input/Output Operations (`np.save`, `np.load`, `np.loadtxt`)
* [ ] Deep Dive Memory Benchmarking: Contiguous Block C-Views vs Copies
* [ ] Coordinate Mapping Evaluation via Meshgrids (`np.meshgrid`)

---
