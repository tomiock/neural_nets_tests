import numba
import numpy as np
import matplotlib.pyplot as plt

@numba.jit(nopython=True)
def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges

@numba.jit(nopython=True)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin

@numba.jit(nopython=True)
def numba_histogram(a, bins):
    hist = np.zeros((bins,), dtype=np.intp)
    bin_edges = get_bin_edges(a, bins)

    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges

# Generate random data
rng = np.random.default_rng(42)  # Use a fixed seed for reproducibility
a = rng.random(100) * 10  # Random numbers between 0 and 10
bins = 15

# Calculate histogram using Numba
numba_hist, numba_bin_edges = numba_histogram(a, bins)

# Calculate histogram using NumPy
numpy_hist, numpy_bin_edges = np.histogram(a, bins=bins)

plt.plot(numpy_hist)
plt.show()

plt.plot(numba_hist)
