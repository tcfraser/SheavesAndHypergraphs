from itertools import combinations
import numpy as np

def partitions(n, k):
    """
        Generates all arrays of k non-negative integers that sum to n
    """
    out = np.zeros(k, dtype='int32')
    yield from _p_worker(out, 0, n)

def _p_worker(out, i, n_left):
    if i == out.size - 1:
        out[i] = n_left
        yield out
    else:
        for c in range(n_left + 1):
            out[i] = c
            yield from _p_worker(out, i+1, n_left - c)

def restricted_partitions(n, r):
    """
        Generates all arrays of |r| non-negative integers bounded by r that sum to n
    """
    out = np.zeros(r.size, dtype='int32')
    yield from _rp_worker(out, 0, n, r, np.sum(r))

def _rp_worker(out, i, n_left, r, r_left):
    if r_left >= n_left:
        if i == out.size - 1:
            out[i] = n_left
            yield out
        else:
            for c in range(min(r[i], n_left)+1):
                out[i] = c
                yield from _rp_worker(out, i+1, n_left - c, r, r_left - r[i])