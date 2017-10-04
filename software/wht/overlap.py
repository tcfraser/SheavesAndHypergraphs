import numpy as np
from .config import *

def overlap(A, B):
    """
        Symmetric overlap function that assumes a and b are sorted with unique elements

        returns (A / B, A & B, B / A)
    """
    A = np.asarray(A, dtype=DTYPE)
    B = np.asarray(B, dtype=DTYPE)

    # Masks for later selection
    A_minus_B_mask = np.ones(A.size, dtype='bool')
    B_minus_A_mask = np.ones(B.size, dtype='bool')
    a, b = 0, 0
    while (a < A.size and b < B.size):
        add_a = 0
        add_b = 0
        if A[a] == B[b]:
            A_minus_B_mask[a] = False
            B_minus_A_mask[b] = False
        if A[a] <= B[b]:
            add_a = 1
        if A[a] >= B[b]:
            add_b = 1
        a += add_a
        b += add_b

    A_minus_B = A[A_minus_B_mask]
    B_minus_A = B[B_minus_A_mask]

    A_intersect_B_mask = np.invert(A_minus_B_mask, out=A_minus_B_mask) # A_minus_B_mask has the right size but is no longer needed
    A_intersect_B = A[A_intersect_B_mask]

    return (A_minus_B, A_intersect_B, B_minus_A)
