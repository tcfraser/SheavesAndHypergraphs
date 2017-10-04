"""
    The hypergraph datastructure and data structure conversion methods
"""
import numpy as np
from .partial_sparse import PartialSparseMatrix, PartialSparseVector
from .config import *

class Hypergraph(PartialSparseMatrix):
    def __init__(self, indices, indptr):
        super().__init__(indices, indptr, dtype=HYPERGRAPH_DTYPE)

def dualize(H):
    """
        Calculates the dual hypergraph (transpose) of a hypergraph H
        Assumes the edges are sorted and no rows are degenerate
    """
    dual_indices = np.zeros(H.indices.size, dtype=HYPERGRAPH_DTYPE)
    dual_indptr  = np.zeros(H.shape[0] + 1, dtype=HYPERGRAPH_DTYPE)

    # First determine dual_indptr
    for index in H.indices:
        dual_indptr[index+1] += 1

    ptr_sum = 0
    for i, count in enumerate(dual_indptr):
        if i != 0:
            ptr_sum += dual_indptr[i]
            dual_indptr[i] = ptr_sum
    # dual_indptr finished and valid

    # dual_counts is memory space for keeping track of where we are in each dual_indices
    dual_counts = np.zeros(H.shape[0], dtype=HYPERGRAPH_DTYPE)
    dual_index = 0
    for i, index in enumerate(H.indices):
        if i == H.indptr[dual_index+1]:
            dual_index += 1
        dual_indices[dual_indptr[index] + dual_counts[index]] = dual_index
        dual_counts[index] += 1

    return Hypergraph(dual_indices, dual_indptr)

def hypergraph_to_dense(H):
    """
        Converts a hypergraph H into a boolean dense matrix D
    """
    D = np.zeros(H.shape, dtype='bool')
    for e in range(H.shape[1]):
        D[H[e], e] = True
    return D

def dense_to_hypergraph(D):
    """
        Converts a dense matrix D (casted as a boolean) into a hypergraph H
    """
    D = D.astype('bool', copy=True)
    indices_list = [np.nonzero(D[:,i])[0].astype(HYPERGRAPH_DTYPE) for i in range(D.shape[1])]

    indptr = np.zeros(len(indices_list)+1,dtype=HYPERGRAPH_DTYPE)
    indptr_counter = 0
    for e, edge in enumerate(indices_list):
        indptr_counter += edge.size
        indptr[e+1] = indptr_counter

    indices = np.concatenate(indices_list)

    return Hypergraph(indices, indptr)