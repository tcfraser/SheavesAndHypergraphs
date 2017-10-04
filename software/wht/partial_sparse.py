import numpy as np
from .config import *

class PartialSparseVector():
    def __init__(self, indices, values, shape, dtype=PARTIAL_SPARSE_DEFAULT_DTYPE):
        self.indices = np.asarray(indices, dtype=dtype)
        self.values  = np.asarray(values,  dtype=dtype)
        self.dtype   = dtype

        self.shape   = shape
        # self.shape   = shape if shape is not None else (np.max(self.indices) + 1)
        self.size    = self.indices.size
        self.density = self.size / (self.shape[0])

    def __getitem__(self, slice):
        # Always assuming slice is sorted

        return self.indices[self.indptr[i]:self.indptr[i+1]]


class PartialSparseMatrix():
    def __init__(self, indices, indptr, shape = None, dtype=PARTIAL_SPARSE_DEFAULT_DTYPE):
        self.indices = np.asarray(indices, dtype=dtype) # Assumed sorted
        self.indptr  = np.asarray(indptr,  dtype=dtype)
        self.dtype   = dtype

        self.shape   = shape if shape is not None else (np.max(self.indices) + 1, self.indptr.size - 1)
        self.size    = self.indices.size
        self.density = self.size / (self.shape[0] * self.shape[1])

    def __getitem__(self,i):
        return self.indices[self.indptr[i]:self.indptr[i+1]]

    def __repr__(self):
        return "{} :: shape={}, size={}, density={:.4f}, dtype={}".format(
            self.__class__.__name__, self.shape, self.size, self.density, self.dtype)