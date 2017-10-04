"""
    Saving and loading methods for hypergraphs
"""
import numpy as np
# from .hypergraph import Hypergraph

def save_hypergraph(filename, h):
    np.savez(filename, indices=h.indices, indptr=h.indptr)

def load_hypergraph(filename):
    loader = np.load(filename+'.npz')
    return Hypergraph(indices=loader['indices'], indptr=loader['indptr'])