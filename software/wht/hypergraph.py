"""
    The hypergraph datastructure and data structure conversion methods
"""
import numpy as np

HYPERGRAPH_DTYPE = 'int32'

class Hypergraph():
    def __init__(self, indices, indptr):
        self.indices = np.asarray(indices,dtype=HYPERGRAPH_DTYPE)
        self.indptr  = np.asarray(indptr, dtype=HYPERGRAPH_DTYPE)

        self.num_edges = self.indptr.size - 1
        self.num_nodes = np.max(self.indices) + 1
        self.shape     = (self.num_nodes, self.num_edges)
        self.size      = self.indices.size
        self.density   = self.size / (self.num_edges * self.num_nodes)
        self.dtype     = self.indices.dtype

    def __repr__(self):
        return "{} :: shape={}, size={}, density={:.4f}, dtype={}".format(
            self.__class__.__name__, self.shape, self.size, self.density, self.dtype)

    def get_edge(self, e):
        return self.indices[self.indptr[e]:self.indptr[e+1]]

def hypergraph_to_dense(h):
    """
        Converts a hypergraph h into a boolean dense matrix d
    """
    d = np.zeros(h.shape, dtype='bool')
    for e in range(h.num_edges):
        d[h.get_edge(e), e] = True
    return d

def dense_to_hypergraph(d):
    """
        Converts a dense matrix d (casted as a boolean) into a hypergraph h
    """
    d = d.astype('bool', copy=True)
    indices_list = [np.nonzero(d[:,i])[0].astype(HYPERGRAPH_DTYPE) for i in range(d.shape[1])]

    indptr = np.zeros(len(indices_list)+1,dtype=HYPERGRAPH_DTYPE)
    indptr_counter = 0
    for e, edge in enumerate(indices_list):
        indptr_counter += edge.size
        indptr[e+1] = indptr_counter

    indices = np.concatenate(indices_list)

    return Hypergraph(indices, indptr)

if __name__ == "__main__":
    h = Hypergraph(
            np.array([0,1,1,2,0,2]),
            np.array([0,2,4,6])
        )
    d = [[1,0,1],
         [1,1,0],
         [0,1,1]]
    d = np.array(d)
    print('h')
    print(h)
    print(dense_to_hypergraph(d))
    print('d')
    print(d)
    print(hypergraph_to_dense(h).astype('int'))

