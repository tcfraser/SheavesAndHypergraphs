import wht
# import wht.fileio
# import wht.overlap
import wht.transversal
import wht.gnr
# import wht.combinatorics
import numpy as np
from scipy import sparse as sp


def go():
    # h = sparse.csr_matrix((np.random.rand(18,18)*2).astype('int32'))
    # x1 = sparse.csr_matrix(
    #     (np.array(list(range(6))),np.array(list(range(0,12,2))), np.array([0,6])), shape=(1, 18)
    #     )
    # x2 = sparse.csr_matrix(
    #     (np.array(list(range(6))),np.array(list(range(0,18,3))), np.array([0,6])), shape=(1, 18)
    #     )
    # print(x1.indices, x1.data)
    # print(x2.indices, x2.data)
    # print(x1.has_sorted_indices)
    # print(x2.has_sorted_indices)
    # x3 = x1 + x2
    # print(x3.has_sorted_indices)
    # print(x3.indices, x3.data)
    # print(h.todense())
    # print(x3.dot(h))

    # x = [1, 0, 0, 0, 2]
    # h = [[1, 0, 0, 0],
    #      [0, 1, 0, 1],
    #      [0, 1, 0, 0],
    #      [0, 0, 1, 0],
    #      [0, 0, 0, 1]]
    # xsr = sparse.csr_matrix(x)
    # xsc = sparse.csc_matrix(x)
    # hsr = sparse.csr_matrix(h)
    # hsc = sparse.csc_matrix(h)
    # print(xsr.indices, xsr.indptr)
    # print(xsc.indices, xsc.indptr)
    # print(hsr.indices, hsr.indptr)
    # print(hsc.indices, hsc.indptr)
    # print(xsr * hsr)
    # print(xsc * hsc)

    # xsr[:,np.array([0,2])] = np.array([2,2])
    # return
#     h = wht.hypergraph.Hypergraph(
#             np.array([0,1,1,2,0,2]),
#             np.array([0,2,4,6])
#         )
#     d = [[1,0,1],
#          [1,1,0],
#          [0,1,1]]
#     d = np.array(d)
#     print('h')
#     print(h)
#     print(wht.hypergraph.dense_to_hypergraph(d))
#     print('d')
#     print(d)
#     print(wht.hypergraph.hypergraph_to_dense(h).astype('int'))
#     print("Partitions of 5 with restriction [2,2,2,2]")
#     for p in wht.combinatorics.restricted_partitions(5,np.array([2,2,2,2])):
#         print(p)
#     print("Partitions of 5 into k integers")
#     for p in wht.combinatorics.partitions(5,4):
#         print(p)

    d = [[0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0],
         [0,1,0,1,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0],
         [1,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
         [0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,0],
         [0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0],
         [0,0,1,0,1,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,0,0],
         [0,0,1,0,1,0,1,0,0,1,1,0,1,0,0,0,0,1,0,1,0,1,1,1,0,0],
         [1,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,1],
         [1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0],
         [1,0,1,1,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0],
         [1,0,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
         [1,0,1,0,0,0,0,1,1,0,0,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0]]
    # d = [[0,1,0,0,0,1],
    #      [1,1,0,1,0,1],
    #      [1,1,1,0,0,1],
    #      [1,0,1,0,1,1],
    #      [0,0,1,0,0,0]]
    # d = [[1,0,0,0],
    #      [0,1,0,0],
    #      [0,0,1,1],
    #      [0,0,0,1]]
    # d = [[1,0],
    #      [1,1]]
    d = np.array(d)
    print(d)
    h = sp.csr_matrix(d)
    relators = wht.gnr.Relators(h)
    # print(relators)
    # w = sp.csr_matrix([[2,1,3,1,4,1]])
    w = sp.csr_matrix([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    # w = sp.csr_matrix([[1,1,1,3]])
    # w = sp.csr_matrix([[1,2]])
    print(w.todense())
    print(' -- Transversals -- ')
    for x in wht.transversal.generate_w_transversals(h, w):
        print(x.todense())


#     wht.fileio.save_hypergraph('tmp/test', h)

#     h = wht.fileio.load_hypergraph('tmp/test')

#     ht = wht.hypergraph.dualize(h)
#     print(h)
#     print(d)
#     # print(wht.hypergraph.hypergraph_to_dense(h).astype('int'))
#     # print(ht)
#     # print(wht.hypergraph.hypergraph_to_dense(ht).astype('int'))

#     # print(wht.overlap.overlap([0,1,2,3,4,5],[3,4,5,6,7,8]))
#     # print(wht.overlap.overlap([0,3,6,8],[]))

#     # gnr = wht.transversal.GeneralizedNodeRelation([h[0]],h[1])
#     # print(gnr)
#     # for e in h:
#     #     gnr = wht.transversal.GeneralizedNodeRelation(gnr.future,e)
#     #     print(gnr)
#     # w = np.array([2,1,3,1,4,1])
#     # print(wht.transversal.generate_w_transversals(h,w))
    # print('w', str(w))

if __name__ == "__main__":
    go()
