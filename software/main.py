import wht
import wht.hypergraph
import wht.fileio
import wht.overlap
import wht.transversal
import wht.combinatorics
import numpy as np

if __name__ == "__main__":
    h = wht.hypergraph.Hypergraph(
            np.array([0,1,1,2,0,2]),
            np.array([0,2,4,6])
        )
    d = [[1,0,1],
         [1,1,0],
         [0,1,1]]
    d = np.array(d)
    print('h')
    print(h)
    print(wht.hypergraph.dense_to_hypergraph(d))
    print('d')
    print(d)
    print(wht.hypergraph.hypergraph_to_dense(h).astype('int'))
    print("Partitions of 5 with restriction [2,2,2,2]")
    for p in wht.combinatorics.restricted_partitions(5,np.array([2,2,2,2])):
        print(p)
    print("Partitions of 5 into k integers")
    for p in wht.combinatorics.partitions(5,4):
        print(p)

    d = [[0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
         [0,1,0,1,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0],
         [1,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
         [0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,0],
         [0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0],
         [0,0,1,0,1,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,0,0],
         [0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,1,1,1,0,0],
         [1,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0],
         [1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0],
         [1,0,1,1,0,0,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0],
         [1,0,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
         [1,0,1,0,0,0,0,1,1,0,0,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0]]
    d = [[0,1,0,0,0,1],
         [1,1,0,1,0,1],
         [1,1,1,0,0,1],
         [1,0,1,0,1,1],
         [0,0,1,0,0,0]]
    d = np.array(d)

    h = wht.hypergraph.dense_to_hypergraph(d)
    wht.fileio.save_hypergraph('tmp/test', h)

    h = wht.fileio.load_hypergraph('tmp/test')

    ht = wht.hypergraph.dualize(h)
    print(h)
    print(d)
    # print(wht.hypergraph.hypergraph_to_dense(h).astype('int'))
    # print(ht)
    # print(wht.hypergraph.hypergraph_to_dense(ht).astype('int'))

    # print(wht.overlap.overlap([0,1,2,3,4,5],[3,4,5,6,7,8]))
    # print(wht.overlap.overlap([0,3,6,8],[]))

    # gnr = wht.transversal.GeneralizedNodeRelation([h[0]],h[1])
    # print(gnr)
    # for e in h:
    #     gnr = wht.transversal.GeneralizedNodeRelation(gnr.future,e)
    #     print(gnr)
    # w = np.array([2,1,3,1,4,1])
    # print(wht.transversal.generate_w_transversals(h,w))
    w = np.array([2,1,3,1,4,1])
    print('w', str(w))
    wht.transversal.generate_w_transversals(h, w)
