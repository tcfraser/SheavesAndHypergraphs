from .overlap import overlap
from .config import *
from .gnr import *
from .combinatorics import restricted_partitions, partitions
from scipy import sparse as sp
import numpy as np

def gdot(t, H, gn):
    # First convert t to a transversal over the representatives of H
    rep_indices = np.fromiter((gn[i][0] for i in t.indices), dtype=DTYPE, count=t.indices.size)

    rep = sp.csr_matrix((t.data, rep_indices, t.indptr), shape=(1, H.shape[0]))
    rep.has_sorted_indices = True
    return rep * H

def add_next_hyperedge(H, w, rels, k, t):
    """
        yields t_k
    """
    assert(k < H.shape[1])
    rel = rels[k]

    # Determine manditory members (mm)
    #   Each X of type α or β need to be kept with the same weight
    t_data = [t[rel[g].orig].data for g in (α,β,γ)]

    β_contrib = np.sum(t_data[α])
    γ_contrib_max = np.sum(t_data[γ])

    def fu(dest, data): # macro
        return sp.csr_matrix((data, dest, np.array([0,dest.size], dtype=DTYPE)), shape=(1,len(rel.next)))

    t_k_α = fu(rel[α].dest, t_data[α])
    t_k_β = fu(rel[β].dest, t_data[β])
    t_k_m = t_k_α + t_k_β # TODO disjoint sum inefficiency

    if w[0,k] - β_contrib > 0:
        # will need to perform a number of expensive operations in order to determine how to supplement the offspring
        # (see the case below where r_w > 0)
        # locations of the potential contributors
        loc_contrib = np.concatenate((rel[β].dest, rel[δ].dest, rel[γ].dest[1]))
        t_ext = (gdot(t, H, rel.prev) - w)[0,:k+1] # TODO inefficiency
        failure_points = []
        for index in t.indices:
            rep = rel.prev[index][0]
            removal_indices = H.indices[H.indptr[rep]:H.indptr[rep+1]]
            removal_indices = removal_indices[removal_indices <= k]
            failure_points.append(overlap(removal_indices, t_ext.indices)[0]) # removal_indices / T.indices

    def generate_γ_offspring(contrib):
        for γ_choice in restricted_partitions(contrib, t_data[γ]):
            t_k_γ1 = fu(rel[γ].dest[0], t_data[γ] - γ_choice)
            t_k_γ2 = fu(rel[γ].dest[1], γ_choice)
            t_k_γ = t_k_γ1 + t_k_γ2 # TODO can ravel these together # TODO disjoint sum inefficiency
            t_k = t_k_γ + t_k_m # TODO disjoint sum inefficiency
            yield t_k

    for γ_contrib in range(γ_contrib_max + 1, - 1, -1): # start with largest contributions and work backwards
        r_w = w[0,k+1] - β_contrib - γ_contrib # r_w is the remaining weight to be covered

        if r_w <= 0: # Already covered next edge!
            for offspring in generate_γ_offspring(γ_contrib):
                yield offspring
        else: # Have not covered the next edge, need to cover the difference
            # need to generate appropriate suppliments
            for supp_values in partitions(r_w, loc_contrib.size):
                supp = fu(loc_contrib, supp_values)
                supp_ext = gdot(supp, H, rel.next)[0,:k+1] # TODO inefficiency
                appropriate = True
                for fp in failure_points:
                    if supp_ext[fp].nnz >= fp.size:
                        # The suppliment makes a previous node redundant
                        appropriate = False
                        break
                if appropriate:
                    for offspring in generate_γ_offspring(γ_contrib):
                        yield supp + offspring
                else:
                    pass # TODO solve the hierarchy problem for partition inclusion

def _transversal_worker(H, w, rels, k, t):
    if k == H.shape[1]:
        yield t
    else:
        for tt in add_next_hyperedge(H, w, rels, k, t):
            yield from _transversal_worker(H, w, rels, k+1, tt)

def generate_w_transversals(H, w):
    """
        generates (iterative depth-first) all of the minimal w-transversals of H

        H : hypergraph
        w : weight vector
    """
    assert(H.shape[1] != 0), "The hypergraph is empty!"

    w = sp.csr_matrix(w, dtype=DTYPE)
    H_csr = H.tocsr()
    H_csc = H.tocsc()

    rels = Relators(H_csc)

    t_init = sp.csr_matrix(
        (np.array([w[0,0]], dtype=DTYPE), # data
         np.array([0],      dtype=DTYPE), # indices
         np.array([0,1],    dtype=DTYPE))  # indptr
        , shape=(1,1))

    transversal_generator = _transversal_worker(H, w, rels, 0, t_init)
    for t in transversal_generator:
        print(t.todense())



