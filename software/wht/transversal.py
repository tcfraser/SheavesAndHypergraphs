from .hypergraph import Hypergraph, dualize
from .partial_sparse import PartialSparseVector
from .overlap import overlap
from .config import *
from .combinatorics import restricted_partitions, partitions
import numpy as np

class Transversal(PartialSparseVector):

    def __init__(self, indices, values, shape):
        super().__init__(indices, values, shape=shape, dtype=TRANSVERSAL_DTYPE)

α, β, γ, δ = 0, 1, 2, 3 # enumerate the types
class GeneralizedNodeRelator():

    def __init__(self, orig, dest):
        """
            orig : the position of the node in a past set of generalized nodes
            dest : the position(s) of the node in a future set of generalized nodes
                   '(s)' includes the case of a γ type relation
        """
        self.orig = orig
        self.dest = dest

    def __repr__(self):
        if isinstance(self.dest, tuple):
            return "  orig      : {} \n    dest : {} \n        {}".format(self.orig, self.dest[0], self.dest[1])
        else:
            return "  orig      : {} \n    dest : {}".format(self.orig, self.dest)

class GeneralizedNodeRelation():

    def __init__(self, past, next_edge):
        """
            Given
                past      : generalized nodes of previous partial hypergraph
                next_edge : the next edge E to be added to past

            Computed
                future    : generalized nodes of next partial hypergraph
                rel       : (relators) an ordered list between past and future
        """
        self.past = past
        self.future = []
        _tmp = [[],[],[],[]] # Temporary data storage for self._rel
        self.rel = [None,None,None,None]

        E = next_edge # just aliasing
        E_r = E.copy() # Remainder of new edge

        for p, X in enumerate(past):
            # @Efficiency: E_r is here instead of E because the generalized_nodes are assumed disjoint
            # @Efficiency: E_r is typically smaller than X. Should always put smaller list first
            E_minus_X, E_intersect_X, X_minus_E = overlap(E_r, X)
            E_r = E_minus_X

            if E_intersect_X.size == 0: # X and E disjoint
                _tmp[α].append((p, len(self.future)))
                self.future.append(X)
            elif X_minus_E.size == 0: # X subset of E
                _tmp[β].append((p, len(self.future)))
                self.future.append(X)
            else: # X and E strictly overlap OR E is subset of X
                _tmp[γ].append((p, len(self.future)))
                self.future.append(X_minus_E)
                self.future.append(E_intersect_X)

        if E_r.size > 0: # New nodes were introduced
            _tmp[δ].append((0, len(self.future))) # Setting orig to 0 for dtype reasons. set to None later
            self.future.append(E_r)

        for t in (α, β, γ, δ):
            orig = np.empty(len(_tmp[t]), dtype=HYPERGRAPH_DTYPE) # locations in past
            dest = np.empty(len(_tmp[t]), dtype=HYPERGRAPH_DTYPE) # locations in future
            for i, r in enumerate(_tmp[t]):
                orig[i] = r[0]
                dest[i] = r[1]

            if t == γ:
                dest = (dest, np.add(dest, 1))
            if t == δ:
                orig = None

            self.rel[t] = GeneralizedNodeRelator(orig, dest)

    def __repr__(self):
        return "{} :: \n past   = \n {} \n future = \n {} \n rel = \n {}".format(
            self.__class__.__name__, self.past, self.future, '\n '.join((['α','β','γ','δ'][t] + str(r) for t, r in enumerate(self.rel))))

def generate_γ_offspring(gnr, t, contrib):
    """
        For nodes of type γ (γ1, γ2), there is a choice for each pair
        Need to select t[r.orig] elements out of (t[r.dest[0]], t[r.dest[1]])
    """
def compute_t_dot_H(t, H, gn, k = -1):
    """
        Given a transversal t over generalized nodes gn of H, compute the multiplication t.H
        This is simply fast sparse matrix multiplication

        t  : the transversal
        gn : the generalized nodes for H
        H  : the hypergraph
        k  : the size of the output extension
    """
    if k < 0:
        k = H.shape[0]
    T = np.zeros(k, dtype=WEIGHT_DTYPE) # result of t.H
    for i in range(t.size):
        rep = gn[i][0] # picks out a representative
        ext = H.indices[H.indptr[rep]:H.indptr[rep+1]]

        T[ext] += t[i] # effectively generates partial hypergraph dual
    return T

def add_next_hyperedge(H, H_dual, w, gn, t, T, k):
    """
        H       : a hypergraph transversed by (trans) w.r.t. (gen_nodes)
        H_dual  : the hypergraph's dual
        w       : a weights overcome by (trans).(hypergraph)
        gn      : the generalized nodes associated with (trans)
        t       : a (weights)-transversal of (hypergraph)
        T       : extension of (trans)
        k       : the hyperedge being added
    """
    if k >= H.shape[1]: # there was no edge to add
        yield t
    else: # otherwise continue as normal
        gnr = GeneralizedNodeRelation(gn, H[k])

        # Determine manditory members (mm)
        #   Each X of type α or β need to be kept with the same weight
        blank = np.zeros(len(gnr.future), dtype=TRANSVERSAL_DTYPE) # blank for new transversal
        mm = blank.copy()
        mm[gnr.rel[α].dest] = t[gnr.rel[α].orig]
        mm[gnr.rel[β].dest] = t[gnr.rel[β].orig]
        β_contrib = np.sum(mm[gnr.rel[β].dest])

        γ_contrib_max = np.sum(t[gnr.rel[γ].orig])
        if w[k] - β_contrib > 0:
            # will need to perform a number of expensive operations in order to determine how to supplement the offspring
            # (see the case below where r_w > 0)
            # locations of the potential contributors
            loc_contrib = np.concatenate((gnr.rel[β].dest, gnr.rel[δ].dest, gnr.rel[γ].dest[1]))

        for γ_contrib in range(γ_contrib_max + 1, -1, -1):
            r_w = w[k] - β_contrib - γ_contrib # r_w is the remaining weight to be covered

            if r_w <= 0: # Already covered next edge!
                for γ_choice in restricted_partitions(γ_contrib, t[gnr.rel[γ].orig]):
                    offspring = blank.copy()
                    offspring[gnr.rel[γ].dest[0]] = t[gnr.rel[γ].orig] - γ_choice
                    offspring[gnr.rel[γ].dest[1]] = γ_choice
                    yield from add_next_hyperedge(H, H_dual, w, gnr.future, offspring + mm, np.append(T, β_contrib + γ_contrib), k+1)
            else: # Have not covered the next edge, need to cover the difference
                # need to generate appropriate suppliments
                for suppliment_values in partitions(r_w, loc_contrib.size):
                    suppliment = blank.copy()
                    suppliment[loc_contrib] = suppliment_values
                    print('mm', str(mm))
                    print('s', str(suppliment))
                    pass

def generate_w_transversals(H, w):
    """
        generates (iterative depth-first) all of the minimal w-transversals of H

        H : hypergraph
        w : weight vector
    """
    w = np.asarray(w, dtype=WEIGHT_DTYPE)
    H_dual = dualize(H) # possible bottleneck for memory
    if H.shape[1] == 0: # the hypergraph is empty
        return []
    else:
        # need to construct the initial transversal and initial gnr
        gnr_init = GeneralizedNodeRelation([], H[0])
        gn_init  = gnr_init.future
        t_init   = Transversal(np.array([0]), np.array([w[0]]), shape=(1,))
        T_init   = np.array([w[0]], dtype=WEIGHT_DTYPE)
        k_init   = 1

        transversal_generator = add_next_hyperedge(H, H_dual, w, gn_init, t_init, T_init, k_init)
        print(list(transversal_generator))
        return transversal_generator

