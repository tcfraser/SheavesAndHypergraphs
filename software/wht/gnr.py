"""
    GNR: Generalized node relators
"""
from scipy import sparse as sp
import numpy as np
from .config import *
from .overlap import *
from .combinatorics import inv

α, β, γ, δ = 0, 1, 2, 3 # enumerate the types

class OrigDestMap():

    def __init__(self, orig, dest):
        """
            orig : the position of the node in a past set of generalized nodes
            dest : the position(s) of the node in a future set of generalized nodes
                   '(s)' includes the case of a γ type relation
        """
        self.orig = orig
        self.dest = dest
        # self.map = {}
        # if self.orig is not None:
        #     for i in range(self.orig.size):
        #         if isinstance(self.dest, tuple):
        #             self.map[self.orig[i]] = tuple(self.dest[q][i] for q in range(len(self.dest)))
        #         else:
        #             self.map[self.orig[i]] = self.dest[i]

    def __repr__(self):
        if isinstance(self.dest, tuple):
            return "  orig : {} \n    dest : {} \n           {}".format(self.orig, self.dest[0], self.dest[1])
        else:
            return "  orig : {} \n    dest : {}".format(self.orig, self.dest)

class Relator():

    def __init__(self, prev, next_edge):
        """
            Given
                prev      : generalized nodes of previous partial hypergraph
                next_edge : the next edge E to be added to past

            Computed
                next      : generalized nodes of next partial hypergraph
                rel       : an ordered list between past and future
        """
        self.prev = prev
        _rel = [[],[],[],[]] # Temporary data storage for self._rel
        _next = [] # Temporary storage for self.next

        E = next_edge # just aliasing
        E_r = E.copy() # Remainder of new edge

        for p, X in enumerate(prev):
            # @Efficiency: E_r is here instead of E because the generalized_nodes are assumed disjoint
            # @Efficiency: E_r is typically smaller than X. Should always put smaller list first
            E_minus_X, E_intersect_X, X_minus_E = overlap(E_r, X)
            E_r = E_minus_X

            if E_intersect_X.size == 0: # X and E disjoint
                _rel[α].append((p, len(_next)))
                _next.append(X)
            elif X_minus_E.size == 0: # X subset of E
                _rel[β].append((p, len(_next)))
                _next.append(X)
            else: # X and E strictly overlap OR E is subset of X
                _rel[γ].append((p, (len(_next), len(_next)+1)))
                _next.append(X_minus_E)
                _next.append(E_intersect_X)

        if E_r.size > 0: # New nodes were introduced
            _rel[δ].append((0, len(_next))) # Setting orig to 0 for dtype reasons. set to None later
            _next.append(E_r)


        self._rel = [None,None,None,None]
        # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        _next_argsort = sorted(range(len(_next)), key=lambda i: _next[i][0])
        self.next = [_next[i] for i in _next_argsort]
        _next_argsort_inv = inv(_next_argsort)

        for g in (α, β, δ):
            orig = np.empty(len(_rel[g]), dtype=DTYPE) # locations in prev
            dest = np.empty(len(_rel[g]), dtype=DTYPE) # locations in next
            for i, r in enumerate(_rel[g]):
                orig[i] = r[0]
                dest[i] = _next_argsort_inv[r[1]]
            if g == δ: orig = None
            self._rel[g] = OrigDestMap(orig, dest)

        for g in (γ,):
            orig = np.empty(len(_rel[g]), dtype=DTYPE) # locations in prev
            dest0 = np.empty(len(_rel[g]), dtype=DTYPE) # locations in next
            dest1 = dest0.copy()
            for i, r in enumerate(_rel[g]):
                orig[i] = r[0]
                dest0[i] = _next_argsort_inv[r[1][0]]
                dest1[i] = _next_argsort_inv[r[1][1]]
            self._rel[g] = OrigDestMap(orig, (dest0, dest1))

    def __getitem__(self, item):
        return self._rel[item]

    def __repr__(self):
        return "{} :: \n past   = \n {} \n future = \n {} \n rel = \n {}".format(
            self.__class__.__name__, self.prev, self.next, '\n '.join((['α','β','γ','δ'][g] + str(r) for g, r in enumerate(self._rel))))

class Relators():

    def __init__(self, H):
        self.H = sp.csc_matrix(H)
        self.length = self.H.shape[1] - 1 # the number of relators is equal to the columns of H minus 1

        edge_init = self.H.indices[self.H.indptr[0]:self.H.indptr[1]]
        relator = Relator([], edge_init)
        relators = []
        for i in range(1, self.length + 1, 1):
            relator = Relator(relator.next, self.H.indices[self.H.indptr[i]:self.H.indptr[i+1]])
            relators.append(relator)
        self.relators = relators

    def __getitem__(self, slice):
        return self.relators[slice]

    def __repr__(self):
        return "{} :: \n {}".format(self.__class__.__name__, '\n'.join(map(str, self.relators)))