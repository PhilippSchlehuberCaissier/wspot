import itertools
from dataclasses import dataclass
from typing import List, Set, Tuple, Union
from copy import copy, deepcopy

def dprint(*args, **kwargs):
    print(*args, **kwargs)
def transpose(G:List[List[int]]) -> List[List[int]]:
    """
    Returns the transposed graph
    """
    GT = [[] for _ in range(len(G))]

    for src, alist in enumerate(G):
        for dst in alist:
            GT[dst].append(src)

    return GT

# List of Predescessor
# 0 : Dest state
# 0' <-> 7 : Src state
# The ith list here corresponds to the list of optimal predecessors
# (ordered by their time of appearance) of the ith state
# The backedge is so to speak "bent away" for simplicity.
# Instead of going back to 0 from 1 we go to 7 which is
# implicitly the same as 0
P1 = [[],
      [0, 2, 5, 2],
      [1, 3, 1, 4],
      [2, 3],
      [2, 4],
      [1, 6, 1],
      [5, 6],
      [1]]

# Corresponding graph
# adjacency list
G1 = [[1],  # 0
      [2, 5, 7],  # 1
      [1, 3, 4],  # 2
      [2, 3],  # 3
      [2, 4],  # 4
      [2, 6],  # 5
      [5, 6],  # 6
      [],  # 7
      ]


# Energy definition of the problem
WUP1 = 75
W1 = dict()
for (s, d, w) in [(0, 1, 0),
                  (1, 2, 0), (1, 5, 0), (1, 7, -WUP1),
                  (2, 1, 0), (2, 3, 0), (2, 4, -WUP1 * 2 / 3),
                  (3, 2, -WUP1 * 2 / 3), (3, 3, 1),
                  (4, 2, 0), (4, 4, 1),
                  (5, 1, 0), (5, 6, -WUP1 * 1 / 3),
                  (6, 5, -WUP1 * 1 / 3), (6, 6, 1),
                  ]:
    W1[(s, d)] = w

P2 = [[],
      [0, 1],
      [1]]

P3 = [[],
      [0, 2],
      [1, 3],
      [2],
      [1],
      [4]]

P = P1
W = W1
wup = WUP1
G = G1
GT = transpose(G)


InnerMost = [0] * len(P)


def rec(s: int, dst: int):
    if s == 0:
        return [[0]]

    if InnerMost[s] == len(P[s]):
        return []

    if s == dst:
        return [[s]]

    InnerMost[s] += 1

    prec = []
    for idxp in range(InnerMost[s], len(P[s])):
        print()
        prec += [x for x in rec(P[s][idxp], s) if x[0] == s]  # Only path that loop back to s or go to 0 are valid

    all_inter = [[tuple(x + [s])] for x in prec]  # All of these are cycles
    all_inter.append([s])

    # Form combinations
    all_pre = rec(P[s][InnerMost[s] - 1], 0)

    all_paths = []
    for apre in all_pre:
        for ainter in all_inter:
            all_paths.append(apre + ainter)

    InnerMost[s] -= 1

    return all_paths


Pprime = [[]] + [l[1:] + [l[0]] for l in P[1:]]

# Pprime = [[]] + [l+[l[0]] for l in P[1:]]

from copy import deepcopy


# allPaths = []
# hashedIdx = set()
# NOOPT = True
def rec2(s: int, idx: List[int], path: List[int]):
    path = [s] + path

    idxp = deepcopy(idx)
    idxp[s] += 1

    if s == 0:
        allPaths.append(path)

    for i in range(idxp[s], len(Pprime[s])):
        idxpp = deepcopy(idxp)
        idxpp[s] = i
        idxpptp = tuple(idxpp)
        if (idxpptp not in hashedIdx) or NOOPT:
            hashedIdx.add(idxpptp)
            rec2(Pprime[s][idxpp[s]], idxpp, path)


def rec3(visited: Union[None, Set[Tuple[int, ...]]],
         all_path: List[List[int]],
         Pprime: List[List[int]], dst: int,
         s: int, idx: List[int], path: List[int]):
    """
    visited : tuple([state, idxtuple])
    """

    path = [s] + path

    idxp = deepcopy(idx)
    idxp[s] += 1

    if s == dst:
        all_path.append(path)

    for i in range(idxp[s], len(Pprime[s])):
        idxpp = deepcopy(idxp)
        idxpp[s] = i
        idxpptp = tuple([s] + idxpp)
        if (visited is None) or (idxpptp not in visited):
            if visited is not None:
                visited.add(idxpptp)
            rec3(visited, all_path, Pprime, dst,
                 Pprime[s][idxpp[s]], idxpp, path)


def rec4(dir: bool,
         all_path: List[List[int]],
         Pprime: List[List[int]], dst: int,
         s: int, idx: List[int], path: List[int]):
    """
    Backward Exploration. This function generates all the possible
    *usefull* paths from the current state \a s to the "global"
    destination \a dst.

    The arg \a dir determines whether earlier (in the sense of predecessor
    exploration) or older paths are explored first.
    This does not influence correctness.
    \a all_path simply serves to stores all the results
    \a Pprime is the list of predecessors for each state.
    \a idx: List that stores for each state which was the
    earliest predecessor that has already been explored for this state.
    To avoid reexploring loops already seen and to use loops / predecessors
    in the correct chronological order, we only recurse on predecessors
    that are older than the current idx[s].
    \a path is the already preassembled path for "how we got to s"
    """

    path = [s] + path # Creates independent instance

    if s == dst:
        all_path.append(path) # We arrived

    # Recurse on predecessor #i
    def rr(i: int) -> None:
        idxp = deepcopy(idx)
        idxp[s] = i
        rec4(dir, all_path, Pprime, dst,
             Pprime[s][i], idxp, path)

    # Recursion
    if dir: # Oldest first
        for i in range(0, idx[s]):
            rr(i)
    else: # Newest first
        for i in range(idx[s] - 1, -1, -1):
            rr(i)


@dataclass
class path_segment:
    """
    We do not need nested loops in the sense for energy.
    Either
    the other loop is energy optimal,
    in this case the nested loop might only have to be taken as
    a prefix : B0, A*, B1, (B0,B1)* which will be generated by
    the backwards search
    OR
    the inner loop energy optimal, in which case the other
    loop does not need to be looped at all
    B0, A*, B1 which will be generated by the backwards search

    Therefore, every path can be decomposed into (several) prefix + simple loop
    parts which can then be recombined.

    Prefix and loop maybe empty.
    If both are non-empty the last state of the prefix is the same as the
    first state of loop.
    If loop is non-empty then the first state is always equal to the
    last state.
    """
    prefix: List[int]
    loop: List[int]


def compress(path:List[int]) -> List[path_segment]:
    """
    Compress a path into a list of equivalent path_segments
    The algorithm will always decompose upon the first loop found.
    """
    idx = 0
    N = len(path)

    curr_path_idx = dict()
    c_path = []

    res = []

    while idx < N:
        s = path[idx]
        if s in curr_path_idx:
            # We found a loop
            # append current for completeness
            c_path.append(s)

            sidx1 = curr_path_idx[s]
            ps = path_segment(c_path[0:sidx1+1] if sidx1 != 0 else [],
                              c_path[sidx1:])
            res.append(ps)
            # Reset
            curr_path_idx = dict()
            c_path = []
        else:
            curr_path_idx[s] = len(c_path)
            c_path.append(s)
            idx += 1

    # Check if c_path is empty, if not it is a pure prefix (well its a post-fix
    # but here it is represented by a path_segment that has an empty loop
    # -> so it has only a prefix)
    if c_path:
        res.append(path_segment(c_path, []))

    return res


@dataclass
class energy:
    """
    Class representing an energy level.
    Can be used in generalized BF in forward propagation
    """
    e : int

    def __str__(self):
        return f"(e={self.e}"

    def __repr__(self, other):
        return self.__str__()

    @staticmethod
    def get_neutral_plus():
        return energy(0)

    @staticmethod
    def get_neutral_times():
        from math import inf
        return energy(-inf)

    def prop(self, w: "weight like") -> Tuple[bool, "energy"]:
        """
        returns a new energy when propagated along an edge
        of weight w.
        Implicitly, energies are always forward propagated
        Returns false if the propagation fails
        """
        en = energy(min(self.e + w, wup))
        return en.e >= 0, en

    def o_plus(self, w: "weight like") -> "energy":
        """
        This corresponds to propagation,
        however safeguarded against failure
        """
        succ, eprime = self.prop(w)
        if succ:
            return eprime
        else:
            return self.get_neutral_times()

    def o_times(self, other: "energy") -> "energy":
        """
        For energy propagation, this corresponds to the max operation
        """
        return energy(max(self.e, other.e))

    def __le__(self, rhs: "energy") -> bool:
        """
        For comparison, energies behave like ints
        """
        return self.e <= rhs.e

    def __lt__(self, rhs: "energy") -> bool:
        return self.e < rhs.e

    def __eq__(self, other: "energy") -> bool:
        return self.e == other.e

    def __deepcopy__(self, memodict={}):
        return energy(self.e)

    def __copy__(self):
        return self.__deepcopy__()

def check_energy_feas(cpath: List[path_segment], ic: energy,
                      wup: int) -> bool:
    """
    Check if the given (compressed) path is energy feasible in a
    strong sense, that is we can loop through it infinitely often,
    from the given initial credit \a ic with the weak upper bound
    \a wup.
    For convenience, we assume that the last state is the same
    the initial state (For instance in example 1, 7 is an alias for 0)

    To check this we need to propagate energy at most two times.
    Propagate once: If the energy is higher or equal to ic -> ok
    else:
    Second propagation from new initial energy. Path is unambiguous,
    so the second iteration makes a proper distinction between
    accepted and rejected cycles.
    """

    # Check up
    for ps in cpath:
        if ps.prefix and ps.loop:
            assert ps.prefix[-1] == ps.loop[0]
        if ps.loop:
            assert len(ps.loop) >= 2
            assert ps.loop[-1] == ps.loop[0]

    def prop_along(e: energy, p: List[int]):
        for seg in zip(p[:-1], p[1:]):
            succ, e = e.prop(W[seg])
            if not succ:
                return e
        return e

    e = energy(ic)

    for ps in cpath:
        # Prefix to next loop, auto skipped if empty
        e = prop_along(e, ps.prefix)
        if e.e < 0:
            return False
        # Loop part
        # Has loop?
        if ps.loop:
            estart = copy(e)
            # Loop once
            e = prop_along(e, ps.loop)
            if e.e < 0:
                return False
            if e <= estart:
                # Discard neutral or negativ loops.
                # In the set of paths to examine there will be the
                # same path just without the loop -> Avoid additional work
                return False
            # Pump the loop
            # This can be done by propagating twice from wup
            e.e = wup
            e = prop_along(e, ps.loop)
            e = prop_along(e, ps.loop)

    if e.e >= ic:
        return True
    else:
        return check_energy_feas(cpath, e, wup)


### Optimisation over wup and ic

@dataclass
class bounds:
    ic: int  # Minimal initial credit
    wup: int  # Minimal wup necessary

    def __str__(self):
        return f"(ic={self.ic}, wup={self})"

    def __repr__(self, other):
        return self.__str__()

    @staticmethod
    def get_min_element():
        return bounds(0, 0)

    @staticmethod
    def get_max_element():
        from math import inf
        return bounds(inf, inf)

    def prop(self, w: "weight like") -> Tuple[bool, "bounds"]:
        """
        Update the structure when propagated along an edge
        of weight w.
        Implicitly, bounds aare always backpropagated
        Here, propagation has no limits, so the boolean returned is
        always True
        """
        bprime = bounds(max(self.ic - w, 0), self.wup)
        bprime.wup = max(bprime.ic, bprime.wup)
        return True, bprime

    # Can not define o_plus and o_times on bounds

    def __le__(self, rhs: "bounds") -> bool:
        """
        An instance of bounds is considered worse (-> larger)
        if it is worse (larger) for both items
        """
        return self.ic <= rhs.ic and self.wup <= rhs.wup


class pareto_front(object):
    def __init__(self, element_type):
        self.elements = []
        self.element_type = element_type

    def __str__(self):
        return str(self.elements)

    def __repr__(self, other):
        return self.__str__()

    def __eq__(self, other: "pareto_front"):
        return (len(self.elements) == len(other.elements)) \
               and (set(self.elements) == set(other.elements))

    def get_neutral_plus(self) -> "pareto_front":
        """
        Get the pareto_front that is neutral with
        respect to the plus operation
        \note this is somewhat ambiguous
        """
        pr = pareto_front(self.element_type)
        pr.add(self.element_type.get_min_element())
        return pr

    def get_neutral_times(self) -> "pareto_front":
        """
        Get the pareto_front that is neutral with
        respect to the times operation
        \note this is somewhat ambiguous
        """
        pr = pareto_front(self.element_type)
        pr.add(self.element_type.get_max_element())
        return pr

    def o_plus(self, w: "weight like") -> "pareto_front":
        """
        Propagate each element of the pareto front along a weight w
        """
        newp = pareto_front(self.element_type)

        for elem in self.elements:
            newp.add(elem.prop(w))

        return newp

    def o_times(self, other: "pareto_front") -> "pareto_front":
        """
        Fuse the two fronts into one
        """
        newp = pareto_front(self.element_type)
        for op in [self, other]:
            for elem in op.elements:
                newp.add(elem)
        return newp

    def add(self, elem: "Element type") -> bool:
        """
        Adds an element to the front if it is pareto optimal.
        If so existing elements are possibly discarded.
        Returns True iff the element was inserted.
        """
        if any(map(lambda x: x <= elem, self.elements)):
            return False

        # Discard all that are larger in the current set
        for idx in range(len(self.elements)-1, -1, -1):
            if elem <= self.elements[idx]:
                # Discard this element
                elem[idx] = elem[-1] # elements are unordered -> erase by last
                self.elements.pop() # Discard last

    def __deepcopy__(self, memodict={}):
        newp = pareto_front(self.element_type)
        for elem in self.elements:
            newp.elements.append(deepcopy(elem, memodict))
        return newp


def compute_minimal_bounds(cpath: List[path_segment]) -> bounds :
    """
    Compute the minimal bounds necessary to traverse the given graph
    """

    def back_prop_along(p: List[int], b: bounds) -> bounds:
        """
        back-propagate a bound along a path
        """

        for seg in zip(p[-2::-1], p[:0:-1]):
            succ, b = b.prop(W[seg])
            assert succ

        return b

    def try_pump_cycle(p: List[int], b: bounds) -> bounds:
        """
        Pump a positive loop -> This does not change the
        needed wup, but does affect the ic
        """

        # 1 back prop to get correct wup
        bic = b.ic
        b = back_prop_along(p, b)
        # Check if the loop was positive
        # AND actually needs to be pumped
        if b.ic < bic:
            # 2 "inverse" pump initial energy
            # Set the initial energy to zero
            # is corrected via the two calls
            b.ic = 0
            b = back_prop_along(p, b)
            b = back_prop_along(p, b)
        else:
            dprint("Avoiding unnecessary or neutral loop")

        return b

    rb = bounds.get_min_element()

    for pe in reversed(cpath):
        # everything needs to be reversed
        # So loop before cycle
        if pe.loop:
            rb = try_pump_cycle(pe.loop, rb)
        rb = back_prop_along(pe.prefix, rb)

    return rb


class gen_ext_bf:
    """
    Extended generic Bellman-Ford
    """
    def __init__(G:"graph", W: "weight",
                 src: int, energy_type: "Type used as an energy",
                 init: "energy_like or None"):
        self.G = G
        self.W = W
        self.src = src
        self.e_type = energy_type
        self.init = init

        self.E = None  # Current energy-like
        self.P = None  # Current predecessor
        self.Ep = None  # Next energy-like
        self.Pp = None  # Next predecessor

    def run(self):

        self.E = [self.e_type.get_neutral_times() for _ in range(len(self.G))]
        self.P = [None]*len(self.E)
        self.Ep = [None]*len(self.E)
        self.Pp = [None]*len(self.E)

        while self.E != self.Ep:
            # New to old
            self.E = deepcopy(self.Ep)
            self.P = deepcopy(self.Pp)

            # Run one round of gen BF
            # Note: always runs on primed vars
            self.run_gen_bf()
            # Try to pump if changed
            if self.E != self.Ep:
                self.try_pump_all()







if __name__ == '__main__':

    Pprime = []
    for l in P:
        if not l:
            Pprime.append([])
        elif len(l) == 1:
            Pprime.append([l[0]])
        else:
            thisl = [l[0]]
            for x in l[1:]:
                thisl.append(x)
                thisl.append(l[0])
            Pprime.append(thisl)

    print(Pprime)

    allPath3_NOOPT = []
    rec3(None, allPath3_NOOPT, Pprime, 0, len(Pprime) - 1, [-1] * len(Pprime), [])

    allPath3_OPT = []
    rec3(set(), allPath3_OPT, Pprime, 0, len(Pprime) - 1, [-1] * len(Pprime), [])

    allPath4_F = []
    rec4(True, allPath4_F, P, 0, len(P) - 1, [len(l) for l in P], [])

    allPath4_B = []
    rec4(False, allPath4_B, P, 0, len(P) - 1, [len(l) for l in P], [])

    up = []

    for i, ap in enumerate([allPath3_NOOPT, allPath3_OPT,
                            allPath4_F, allPath4_B]):
        unique_paths = [tuple(p) for p in ap]
        unique_paths = set(unique_paths)

        print("Loop nbr: ", i)
        print(len(ap))
        for x in ap:
            print(x)
        print(len(unique_paths))
        for x in unique_paths:
            print(x)
        up.append(deepcopy(unique_paths))
    print("Comparing opt and noopt")
    for p in up[0]:
        if p not in up[1]:
            print(p, " was not found in opt")

    allCPath4_F = [compress(p) for p in allPath4_F]
    allCPath4_B = [compress(p) for p in allPath4_B]

    allCPath4_F_R = [(check_energy_feas(cp, 0, wup), cp) for cp in allCPath4_F]
    allCPath4_B_R = [(check_energy_feas(cp, 0, wup), cp) for cp in allCPath4_B]

    print("Results with energy; Forward")
    for x in allCPath4_F_R:
        print(x)
    print("Results with energy; Backward")
    for x in allCPath4_B_R:
        print(x)

    # Compute for each (open) path the minimal requirements
    allCPath4_F_B = [(compute_minimal_bounds(cp), cp) for cp in allCPath4_F]
    allCPath4_B_B = [(compute_minimal_bounds(cp), cp) for cp in allCPath4_B]

    print("Minimal bounds; Forward")
    for x in allCPath4_F_B:
        print(x)
    print("Minimal bounds; Backward")
    for x in allCPath4_B_B:
        print(x)


