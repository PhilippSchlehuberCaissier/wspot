import itertools
import dataclasses
import sys
from dataclasses import dataclass
from typing import List, Set, Tuple, Union, Type, ClassVar, Callable
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

# G1 : Rabbit of death example
# Each loop needs several sub-loops
# States can have multiple necessary predecessors
# and some states need to visit the same predecessor multiple times
P1 = [[],  # 0
      [0, 2, 5, 2],  # 1
      [1, 3, 1, 4],  # 2
      [2, 3],  # 3
      [2, 4],  # 4
      [1, 6, 1],  # 5
      [5, 6],  # 6
      [1]]  # 7

# Corresponding graph
# adjacency list
G1 = [[1],  # 0
      [2, 5, 7],  # 1
      [1, 3, 4],  # 2
      [2, 3],  # 3
      [2, 4],  # 4
      [1, 6],  # 5
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


# G1p

# G2 The "base" example for which our old algorithm failed
# Each time you take the "main" loop you loose a bit of initial energy
G2 = [[1],  # 0
      [2, 4, 5, 6, 8],  # 1 Succ (0,1,2,3) must be neg
      [3],  # 2
      [0],  # 3 3 -> 0 accepting
      [2, 4],  # 4 State with pos selfloop
      [2, 5],  # 5 State with pos selfloop
      [7],  # 6 State with pos loop
      [2, 6],  # 7 States (6, 7) form a pos loop
      [9],  # 8 State with linear successor
      [2],  # 9: 1 -> 8 -> 9 -> 2
      ]

special_src2 = 3
special_dst2 = 0
WUP2 = 1000
W2 = dict()
for (s, d, w) in [(0, 1, 0),  # 0
                  (1, 2, -1), (1, 4, -10), (1, 5, -9), (1, 6, -5), (1, 8, +10),  # 1
                  (2, 3, 0),  # 2,
                  (3, 0, 0),  # 3
                  (4, 2, -1), (4, 4, 1),  # 4
                  (5, 2, -2), (5, 5, 1),  # 5
                  (6, 7, -10),  # 6
                  (7, 2, -5), (7, 6, +11),  # 7
                  (8, 9, +10),  # 8
                  (9, 2, -21),  # 9
                  ]:
    W2[(s, d)] = w

P2 = []



#defining which prob to use
probnr = 2

P = eval(f"P{probnr}")
W = eval(f"W{probnr}")
wup = eval(f"WUP{probnr}")
G = eval(f"G{probnr}")
GT = transpose(G)
# Weights for transposed graph
WT = dict()
for (s, d), w in W.items():
    WT[(d,s)] = w


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
            # Avoid generating a prefix with only a single state
            # Upon the next iteration in the case we have just treated the last element
            if idx == (N - 1):
                idx += 1
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

@dataclass(frozen=True)
class has_plus(object):
    """
    Object that can be propagated along a transition
    """
    @staticmethod
    def get_neutral_plus() -> "has_plus":
        """
        Neutral element with respect to plus
        """
        raise NotImplementedError()

    def o_plus(self, src:"state", dst:"state"):
        """
        Plus operation needs to propagate an instance along a transition
        """
        raise NotImplementedError()

    def __eq__(self, other: "has_plus") -> bool:
        """
        Objects need to be comparable for equality
        """
        raise NotImplementedError()


@dataclass(frozen=True)
class energy_like(has_plus):
    """
    Objects that can be used within extended bellmann ford
    """
    @staticmethod
    def get_run_multiplier() -> int:
        return 1
    @staticmethod
    def get_neutral_times() -> "energy_like":
        """
        Needs to return the neutral element with respect to times
        """
        raise NotImplementedError()

    def o_times(self, other:"energy_like") -> "energy_like":
        """
        Times operation is called on two instances of energy_like
        and computes their "optimum"
        """
        raise NotImplementedError()

    def get_pump_value(self) -> "energy_like":
        """
        Initial value for pumping a positive loop
        """
        raise NotImplementedError()

    def __le__(self, other: "energy_like") -> bool:
        """
        Energy like elements also need to be at least partially ordered
        """
        raise NotImplementedError()



@dataclass(frozen=True, order=True)
class energy(energy_like):
    """
    Class representing an energy level.
    Can be used in generalized BF in forward propagation
    We use energies as immutable entities and always want to generate new instances
    For comparison energy behaves like an int
    """
    e : int

    def __str__(self):
        return f"(e={self.e})"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def get_neutral_plus():
        return energy(0)

    @staticmethod
    def get_neutral_times():
        from math import inf
        return energy(-inf)

    def prop(self, src: "state", dst: "state") -> Tuple[bool, "energy"]:
        """
        returns a new energy when propagated along an edge
        from src to dst.
        Implicitly, energies are always forward propagated
        Returns false if the propagation fails
        """
        w = W[(src, dst)]  # For energy propagation we only need the weight
        en = energy(min(self.e + w, wup))
        return en.e >= 0, en

    def o_plus(self, src: "state", dst: "state") -> "energy":
        """
        This corresponds to propagation,
        however safeguarded against failure
        """
        succ, eprime = self.prop(src, dst)
        if succ:
            return eprime
        else:
            return self.get_neutral_times()

    def o_times(self, other: "energy") -> "energy":
        """
        For energy propagation, this corresponds to the max operation
        """
        return energy(max(self.e, other.e))

    def get_pump_value(self) -> "energy":
        """
        Return the energy to its "before pumping value".
        For an actual energy this is simply the WUP,
        for other energy-like classes this can be more involved
        """
        return energy(wup)

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
        e_fail = e.get_neutral_times()
        for (src, dst) in zip(p[:-1], p[1:]):
            e = e.o_plus(src, dst)
            if e == e_fail:
                return e
        return e

    e = deepcopy(ic)

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
            e = e.get_pump_value()
            e = prop_along(e, ps.loop)
            e = prop_along(e, ps.loop)

    if e >= ic:
        return True
    else:
        return check_energy_feas(cpath, e, wup)


### Optimisation over wup and ic

@dataclass(kw_only=True, frozen=True, repr=False)
class bounds(has_plus):
    """
    Note bounds are always propagated on the transposed graph.
    It can not behave like an energy, as the times operation
    does not have the same return type.
    (The comparison is only a partial order)
    """
    ic: int  # Minimal initial credit
    wup: int  # Minimal wup necessary

    def __str__(self):
        return f"(ic={self.ic}, wup={self.wup})"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def get_min_element():
        return bounds(ic=0, wup=0)

    @staticmethod
    def get_max_element():
        from math import inf
        return bounds(ic=inf, wup=inf)

    @staticmethod
    def get_neutral_plus() -> "bounds":
        return bounds.get_min_element()

    @staticmethod
    def get_neutral_times() -> "bounds":
        return bounds.get_max_element()

    def get_pump_value(self) -> "bounds":
        """
        Pumping a bound along an energy positiv cycle
        only reduces the initial credit, the wup remains unchanged
        """
        return bounds(ic=0, wup=self.wup)

    def prop(self, src: "state", dst: "state") -> Tuple[bool, "bounds"]:
        """
        Update the structure when propagated along an edge
        from src to dst.
        Implicitly, bounds aare always backpropagated
        Here, propagation has no limits, so the boolean returned is
        always True
        """
        w = WT[(src, dst)]
        new_ic = max(self.ic - w, 0)
        new_wup = max(self.wup, new_ic)
        return True, bounds(ic=new_ic, wup=new_wup)

    def o_plus(self, src: "state", dst: "state") -> "bounds":
        succ, bprime = self.prop(src, dst)
        if succ:
            return bprime
        else:
            return self.get_max_element()

    # Can not define o_plus and o_times on bounds

    def __le__(self, rhs: "bounds") -> bool:
        """
        An instance of bounds is considered worse (-> larger)
        if it is worse (larger) for both items
        """
        return self.ic <= rhs.ic and self.wup <= rhs.wup

@dataclass(kw_only=True, frozen=True, repr=False)
class accepting_bounds(bounds):
    esrc : ClassVar[int]  # Source vertex of special trans
    edst : ClassVar[int]  # Destination vertex of special trans
    crit_ic : ClassVar[int]  # critical initial energy.
                             # If below, a positive loop must have been encountered

    loop_counter : int = 0 # Keeps track of how often the loop was visited, bounded in [0, 3]

    @staticmethod
    def get_run_multiplier() -> int:
        return 4

    def __str__(self):
        return f"(ic={self.ic}, wup={self.wup}, lc={self.loop_counter})"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def get_min_element():
        """
        Beware, this can not be used as an initial value
        """
        b_min = bounds.get_min_element()
        return accepting_bounds(ic=b_min.ic, wup=b_min.wup, loop_counter=3)

    @staticmethod
    def get_max_element():
        """
        Worst element -> Least repetitions
        """
        b_max = bounds.get_max_element()
        return accepting_bounds(ic=b_max.ic, wup=b_max.wup, loop_counter=0)

    @staticmethod
    def get_neutral_plus() -> "accepting_bounds":
        return accepting_bounds.get_min_element()

    @staticmethod
    def get_neutral_times() -> "accepting_bounds":
        return accepting_bounds.get_max_element()

    def get_pump_value(self) -> "accepting_bounds":
        """
        Pumping a bound along an energy positiv cycle
        only reduces the initial credit, the wup remains unchanged.
        If the special transition is on the loop, it will be taken into account
        automatically. todo Possible fail: Overflowing the number of iterations till fix-point?
        """
        bpump = bounds.get_pump_value(self)
        return accepting_bounds(ic=bpump.ic, wup=bpump.wup, loop_counter=self.loop_counter)

    def prop(self, src: "state", dst: "state") -> Tuple[bool, "accepting_bounds"]:
        from math import inf
        # There is a special case for propagation:
        # going through the transition from esrc to edst
        if (src == accepting_bounds.esrc) and (dst == accepting_bounds.edst):
            # If it is the first time and the energy is below critical:
            # Perform the reset
            bprop = None
            if ((self.ic < accepting_bounds.crit_ic) and
                (self.wup >= accepting_bounds.crit_ic)):
                succ, bprop = bounds.prop(bounds(ic=0, wup=0), src, dst)
                assert succ
                if (bprop.ic != inf):
                    return True, accepting_bounds(ic=bprop.ic, wup=bprop.wup, loop_counter=1)
                else:
                    return True, self.get_neutral_times()
            else:
                succ, bprop = bounds.prop(self, src, dst)
                assert succ
                if (bprop.ic != inf):
                    return True, accepting_bounds(ic=bprop.ic, wup=bprop.wup,
                                                  loop_counter=min(self.loop_counter+1, 3))
                else:
                    return True, self.get_neutral_times()
        else:
            # Usual propagation
            succ, bprop = bounds.prop(self, src, dst)

            if not succ:
                return False, accepting_bounds.get_max_element()

            return True, accepting_bounds(ic=bprop.ic, wup=bprop.wup, loop_counter=self.loop_counter)

    def o_plus(self, src: "state", dst: "state") -> "accepting_bounds":
        succ, bprop = self.prop(src, dst)
        if succ:
            return bprop
        else:
            return accepting_bounds.get_max_element()

    def __le__(self, rhs: "accepting_bounds") -> bool:
        """
        An instance is considered worse (-> larger)
        if it is worse for the bounds part AND has an <= loop_counter
        """
        assert (0 <= self.loop_counter) and (self.loop_counter <= 3)
        assert (0 <= rhs.loop_counter) and (rhs.loop_counter <= 3)
        return rhs.loop_counter <= self.loop_counter and bounds.__le__(self, rhs)

    def __eq__(self, other: "accepting_bounds") -> bool:
        return ((self.loop_counter == other.loop_counter)
                and (bounds.__eq__(self, other)))



class pareto_front(energy_like):
    """
    Holds a pareto front of partially ordered elements of element_type.
    To be valid, element_type needs to be propagated
    from a source state to a destination state
    """

    element_type : ClassVar["element_type"] = None

    def get_run_multiplier(self) -> int:
        return self.element_type.get_run_multiplier()

    def __init__(self, element_type = None):
        self.elements = []
        if element_type is not None:
            pareto_front.element_type = element_type

    def __str__(self):
        return str(self.elements)

    def __repr__(self):
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
        pr = pareto_front()
        pr.add(pareto_front.element_type.get_neutral_plus())
        return pr

    def get_neutral_times(self) -> "pareto_front":
        """
        Get the pareto_front that is neutral with
        respect to the times operation
        \note this is somewhat ambiguous
        """
        pr = pareto_front()
        pr.add(pareto_front.element_type.get_neutral_times())
        return pr

    def o_plus(self, src: "state", dst: "state") -> "pareto_front":
        """
        Propagate each element of the pareto front along a weight w
        """
        newp = pareto_front()

        for elem in self.elements:
            newp.add(elem.o_plus(src, dst))

        return newp

    def o_times(self, other: "pareto_front") -> "pareto_front":
        """
        Fuse the two fronts into one
        """
        newp = pareto_front()
        for op in [self, other]:
            for elem in op.elements:
                newp.add(elem)
        return newp

    def get_pump_value(self) -> "pareto_front":
        """
        Pumping a pareto front pumps each element.
        \note, for bounds for instance, this collapses the front into
        unique element.
        """
        newp = pareto_front()

        for elem in self.elements:
            newp.add(elem.get_pump_value())

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
                self.elements[idx] = self.elements[-1]  # elements are unordered -> erase by last
                self.elements.pop()  # Discard last

        # Finally add the element itself
        self.elements.append(elem)

        return True

    def __deepcopy__(self, memodict={}):
        newp = pareto_front()
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
        # The path is expected to be in "normal" direction
        # We need to "transpose" each transition
        b_fail = b.get_neutral_times()
        for src, dst in zip(p[:-1], p[1:]):
            b = b.o_plus(src, dst)
            assert b != b_fail

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
            b = b.get_pump_value()
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
            rb = try_pump_cycle(list(reversed(pe.loop)), rb)
        rb = back_prop_along(list(reversed(pe.prefix)), rb)

    return rb

@dataclass(kw_only=True, repr=False)
class pre_type(object):
    """
    Base class used to store optimal predecessors
    -1 is a special predecessor indicating that new_pred has
    never been called on this instance
    """

    def new_pred(self, src: "state") -> None:
        """
        Register a new optimal predeceessor with the structure
        """
        raise NotImplementedError

    def __str__(self):
        return str(self.all_pred())

    def __repr__(self):
        return self.__str__()

    def last_pred(self) -> "state":
        """
        Lets you access the last one added
        """
        raise NotImplementedError

    def all_pred(self) -> List["state"]:
        """
        Returns all predecessors encountered as a list
        """
        raise NotImplementedError

@dataclass(kw_only=True, repr=False)
class last_predec(pre_type):
    """
    Derived class that only stores the very last predecessor encountered
    """
    pred : int = -1

    def new_pred(self, src: "state") -> "state":
        self.pred = src

    def last_pred(self) -> "state":
        return self.pred

    def all_pred(self) -> List["state"]:
        return [self.pred]

@dataclass(kw_only=True, repr=False)
class all_predec(pre_type):
    """
    Derived class storing all predecessors avoiding stuttering
    """
    pred : list[int] = dataclasses.field(default_factory=list)

    def new_pred(self, src: "state") -> "state":
        if self.last_pred() != src:
            self.pred.append(src)

    def last_pred(self) -> "state":
        if self.pred:
            return self.pred[-1]
        else:
            return -1

    def all_pred(self) -> List["state"]:
        return self.pred

def no_op(*args, **kwargs):
    pass

class gen_ext_bf:
    """
    Extended generic Bellman-Ford
    """
    def __init__(self, G:"graph",
                 src: int, energy_type: "Type used as an energy",
                 init: "energy_like or None",
                 predec_type: Type[pre_type],
                 post_proc: Callable = no_op):
        self.G = G
        self.src = src
        self.e_type = energy_type
        self.init = init
        self.predec_type = predec_type
        self.post_proc = post_proc  # Call after each iteration

        self.E = None  # Current energy-like
        self.P = None  # List of all predecessors
        self.Ep = None  # Next energy-like

    def run_gen_bf(self):
        """
        Run one round of the generalised Bellmann-Ford
        todo: unoptimized, brute force version
        """

        for _ in range(len(self.G)*self.e_type.get_run_multiplier()): # todo hack
            for src, succ_list in enumerate(self.G):
                for dst in succ_list:
                    Eprime = self.Ep[src].o_plus(src, dst)
                    Eprime = Eprime.o_times(self.Ep[dst])
                    if Eprime != self.Ep[dst]:
                        self.Ep[dst] = Eprime
                        # todo do we really only need to append if
                        # the predecessor changed?
                        self.P[dst].new_pred(src)

        return None  # Done

    def pump_loop(self, s: int):
        """
        Pump the loop associated to some state
        """
        path = [s]
        cp = None
        self.onLoop[s] = 1

        # Find the loop
        while True:
            pred = self.P[path[-1]].last_pred()
            assert pred is not None  # Only loops should be found here
            path.append(pred)

            if self.onLoop[pred] == 1:
                # Returned back to some state
                # The path is generated in a reversed manner from the predecessors
                # -> Reverse it
                path.reverse()
                # Compress for easier handling
                # Must correspond to one or two path_elements,
                # the first only having a loop, the second only having a prefix
                cp = compress(path)
                assert len(cp) and (len(cp) <= 2)
                assert len(cp[0].loop) and (cp[0].loop[0] == cp[0].loop[-1])
                assert (len(cp) == 1) or (cp[0].loop[0] == cp[1].prefix[0])
                break
            elif self.onLoop[pred] == 2:
                # We found a new prefix to an existing loop
                path.reverse()
                cp = compress(path)
                assert len(cp) == 1
                assert len(cp[0].loop) == 0
                # Prepend an empty loop for semantics
                cp = [path_segment([], [])] + cp
                break

            self.onLoop[pred] = 1

        assert cp is not None

        # Pump the loop
        loop = cp[0].loop
        # todo: Do we need to (pre-)propagate all states
        # once more; I suppose not, everything is properly in
        # place after BF I suppose
        if loop:
            # Loop might be empty if prefix only
            counter = 0
            self.Ep[loop[0]] = self.Ep[loop[0]].get_pump_value()
            reached_fixed = False
            while not reached_fixed:
                counter += 1
                # Propagate along the loop
                for src, dst in zip(loop[:-1], loop[1:]):
                    Eprimedst = self.Ep[src].o_plus(src, dst)
                    if Eprimedst == self.Ep[dst]:
                        reached_fixed = True
                        break
                    self.Ep[dst] = Eprimedst

        # Propagate along the prefix if existent
        if len(cp) == 2:
            prefix = cp[1].prefix
            for (src, dst) in zip(prefix[:-1], prefix[1:]):
                Eprimedst = self.Ep[src].o_plus(src, dst)
                self.Ep[dst] = self.Ep[dst].o_times(Eprimedst)

        # Mark all the states as already treated
        for ps in cp:
            for pp in [ps.loop, ps.prefix]:
                for s in pp:
                    self.onLoop[s] = 2

        # Done
        return None

    def try_pump_all(self):
        """
        Pump currently existing loops
        """

        self.onLoop = [0]*len(self.G)  # 0 -> untreated; 1 -> on current path; 2 -> on already treated path

        for s in range(len(self.G)):
            if self.onLoop[s] != 0:
                # Already treated
                continue

            pred_s = self.P[s].last_pred()

            if pred_s == -1:
                # Has no predescessor (yet)
                # Is currently (energy) unreachable
                continue

            # Compute the propagation
            esp = self.Ep[pred_s].o_plus(pred_s, s)
            # Check if this changes the set of dst
            esp = self.Ep[s].o_times(esp)

            if esp != self.Ep[s]:
                # We have actually found a loop
                self.pump_loop(s)

    def run(self):

        self.E = [self.e_type.get_neutral_times() for _ in range(len(self.G))]
        self.P = [self.predec_type() for _ in range(len(self.G))]
        self.Ep = [self.e_type.get_neutral_times() for _ in range(len(self.G))]
        self.Ep[self.src] = self.init

        while self.E != self.Ep:
            # New to old
            self.E = deepcopy(self.Ep)

            # Run one round of gen BF
            # Note: always runs on primed vars
            self.run_gen_bf()
            # Try to pump if changed
            if self.E != self.Ep:
                self.try_pump_all()
            if self.post_proc:
                self.post_proc(self)

def test_trace_ext(dir: bool):


    allPath4 = []
    rec4(dir, allPath4, P, 0, len(P) - 1, [len(l) for l in P], [])

    allCPath4 = [compress(p) for p in allPath4]

    allCPath4_R = [(check_energy_feas(cp, energy(0), wup), cp) for cp in allCPath4]

    print("Results with energy; Forward")
    for x in allCPath4_R:
        print(x)

    allCPath4_B = [(compute_minimal_bounds(cp), cp) for cp in allCPath4]

    print("Minimal bounds; Forward")
    for x in allCPath4_B:
        print(x)

def test_BF_config(G, src, e_type, pred_type, saveP = False):

    prob = gen_ext_bf(G, src, e_type, e_type.get_neutral_plus(), pred_type)
    prob.run()
    print(f"Solution for etype: {str(e_type)} and ptype: {str(pred_type)}")
    print("Energy solution is")
    for i, e in enumerate(prob.Ep):
        print(f"{i}: {e}")
    print("Predecessor solution is:")
    for i, p in enumerate(prob.P):
        print(f"{i}: {p}")
    if saveP == True:
        print("Saving predecessor solution")
        for i, p in enumerate(prob.P):
            P[i] = p.all_pred()

def test_pareto():

    p = pareto_front(bounds)

    p.add(bounds(ic=3, wup=3))
    assert len(p.elements) == 1
    p.add(bounds(ic=1, wup=4))
    assert len(p.elements) == 2
    p.add(bounds(ic=4, wup=1))
    assert len(p.elements) == 3
    p.add(bounds(ic=2, wup=2))
    assert len(p.elements) == 3
    p.add(bounds(ic=1, wup=1))
    assert len(p.elements) == 1

    p2 = pareto_front(bounds)
    p2.add(bounds(ic=1, wup=1))
    assert p == p2


def test_accepting():
    accepting_bounds.esrc = special_dst2
    accepting_bounds.edst = special_src2
    accepting_bounds.crit_ic = int(1e6)
    init_b = accepting_bounds(ic=accepting_bounds.crit_ic*2,
                              wup=accepting_bounds.crit_ic*2,
                              loop_counter=0)
    init = pareto_front(accepting_bounds)
    init.add(init_b)

    def print_acc(a_BF: gen_ext_bf) -> None:
        """
        Postproc is called after each iteration,
        prints whenever it finds an accepting energy level
        """

        # Steps one, look for neutral accepting loops
        # These have the same initial energy, a possibly higher wup and
        # a loop count of 1
        for e in a_BF.Ep[accepting_bounds.edst].elements:
            if (e.ic == accepting_bounds.crit_ic * 2
                and e.wup >= accepting_bounds.crit_ic * 2
                and e.loop_counter >= 1):
                print(f"Print found proof for neutral accepting loop: {e}")

        # Check for accepting positive loops
        # They have a loop counter of 3 and there exists
        # an element in the last iteration with loop counter of 2
        # and the same ic and wup
        for e in a_BF.Ep[accepting_bounds.edst].elements:
            if (e.ic < accepting_bounds.crit_ic
                and e.wup < accepting_bounds.crit_ic
                and e.loop_counter == 3):
                # Check if it also exists for loop_counter == 2
                e2 = accepting_bounds(ic=e.ic, wup=e.wup, loop_counter=2)
                try:
                    a_BF.E.index(e2)
                    print(f"Found a proof for positive accepting loop: {e}")
                except ValueError:
                    try:
                        a_BF.E.index(e)
                        print(f"Reappearing: {e}")
                    except ValueError:
                        print(f"This should not happen I suppose: {e}")

    prob = gen_ext_bf(GT, accepting_bounds.edst, pareto_front(accepting_bounds), init,
                      all_predec, print_acc)
    prob.run()

    print("Energy solution is")
    for i, e in enumerate(prob.Ep):
        print(f"{i}: {e}")
    print("Predecessor solution is:")
    for i, p in enumerate(prob.P):
        print(f"{i}: {p}")

if __name__ == '__main__':

    # Test the implementation of pareto
    test_pareto()

    test_accepting()

    sys.exit(0)

    # trace extraction 1
    # with bounds optimisation
    test_trace_ext(True)
    test_trace_ext(False)

    # Testing energy propagation
    test_BF_config(G, 0, energy, last_predec)
    test_BF_config(G, 0, energy, all_predec, True)

    # Retesting trace extraction
    # with bounds optimisation
    test_trace_ext(False)

    # Testing bounds optimisation as a generalised BF problem
    test_BF_config(GT, 7, pareto_front(bounds), all_predec, False)


    sys.exit(0)


def dump():

    # solve the standard energy problem


    # solve the standard energy problem with all predescessors
    prob = gen_ext_bf(G, 0, energy, energy(0), last_predec)
    prob.run()
    print("Base prob energy solution is")
    print(prob.Ep)
    print("Base prob predecessor solution is")
    print(prob.P)





def legacy_main():

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


