from typing import List, Tuple, Union, Callable
from copy import deepcopy
from array import array
from dataclasses import dataclass
import spot

from buechi import BuechiResult


@dataclass
class transition:
    g: spot.twa_graph  # Underlying graph
    n: int  # Edge number

    @property
    def src(self):
        return self.g.edge_storage(self.n).src

    @property
    def dst(self):
        return self.g.edge_storage(self.n).dst

    @property
    def w(self):
        return spot.get_weight(self.g, self.n)

    def __iter__(self):
        yield self.src
        yield self.w
        yield self.dst

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        return f"({self.src}, {self.w}, {self.dst})"

    def __deepcopy__(self, memodict={}):
        return transition(self.g, self.n)


@dataclass
class pathSegment:
    prefix: List[transition]  # Prefix leading to a cycle; possibly empty
    cycle: List[transition]  # Cycle of the path segment; possibly empty

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        res = ""
        if self.prefix:
            res = ", ".join(map(str, self.prefix))
        if self.cycle:
            res += "(" + ", ".join(map(str, self.cycle)) + ")^*"
        return res

    def __deepcopy__(self, memodict={}):
        pnew = [deepcopy(x) for x in self.prefix]
        cnew = [deepcopy(x) for x in self.cycle]
        return pathSegment(pnew, cnew)


def compressPath(path: List[transition]) -> List[pathSegment]:
    """
    Compress a given \a path that may contain cycles into a list of \a pathSegments
    Args:
        path: List of transitions

    Returns: Corresponding list of pathSegments
    """
    tc = []
    idx = 0
    N = len(path)

    def validate():
        for t1, t2 in zip(path[:-1], path[1:]):
            if t1.dst != t2.src:
                return False
        return True

    assert validate(), "Invalid path"

    while idx < N:
        subtrace = []
        srcIdx = dict()
        while idx < N:
            subtrace.append(path[idx])
            idx += 1
            try:
                cutIdx = srcIdx[subtrace[-1].dst]
                tc.append(pathSegment(subtrace[:cutIdx], subtrace[cutIdx:]))
                subtrace = []
                break
            except KeyError:
                pass
            srcIdx[subtrace[-1].src] = len(subtrace) - 1
    if subtrace:
        tc.append(pathSegment(subtrace, []))

    # Clean up step due to repetition of predecessors
    # If there is no prefix and the cycle is the same as
    # the one of the last segment -> delete it
    idx = 1
    while idx < len(tc):
        if (not tc[idx].prefix) and (tc[idx - 1].cycle == tc[idx].cycle):
            tc.pop(idx)
        else:
            idx += 1

    return tc


def propAlong(e: int, t: List[transition], wup: int) -> Tuple[bool, int]:
    """
    Propagates an energy along a given path.
    Args:
        e: starting energy
        t: the path
        wup: the considered weak upper bound

    Returns: (true, energy) after if t is feasible, else (false, -1)
    """

    for at in t:
        ep = min(e + at.w, wup)
        if ep < 0:
            return False, -1
        e = ep
    return True, e


def tryPumpLoop(e: int, t: List[transition], wup: int) -> Tuple[bool, int]:
    """

    Args:
        e: starting energy
        t: the cycle
        wup: the considered weak upper bound

    Returns: (true, maximally attainable energy) after if t is feasible and energy positive, else (false, -1)
    """

    def validate():
        return t[0].src == t[-1].dst

    assert validate(), "Invalid cycle"

    eInit = e
    succ, e = propAlong(e, t, wup)

    if (not succ) or (e <= eInit):
        return False, -1

    # "Pump"
    # Set to max
    # Correct by propagating twice
    e = wup
    succ, e = propAlong(e, t, wup)
    assert succ
    succ, e = propAlong(e, t, wup)
    assert succ, e

    return True, e


def forwardExploration(ic: int, eDst: int, wup: int, t: List[pathSegment],
                       implCyclCost: Union[int, None] = None) -> bool:
    """
    Compute whether at least eDst can be attained after traversing t

    Args:
        ic: Initial credit
        eDst: Minimal energy at destination
        wup: weak upper bound
        t: considered trace
        implCyclCost: Implicit cost for closing the cycle; existence of the transition is not verified

    Returns: True iff the energy after traversing the path is at least \a eDst
    """

    def propOnce(e: int):
        for ps in t:
            if ps.prefix:
                succ, e = propAlong(e, ps.prefix, wup)
                if not succ:
                    return False, -1
            if ps.cycle:
                succ, e = tryPumpLoop(e, ps.cycle, wup)
                if not succ:
                    return False, -1
        return True, e

    succ, e = propOnce(ic)
    if not succ:
        return False

    if e >= eDst:
        return True

    if implCyclCost is None:
        return False

    # Check if the trace corresponds to a energy-feasible cycle with the given implicit cost to
    # pass from the last to the first state
    ic2 = min(e + implCyclCost, wup)  # No energy at start state
    if ic2 < 0:
        return False
    succ, e = propOnce(ic2)
    if not succ:
        return False
    e = min(e + implCyclCost, wup)

    if e >= ic2:  # We loop back with equal or more energy
        return True

    # There is truly no hopy
    return False


def backwardsSearchImpl_(g: spot.twa_graph,
                         pred: List[List[int]],
                         gSrc: int,
                         forwardExp: Callable,
                         ci: List[int],
                         t: List[transition]) -> List[pathSegment]:
    """
    Recurses on optimal predecessors to find a path starting in \a gSrc.
    If such a path is found, then forwardExp will be called to test its viability.
    Args:
        g: The graph
        pred: The extended optimal predecessor list
        gSrc: The initial node of the trace
        forwardExp: A callable evaluating a possible trace
        ci: The current index table into the extended predecessor list
        t: The trace accumulated so far

    Returns: A list of pathSegments representing a valid trace. The list is empty if no such list exists.

    """

    # Check if the source state was attained and if so trigger forward exploration
    if t[-1].src == gSrc:
        tf = list(reversed(t))
        tf = compressPath(tf)
        # Validate via forward exploration
        if forwardExp(tf):
            return tf

    # Search recursively
    # todo implement this with a stack
    v = t[-1].src
    # Decide on exploration directions
    for i in range(ci[v] - 1, -1, -1):
        ciPrime = deepcopy(ci)
        tPrime = deepcopy(t)
        ciPrime[v] = i
        tPrime.append(transition(g, pred[v][i]))
        # recurse, if found return the viable trace
        tr = backwardsSearchImpl_(g, pred, gSrc, forwardExp, ciPrime, tPrime)
        if tr:
            return tr
    return []


def searchTrace(g: spot.twa_graph,
                pred: List[List[int]],
                gSrc: int, gDst: int,
                icSrc: int,
                eDst: int,
                wup: int,
                implCyclCost: Union[int, None] = None) -> List[pathSegment]:
    """
    Search for a trace amongst the optimal predecessors \a pred that arrives at \a gDst with at least \a eDst energy
    when starting in \a gSrc with at least \a icSrc energy
    Args:
        pred: Optimal predecessor list
        gSrc: Initial state of the trace
        gDst: Final state of the trace
        icSrc: Initial credit to start the run
        eDst: Minimal final energy
        wup: Weak upper bound of the trace
        implCyclCost: Implicit cost of closing a cycle. If None we seek for a "linear" path

    Returns: A viable trace as list of pastSegments; Empty if no trace was found

    """

    fforward_ = lambda t: forwardExploration(icSrc, eDst, wup, t, implCyclCost)

    # Initially all predecessors are allowed
    ci = array('q', [len(pp) for pp in pred])
    t = []
    v = gDst
    # All initial calls
    # Decide on exploration directions
    # todo factorise this
    for i in range(ci[v] - 1, -1, -1):
        ciPrime = deepcopy(ci)
        tPrime = deepcopy(t)
        ciPrime[v] = i
        tPrime.append(transition(g, pred[v][i]))
        # recurse, if found return the viable trace
        tr = backwardsSearchImpl_(g, pred, gSrc, fforward_, ciPrime, tPrime)
        if tr:
            return tr
    return []


def projectTrace_(br: BuechiResult, t: List[pathSegment]) -> List[pathSegment]:
    """
    Project a trace in br.gScc onto br.g
    Args:
        br: BuechiResult structure holding all the necessary information
        t: The trace to be projected

    Returns: The projected trace

    """

    revrename = {v: k for k, v in br.renameDict.items()}
    assert len(revrename) == len(br.renameDict), "Should be a isomorphism"

    def fProj(vScc: int) -> int:
        """
        Project a state in degeneralised gScc onto the corresponding state in g
        Args:
            vScc: State in the scc

        Returns: Corresponding state in g
        """
        N = len(revrename)
        ndown = vScc % N  # Project onto zero level

        return revrename[ndown]  # zero level -> g

    # Assuming that there are no two edges with the same (src, dst, cond)
    # (We can not use acc as it is modified via the degen)
    edgeDict = dict()
    for e in br.g.edges():
        en = br.g.edge_number(e)
        eId = (e.src, e.dst, e.cond)
        assert eId not in edgeDict.keys()
        edgeDict[eId] = en

    def fTrans(s: transition) -> transition:
        """
        Transform a transition \a s in br.gScc into one in br.g
        Args:
            ps: transition to be transformed

        Returns: Transformed transition
        """
        eScc = br.gScc.edge_storage(s.n)
        eIdProj = (fProj(eScc.src), fProj(eScc.dst), eScc.cond)
        return transition(br.g, edgeDict[eIdProj])

    tProj = []
    for ps in t:
        tProj.append(pathSegment(
            [fTrans(x) for x in ps.prefix],
            [fTrans(x) for x in ps.cycle]
        ))

    return tProj


def traceExtractionCycle1_(br: BuechiResult,
                           project: bool) -> Tuple[int, List[pathSegment]]:
    """
    Extract a *simple* cycle embedding the backedge
    Args:
        br: BuechiResult structure holding all the information
        project: Project the result onto the original graph

    Returns: A valid cycle

    """

    be = br.gScc.edge_storage(br.be)

    icMinDst = br.sccEn1[be.dst]
    icMinSrc = max(0, icMinDst - spot.get_weight(br.gScc, br.be))
    if (icMinSrc > br.opts["wup"]):
        print("Only implicit cycles can be found")

    # Attention: The destination of the backedge is the source of the trace...
    t = searchTrace(br.gScc, br.sccPred1, be.dst, be.src, icMinDst,
                    icMinSrc, br.opts["wup"], spot.get_weight(br.gScc, br.be))
    assert t, "This is not supposed to happen, there should be a viable trace"

    # Add the backedge
    if not t[-1].cycle:
        # If the last segment has no cycle -> add to "prefix"
        t[-1].prefix.append(transition(br.gScc, br.be))
    else:
        # Add a new pathSegment that is prefix only
        t.append(pathSegment([transition(br.gScc, be)], []))

    if not project:
        # The cycle was constructed
        return icMinDst, t

    # Project the cycle onto g
    return icMinDst, projectTrace_(br, t)


def traceExtractionCycle2_(br: BuechiResult,
                           project: bool) -> Tuple[int, List[pathSegment]]:
    """
    Extract a cycle embedding the backedge and passing by the WUP state br.sWup
    Args:
        br: BuechiResult structure holding all the information
        project: Project the result onto the original graph

    Returns: A valid cycle
    """

    be = br.gScc.edge_storage(br.be)
    sWup = br.sWup
    # We need to project sWup back onto br.g
    # assert br.opts["wup"] == br.prefixEn[sWup], "Expected WUP state"
    assert br.opts["wup"] == br.sccEn1[sWup], "Expected WUP state"
    assert br.opts["wup"] == br.sccEn2[sWup], "Expected WUP state"
    # Correct?

    # Part one, search for one of the energy optimal traces from
    # sWup to be.src
    # These are the energies and predecessors with postfix 1
    icMinSrc = br.opts["wup"]
    icMinDst = br.sccEn1[be.src]

    # Attention: The destination of the backedge is the source of the trace...
    t1 = searchTrace(br.gScc, br.sccPred1, sWup, be.src,
                     icMinSrc, icMinDst, br.opts["wup"])
    assert t1, "This is not supposed to happen, there should be a viable trace"

    # Part two: Take the backedge and get a trace from be.dst to sWup
    # Source of the trace is the destination for the backedge
    icMinSrc = min(br.opts["wup"], br.sccEn1[be.src] + spot.get_weight(br.gScc, be))
    assert icMinSrc >= 0, "Incoherent energy after taking backedge"
    icMinDst = br.opts["wup"]  # We need to return to sWup with WUP energy
    t2 = searchTrace(br.gScc, br.sccPred2, be.dst, sWup, icMinSrc, icMinDst, br.opts["wup"])
    assert t2, "This is not supposed to happen, there should be a viable trace"

    # Join the cycles and add the backedge
    t = t1
    if not t[-1].cycle:
        # If the last segment has no cycle -> add to "prefix"
        t[-1].prefix.append(transition(br.gScc, br.be))
    else:
        # Add a new pathSegment that is prefix only
        t.append(pathSegment([transition(br.gScc, br.be)], []))

    t += t2

    if not project:
        # The cycle was constructed
        return icMinDst, t

    # Project the cycle onto g
    return icMinDst, projectTrace_(br, t)


@dataclass
class lasso:
    prefix: List[pathSegment]  # Prefix leading to a cycle; possibly empty
    cycle: List[pathSegment]  # Cycle part

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"prefix\n{self.prefix}\ncycle\n{self.cycle}\n"

    def __deepcopy__(self, memodict={}):
        return lasso(deepcopy(self.prefix, memodict),
                     deepcopy(self.cycle, memodict))


def traceExctraction(br: BuechiResult, project: bool) -> List[pathSegment]:

    # Part one, find the cycle

    entryState = None  # Entrance state of the cycle
    if br.sccEn2 is None:
        assert br.sWup == -1, "Incoherent Result - Did not expect a WUP state"
        entryState = br.gScc.edge_storage(br.be).dst  # State in gScc
        icCycle, cycle = traceExtractionCycle1_(br, project)
    else:
        assert (0 <= br.sWup) and (br.sWup < br.gScc.num_states()), "Incoherent Result - Missing WUP state"
        entryState = br.sWup  # State in gScc
        icCycle, cycle = traceExtractionCycle2_(br, project)

    # Project the entry state onto the original graph
    # Note entry state is necessarily in 0 level
    revrename = {v: k for k, v in br.renameDict.items()}
    entryState = revrename[entryState]

    # Part two find a prefix for the cycle
    if (entryState != br.g.get_init_state_number()):
        tpre = searchTrace(br.g, br.prefixPred, br.opts["s0"],
                           entryState, br.opts["ic"], icCycle, br.opts["wup"])
        assert tpre, "This is not supposed to happen, there should be a viable trace"
    else:
        tpre = []
    # tpre is always in br.g

    return lasso(tpre, cycle)
