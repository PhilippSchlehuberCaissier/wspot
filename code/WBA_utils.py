# This file contains our main contributions, notably
# The main algorithm, algorithm 1 in the paper
# The helper algorithms for energy computations,
# subsumed in algorithm 2 in the paper

from typing import List, Tuple, Dict, Union, Callable

from dataclasses import dataclass, field
import spot, buddy
from copy import deepcopy as deepcopy
import array

__bench_stats__ = {"n_backedges":0, "n_bf_iter":0, "n_scc":0,
                   "n_pump_loop":0, "n_propagate":0}

def reset_stats():
    for k in __bench_stats__.keys():
        __bench_stats__[k] = 0

def get_stats():
    return __bench_stats__

# The basic counting degeneralization, as described in
# subsection "Degeneralization"

# We are currently working towards using spots built-in, more
# efficient degeneralization

def get_entering_states(aut, ssi, idx):
    """_summary_

    Args:
        aut (twa_graph): the automaton we work on
        ssi (scc_info): spot scc_info structure for aut
        idx (int): For which SCC to compute the entering states

    Returns:
        set[int]: All entering states by number
    """
    res = set()
    for e in aut.edges():
        if (ssi.scc_of(e.dst) == idx) and (ssi.scc_of(e.src) != idx):
            res.add(e.dst)
    return res


def degen_counting(aut, ssi, idx):
    """
    A function that degeneralises a given SCC
    needs the graph, scc_info and the idx of the SCC to treat
    :param aut: original automaton
    :param ssi: scc_info
    :param idx: index of the SCC to degeneralise
    :return: Tuple of (A new twa_graph corresponding to the degeneralization,
                       a list containing the edge numbers of accepting edges,
                       a dict mapping states in the SCC to states in the first level)
                       The last info allows to root the scc in the original automaton
    """

    so = ssi.states_of(idx)
    # "local" number of the state
    rename = dict()
    for i, s in enumerate(so):
        rename[s] = i
    n_states_orig = len(so)
    n_color = aut.get_acceptance().used_sets().max_set()

    aut_degen = spot.make_twa_graph(aut.get_dict())
    aut_degen.copy_ap_of(aut)
    aut_degen.set_buchi()

    aut_degen.new_states(n_color*n_states_orig)

    # No optims at all
    # But store the accepting edgenumbers
    acc_edge = []
    for e in ssi.inner_edges_of(idx):
        src_loc = rename[e.src]
        dst_loc = rename[e.dst]
        w = spot.get_weight(aut, e)
        for c in range(n_color):
            c2 = c
            acc = spot.mark_t()
            if e.acc.has(c):
                c2 = (c + 1)%n_color
                if c2 == 0:
                    acc = spot.mark_t([0])

            ne = aut_degen.new_edge(c * n_states_orig + src_loc, c2 * n_states_orig + dst_loc, e.cond, acc)
            spot.set_weight(aut_degen, ne, w)
            if acc != spot.mark_t():
                # Colored backedges are treated one by one
                # Leave all backedges in the graph.
                # Return a list with all edge numbers of the accepting
                # edges
                acc_edge.append(ne)

    # Initial
    # This is never used but seems nicer
    for si in get_entering_states(aut, ssi, idx):
        aut_degen.set_init_state(rename[si]) # This is only one of possibly several
        break
    return aut_degen, acc_edge, rename


from array import array


class mod_BF_iter:
    """Class allowing to run iterations of the modified bellman-ford algorithm.
    Holds all necessary variables and member functions described in algorithm 2.
    Most of them have additional optimiations
    """

    def __init__(self, g:spot.twa_graph):
        self.g_ = g

    def init(self):
        self.N_ = self.g_.num_states()
        # Base values
        self.E_ = array('q', self.N_*[-1]) #today integer inf?; 0 is currently lower bound so ok I guess
        # Modification for trace extraction: We need to store all
        # transitions that have been optimal at some point
        self.Pred_ = [array('Q', []) for _ in range(self.N_)]
        self.isWaiting_ = array('b', self.N_*[False])
        # Whether the last "action" changed the energy of the node
        # Also used to detect the fixpoint
        self.changedE_ = array('b', self.N_ * [True])
        self.Waiting_ = array('L')
        # For Loop searching
        #-1: Postfix of a loop, 0: "Free", 1: the current loop, 2: old loop or postfix
        self.onLoop_ = array('b', self.N_*[0])
        # From initial state)
        self.E_[self.s0_] = self.c0_
        self.isWaiting_[self.s0_] = True
        self.Waiting_.append(self.s0_)

    # Propagate the energy along e
    # Returns if energy of dst was changed
    def prop_(self, en:"edge number", opt:bool):
        """
        Propagates the energy along an edge
        :param en: Edge number
        :param opt: Whether the optimal energy for dst is chosen or energy is always propagated
        :return: Whether the energy of dst changed
        """
        __bench_stats__["n_propagate"] += 1
        e = self.g_.edge_storage(en)
        src = e.src
        dst = e.dst
        ew = spot.get_weight(self.g_, en)
        en_src = self.E_[src]
        en_dst = self.E_[dst]

        en_prime = min(self.wup_, en_src + ew)

        if (en_prime >= 0) and ((not opt) or (en_prime > en_dst)):
            self.E_[dst] = en_prime
            # Add all optimal predecessors in a stutter free fashion
            # Fix by Sven: In fact we can not be as strict as stutter free
            # If the loop and the prefix overlap we need the same pred twice
            if not ((len(self.Pred_[dst]) >= 2) and (self.Pred_[dst][-1] == en) and (self.Pred_[dst][-2] == en)):
                self.Pred_[dst].append(en)
            return en_prime != en_dst
        return False

    def mark_(self, s:"state"):
        """
        Mark state s as waiting
        :param s: state to mark
        """
        if not self.isWaiting_[s]:
            self.isWaiting_[s] = True
            self.Waiting_.append(s)


    def loop_(self, si:"init state"):
        """
        Helper function to iterate over loops
        Must be constructed with a state on a cycle
        Will eventually raise an error otherwise
        or loop indefinitely otherwise

        :param si: initial state
        :return: yields a state till done
        """
        from collections import deque
        s = si
        loopItems = deque()
        def pred_(s):
            # We need to use the latest predecessor
            en = self.Pred_[s]
            assert en, "No valid Predecessor!"
            return en[-1], self.g_.edge_storage(en[-1])
        def next_(s):
            en, e = pred_(s)
            return e.src, en
        while True:
            s, en = next_(s)
            loopItems.append((s,en))
            if s == si:
                break
        loopItems.rotate(1)
        return loopItems

    def pumpLoop_(self, s:"state"):
        """
        Helper to pump the simple positive loop containing s
        :param s: initial state
        """
        __bench_stats__["n_pump_loop"] += 1

        for (sprime, _) in self.loop_(s):
            self.E_[sprime] = -2 # Special marker
            self.onLoop_[sprime] = 2 #Mark it as old
            # All of these might get their values changed
            self.mark_(sprime)
        self.E_[s] = self.wup_

        counter = 0;
        while True:
            counter += 1
            for (_, en) in self.loop_(s):
                if not self.prop_(en, False):
                    assert counter <= 2, "fixpoint found too late"
                    return #fixpoint

    def checkLoop(self, s:"state"):
        """
        State s is a candidate for a loop state that
        needs to be pumped. It could however
        be either on the loop, or the postfix or
        the postfix of a loop already pumped
        :param s: State to be checked
        """

        sprime = s

        while self.onLoop_[sprime] == 0:
            self.onLoop_[sprime] = 1
            assert self.Pred_[sprime], "Has no predecessor -> Can not be on a loop"
            # Works on the last predecessor set
            sprime = self.g_.edge_storage(self.Pred_[sprime][-1]).src

        if self.onLoop_[sprime] == 1:
            # Found a new simple positive loop
            self.pumpLoop_(sprime)
        # Mark the postfix if necessary
        sprime = s
        while self.onLoop_[sprime] == 1:
            self.onLoop_[sprime] = 2
            sprime = self.g_.edge_storage(self.Pred_[sprime][-1]).src
            # We could propagate energy here directly
            # Would we then need full BF? Optim?

    def pumpAll(self):
        """
        Pump all (energy positive) loops of the current iteration
        :return:
        """
        # Reset who is on a loop
        self.onLoop_ = array('b', self.N_*[0])

        # Check for each state if loop candidate
        for s in range(self.N_):
            if not self.changedE_[s]:
                continue
            if self.onLoop_[s] != 0:
                continue # State belongs to some other loop or postfix
            # Check if energy can increase
            # Todo Code duplication :(
            en = self.Pred_[s]
            if not en:
                # Unreachable
                print("state", s, "pred", en, "energy", self.E_[s])
                if s != self.s0_:
                    assert self.E_[s] == -1
                continue
            e = self.g_.edge_storage(en[-1])

            src = e.src
            dst = e.dst
            ew = spot.get_weight(self.g_, en[-1])
            en_src = self.E_[src]
            en_dst = self.E_[dst]

            en_prime = min(self.wup_, en_src + ew)
            if (en_prime > en_dst):
                #Loop candidate
                self.checkLoop(dst)

    def BF1(self):

        """
        Perform one round of modified, optimised Bellman-Ford
        :return:
        """

        __bench_stats__["n_bf_iter"] += 1

        isWaiting2_ = array('b', self.N_ * [False])
        Waiting2_ = array('L')
        #Swap
        #self.isWaiting_, isWaiting2_ = isWaiting2_, self.isWaiting_
        #self.Waiting_, Waiting2_ = Waiting2_, self.Waiting_

        for _ in range(self.N_):
            if not self.isWaiting_:
                break  # Early exit

            isWaiting2_ = array('b', self.N_ * [False])  #There is no "fill" for a base array
            while self.Waiting_:
                s = self.Waiting_.pop()
                for e in self.g_.out(s):
                    en = self.g_.edge_number(e)
                    changed = self.prop_(en, True)
                    if changed:
                        if not isWaiting2_[e.dst]:
                            isWaiting2_[e.dst] = True
                            Waiting2_.append(e.dst)
            self.isWaiting_, isWaiting2_ = isWaiting2_, self.isWaiting_
            self.Waiting_, Waiting2_ = Waiting2_, self.Waiting_
        return

    def FindMaxEnergy_(self, s0:"state", wup:"weak upper bound", c0:"Initial credit"):
        """
        Computes for each state the maximal energy for which it can be reached from s0 with
        initial credit c0 given the weak upper bound wup
        :param s0:
        :param wup:
        :param c0:
        :param asGen: If set to true, it will yield the current energy levels and predecessors at every iteration
        :return:
        """
        self.s0_ = s0
        self.wup_ = wup
        self.c0_ = c0

        self.init()

        def compE(oldE, newE):
            for i in range(self.N_):
                self.changedE_[i] = newE[i] != oldE[i]

        hasChanged = True
        while hasChanged:
            hasChanged = False
            oldE = deepcopy(self.E_)
            self.BF1()  # One round of (modified) Bellman-Ford
            compE(oldE, self.E_)
            hasChanged = hasChanged or any(self.changedE_)
            oldE = deepcopy(self.E_)
            yield self.E_, self.Pred_
            self.pumpAll()  # Who have got to pump it up!
            compE(oldE, self.E_)
            hasChanged = hasChanged or any(self.changedE_)
            yield self.E_, self.Pred_

    def FindMaxEnergyGen(self, s0:"state", wup:"weak upper bound", c0:"Initial credit"):
        return self.FindMaxEnergy_(s0, wup, c0)

    def FindMaxEnergy(self, s0:"state", wup:"weak upper bound", c0:"Initial credit"):
        for (En, Pred) in self.FindMaxEnergy_(s0, wup, c0):
            continue
        return (En, Pred)


# A helper structure to preserve the additional intermediate solutions
@dataclass
class BuechiResult:
    """
    Class containing all the information needed to perform trace extraction
    after running BuechiEnergy.

    If sccEn2 and sccPred2 are None, then a trace was found without investigating
    the different states attaining an energy equal to the weak upper bound.

    If there was no viable trace, all members are None
    """
    g: spot.twa_graph = None  # Original graph
    gScc: spot.twa_graph = None  # Current accepting SCC
    renameDict: Dict[int, int] = None  # Dict from the original states to the ones in the SCC
    opts: Dict = field(default_factory=dict)  # Stores options like wup etc

    prefixEn: List[int] = None  # Optimal prefix energy for each state
    prefixPred: List[List[int]] = None  # Extended optimal predecessor list

    be: int = None  # Embedded accepting backedge

    sccEn1: List[int] = None  # Energy current SCC part 1
    sccPred1: List[List[int]] = None  # Extended optimal predecessor list current SCC part 1

    sWup: int = -1  # Maximal energy state embedded in loop

    sccEn2: List[int] = None  # Energy current SCC part 2
    sccPred2: List[List[int]] = None  # Extended optimal predecessor list current SCC part 2

    def __bool__(self) -> bool:
        return self.g is not None

# Whole picture
# This is algorithm 1
# todo: Fix we do not need s0
def BuechiEnergy(hoa:"HOA automaton", s0:"state", wup:"weak upper bound", c0:"initial credit",
                 do_display:"show iterations and info"=0) -> BuechiResult:
    """Searches for energy feasible lasso in the given automaton from the initial state
    with a weak upper bound of \a wup and an initial credit of \a c0

    Returns a BuechiResult allowing to extract the trace.

    Args:
        hoa (HOA automaton): generalized weighted büchi automaton,  filename or twa_graph
        s0 (state): initial state
        wup (weak upper bound):
        c0 (initial credit):
        do_display: 0 No information is displayed at all
                    1 Only text is shown
                    2 The (sub)-graphs are shown as well, only works from jupyter
    """

    def print_c(*args, **kwargs):
        if do_display > 0:
            print(*args, **kwargs)
        return
    def display_c(aut, opt=""):
        if do_display > 1:
            display(aut.show(opt))
        return

    def highlight_c(aut: spot.twa_graph, pred: List[List[int]], predColors: List[int]=[1, 2, 3, 4, 5], opt="") -> None:
        """

        Args:
            aut: The automaton
            pred: List of lists. The list pred[s] contains all predecessors for state s
            predColors: How many predecessors should be colored and how. The -ith predecessor will be colored with the i-1th color
            opt: Additional options passed to highlight_edges

        Returns: None

        """
        if do_display > 1:
            # Create a list of edges for each color
            cDict = dict(zip(predColors, [[] for _ in predColors]))
            for s in range(aut.num_states()):
                for c, en in zip(predColors, reversed(pred[s])):
                    cDict[c].append(en)
            for c, edges in cDict.items():
                aut.highlight_edges(edges, c)
        display_c(aut, opt)

    if isinstance(hoa, str):
        aut = spot.automaton(hoa)
    else:
        aut = hoa

    opts = {"wup": wup, "ic": c0, "s0": aut.get_init_state_number()}

    print_c("Original automaton")
    display_c(aut, "tsbrg")

    bf = mod_BF_iter(aut)
    # whole automaton
    # Finds optimal prefix energy for each
    # state, disregarding the colors
    assert s0 == aut.get_init_state_number()
    en, pred = bf.FindMaxEnergy(aut.get_init_state_number(), wup, c0)
    print_c(f"Prefix energy per state\n{en}\nCurrent optimal predescessor\n{pred}")
    print_c("""State names are: "state number, max energy"\nOptimal predescessor is highlighted in pink""");
    aut.set_state_names([f"{i},{ei}" for i, ei in enumerate(en)])
    highlight_c(aut, pred, opt="tsbrg")

    ssi = spot.scc_info(aut)
    # Loop over all SCCs
    for i in range(ssi.scc_count()):
        if not ssi.is_accepting_scc(i):
            continue
        __bench_stats__["n_scc"] += 1
        print_c("Checking SCC", i)
        aut_degen, acc_edge, rename = degen_counting(aut, ssi, i)
        print_c(f"Degeneralized SCC has: {aut_degen.num_states()} states, {aut_degen.num_edges()} edges and {len(acc_edge)} back-edges.")

        revrename = {v: k for k, v in rename.items()}

        # renaming of states
        names = ["" for _ in range(len(rename))]
        for old, new in rename.items():
            names[new] = str(old)
        names = names * aut.get_acceptance().used_sets().max_set()
        for i in range(len(names)):
            names[i] = names[i]+":"+str(i//len(rename))
        aut_degen.set_state_names(names)

        print_c(f"Current SCC with: {aut_degen.num_states()} states and {len(acc_edge)} back-edges")
        print_c(rename)
        display_c(aut_degen, "tsbrg")

        # current degeneralized SCC
        bf2 = mod_BF_iter(aut_degen)

        # Loop over each (accepting) backedge
        # of the degeneralized current SCC
        for be_num in acc_edge:
            __bench_stats__["n_backedges"] += 1
            be = aut_degen.edge_storage(be_num)
            print_c("Analysing backedge "+ names[be.src],"->", names[be.dst]+".")

            start_energy = en[revrename[be.dst]]
            if start_energy < 0:
                continue
            print_c("We start with "+ str(start_energy) + " energy in state "+names[be.dst] + ".")

            # look from backedge->destination
            (en3, pred3) = bf2.FindMaxEnergy(be.dst, wup, start_energy)
            print_c(en3, pred3)
            if en3[be.src] >= 0:
                new_energy = min(en3[be.src]+spot.get_weight(aut_degen, be_num), wup)
            else:
                new_energy = -1
            if new_energy >= start_energy:
                print_c("We found a non-negative loop using edge", names[be.src],
                        "->", names[be.dst]+" directly.")
                highlight_c(aut_degen, pred3, opt="tsbrg")
                return BuechiResult(aut, aut_degen, rename, opts, en, pred, be_num, en3, pred3, -1, None, None)
            else:
                #restart with the new energy
                if new_energy < 0:
                    continue
                print_c("We restart with "+ str(new_energy) + " energy in state "+names[be.dst] + ".")

                # look again from backedge->destination but with lower start energy
                en3, pred3 = bf2.FindMaxEnergy(be.dst, wup, new_energy)
                print_c(en3, pred3)
                if en3[be.src] >= 0:
                    even_newer_energy = min(en3[be.src]+spot.get_weight(aut_degen, be_num), wup)
                else:
                    even_newer_energy = -1
                print_c("We arrived with "+ str(even_newer_energy) + " energy in state "+names[be.dst] + ".")
                if  even_newer_energy >= new_energy:
                    print_c("We found a non-negative loop using edge", names[be.src],
                            "->", names[be.dst]+" in the second iteration.")
                    highlight_c(aut_degen, pred3, opt="tsbrg")
                    return BuechiResult(aut, aut_degen, rename, opts, en, pred, be_num, en3, pred3, -1, None, None)
                else:
                    for node, energy in enumerate(en3):
                        if energy == wup:
                            print_c("we should check also from "+str(names[node])+".")
                            en4, pred4 = bf2.FindMaxEnergy(node, wup, wup)
                            print_c(en4, pred4)
                            if en4[be.src] >= 0:
                                newest_energy = min(en4[be.src]+spot.get_weight(aut_degen, be_num), wup)
                                print_c("We arrived with ", newest_energy,
                                        " energy in state ", names[be.dst], ".")
                                en5, pred5 = bf2.FindMaxEnergy(be.dst, wup, newest_energy)
                                print_c(en5, pred5)
                                print_c("We arrived with ", en5[node],
                                        " energy in state ", names[node], ".")
                                if en5[node] == wup:
                                    print_c("We found a non-negative loop using node",
                                            names[node], "in the third iteration.")
                                    # TODO: look at those highlights, I have no idea
                                    highlight_c(aut_degen, pred4, opt="tsbrg")
                                    highlight_c(aut_degen, pred5, opt="tsbrg")
                                    # en/pred4 : WUP to be.src; en/pred5 : be.dst to WUP
                                    return BuechiResult(aut, aut_degen, rename, opts, en, pred, be_num, en4, pred4, node, en5, pred5)

    print_c("No feasible Büchi run detected!")
    return BuechiResult()

@dataclass
class transition:
    g : spot.twa_graph  # Underlying graph
    n : int  # Edge number

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
                break
            except KeyError:
                pass
            srcIdx[subtrace[-1].src] = len(subtrace) - 1
    if subtrace:
        tc.append(pathSegment(subtrace, []))

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

def tryPumpLoop(e:int, t:List[transition], wup: int) -> Tuple[bool, int]:
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


def forwardExploration(ic: int, eDst:int, wup:int, t: List[pathSegment]) -> bool:
    """
    Compute whether at least eDst can be attained after traversing t

    Args:
        ic: Initial credit
        eDst: Minimal energy at destination 
        wup: weak upper bound
        t: considered tracce

    Returns: True iff the energy after traversing the path is at least \a eDst
    """

    e = ic
    for ps in t:
        if ps.prefix:
            succ, e = propAlong(e, ps.prefix, wup)
            if not succ:
                return False
        if ps.cycle:
            succ, e = tryPumpLoop(e, ps.cycle, wup)
            if not succ:
                return False
    return e >= eDst

def backwardsSearchImpl_(g: spot.twa_graph, pred: List[List[int]], gSrc: int, forwardExp: Callable,
                         ci: List[int], t: List[transition]) -> List[pathSegment]:
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

def searchTrace(g: spot.twa_graph, pred: List[List[int]], gSrc: int, gDst: int, icSrc: int, eDst: int, wup: int) -> List[pathSegment]:
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

    Returns: A viable trace as list of pastSegments; Empty if no trace was found

    """

    fforward_ = lambda t: forwardExploration(icSrc, eDst, wup, t)

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

    revrename = {v: k for k, v in br.rename.items()}
    assert len(revrename) == len(br.rename), "Should be a isomorphism"

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
        tProj.append(pathSegment([fTrans(x) for x in ps.prefix], [fTrans(x) for x in ps.cycle]))

    return tProj


def traceExtractionCycle1_(br: BuechiResult, project: bool) -> Tuple[int, List[pathSegment]]:
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
    assert (icMinSrc <= br.opts["wup"])

    # Attention: The destination of the backedge is the source of the trace...
    t = searchTrace(br.gScc, br.sccPred1, be.dst, be.src, icMinDst, icMinSrc, br.opts["wup"])
    assert t, "This is not supposed to happen, there should be a viable trace"

    # Add the backedge
    if not t[-1].cycle:
        # If the last segment has no cycle -> add to "prefix"
        t[-1].prefix.append(transition(br.gScc, be))
    else:
        # Add a new pathSegment that is prefix only
        t.append(pathSegment([transition(br.gScc, be)], []))

    if not project:
        # The cycle was constructed
        return icMinDst, t

    # Project the cycle onto g
    return icMinDst, projectTrace_(br, t)

def traceExtractionCycle2_(br: BuechiResult, project: bool) -> Tuple[int, List[pathSegment]]:
    """
    Extract a cycle embedding the backedge and passing by the WUP state br.sWup
    Args:
        br: BuechiResult structure holding all the information
        project: Project the result onto the original graph

    Returns: A valid cycle
    """

    be = br.gScc.edge_storage(br.be)
    sWup = br.sWup
    assert br.opt["wup"] == br.prefixEn[sWup], "Expected WUP state"
    assert br.opt["wup"] == br.sccEn1[sWup], "Expected WUP state"
    assert br.opt["wup"] == br.sccEn2[sWup], "Expected WUP state"
    # Correct?

    # Part one, search for one of the energy optimal traces from
    # sWup to be.src
    # These are the energies and predecessors with postfix 1
    icMinSrc = br.opt["wup"]
    icMinDst = br.sccEn1[be.src]

    # Attention: The destination of the backedge is the source of the trace...
    t1 = searchTrace(br.gScc, br.sccPred1, sWup, be.src, icMinDst, icMinSrc, br.opts["wup"])
    assert t1, "This is not supposed to happen, there should be a viable trace"

    # Part two: Take the backedge and get a trace from be.dst to sWup
    # Source of the trace is the destination for the backedge
    icMinSrc = min(br.opts["wup"], br.sccEn1[be.src] + spot.get_weight(br.gScc, be))
    assert icMinSrc >= 0, "Incoherent energy after taking backedge"
    icMinDst = br.opts["wup"]  # We need to return to sWup with WUP energy
    t2 = searchTrace(br.gScc, br.sccPred2, be.dst, sWup, icMinDst, icMinSrc, br.opts["wup"])
    assert t2, "This is not supposed to happen, there should be a viable trace"

    # Join the cycles and add the backedge
    t = t1
    if not t[-1].cycle:
        # If the last segment has no cycle -> add to "prefix"
        t[-1].prefix.append(transition(br.gScc, be))
    else:
        # Add a new pathSegment that is prefix only
        t.append(pathSegment([transition(br.gScc, be)], []))

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
        return lasso(deepcopy(self.prefix, memodict), deepcopy(self.cycle, memodict))


def traceExctraction(br: BuechiResult, project: bool) -> List[pathSegment]:

    # Part one, find the cycle

    entryState = None  #Entrance state of the cycle
    if br.sccEn2 is None:
        assert br.sWup == -1, "Incoherent Result - Did not expect a WUP state"
        entryState = br.gScc.edge_storage(br.be)
        icCycle, cycle = traceExtractionCycle1_(br, project)
    else:
        assert (0 <= br.sWup) and (br.sWup < br.gScc.num_states()), "Incoherent Result - Missing WUP state"
        entryState = br.sWup
        icCycle, cycle = traceExtractionCycle2_(br, project)

    # Part two find a prefix for the cycle
    tpre = searchTrace(br.g, br.prefixPred, br.opts["s0"], entryState, br.opts["ic"], icCycle, br.opts["wup"])
    assert tpre, "This is not supposed to happen, there should be a viable trace"
    # tpre is always in br.g

    return lasso(tpre, cycle)



