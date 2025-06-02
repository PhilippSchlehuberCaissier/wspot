## @package WBA_utils
# Utility functions for solving energy problems in weighted Büchi automata.
#
# This file contains our main contributions, notably\n
# The main algorithm, algorithm 1 in the paper\n
# The helper algorithms for energy computations,\n
# subsumed in algorithm 2 in the paper\n

from typing import List, Tuple, Dict, Union, Callable

from dataclasses import dataclass, field
import spot
from copy import deepcopy
import array

import ipython_utils as ipy
ipy_utils = ipy.IPythonUtils()

__bench_stats__ = {"n_backedges": 0, "n_bf_iter": 0, "n_scc": 0,
                   "n_pump_loop": 0, "n_propagate": 0}


def reset_stats():
    for k in __bench_stats__.keys():
        __bench_stats__[k] = 0


def get_stats():
    return __bench_stats__

# The basic counting degeneralization, as described in
# subsection "Degeneralization"

# We are currently working towards using spots built-in, more
# efficient degeneralization


## Find the entering states of a WBA's SCC.
#
# @param aut (twa_graph): the automaton we work on
# @param ssi (scc_info): spot scc_info structure for aut
# @param idx (int): For which SCC to compute the entering states
# @return A list of int containing all entering states by number
def get_entering_states(aut, ssi, idx):
    res = set()
    for e in aut.edges():
        if (ssi.scc_of(e.dst) == idx) and (ssi.scc_of(e.src) != idx):
            res.add(e.dst)
    return res


## Degeneralise a given SCC.
# needs the graph, scc_info and the idx of the SCC to treat
# @param aut (twa_graph): original automaton
# @param ssi (scc_info): spot scc_info structure
# @param idx (int): index of the SCC to degeneralise
# @return A tuple of (A new twa_graph corresponding to the degeneralization,
# a list containing the edge numbers of accepting edges,
# a dict mapping states in the SCC to states in the first level)\n
# The last info allows to root the scc in the original automaton
def degen_counting(aut, ssi, idx):
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
                c2 = (c + 1) % n_color
                if c2 == 0:
                    acc = spot.mark_t([0])

            ne = aut_degen.new_edge(
                c * n_states_orig + src_loc,
                c2 * n_states_orig + dst_loc,
                e.cond,
                acc
            )
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
        aut_degen.set_init_state(rename[si])
        # This is only one of possibly several
        break
    return aut_degen, acc_edge, rename


from array import array


def format_energie_(en: List[int]) -> str:
    """
    Format the given energy as string
    Args:
        en: Current energy as list like

    Returns: str
    """

    return '\n'.join([f'{snr}:{enx}' for snr, enx in enumerate(en)])


def format_pred_aut_(aut: spot.twa_graph, pred: List[List[int]]) -> str:
    """
    Helper function to correctly format optimal predecessors
    Args:
        aut: Automaton which we currently work on
        pred: List of predecessor lists for each state

    Returns: string representation
    """
    res = ""
    for s in range(aut.num_states()):
        res += f"{s}: {'[' if pred[s] else 'None  '}"
        for en in pred[s]:
            e = aut.edge_storage(en)
            res += f"({e.src},{spot.get_weight(aut, en)}), "
        res = res[:-2] + (']' if pred[s] else '') + '\n'
    return res


## Class allowing to run iterations of the modified bellman-ford algorithm.
#
# Holds all necessary variables and member functions described in algorithm 2.
# Most of them have additional optimiations
class mod_BF_iter:
    def __init__(self, g: spot.twa_graph):
        self.g_ = g

    def init(self):
        self.N_ = self.g_.num_states()
        # Base values
        # today integer inf?; 0 is currently lower bound so ok I guess
        self.E_ = array('q', self.N_*[-1])
        # Modification for trace extraction: We need to store all
        # transitions that have been optimal at some point
        self.Pred_ = [array('Q', []) for _ in range(self.N_)]
        self.isWaiting_ = array('b', self.N_*[False])
        # Whether the last "action" changed the energy of the node
        # Also used to detect the fixpoint
        self.changedE_ = array('b', self.N_ * [True])
        self.Waiting_ = array('L')
        # For Loop searching
        # -1: Postfix of a loop, 0: "Free",
        # 1: the current loop, 2: old loop or postfix
        self.onLoop_ = array('b', self.N_*[0])
        # From initial state)
        self.E_[self.s0_] = self.c0_
        self.isWaiting_[self.s0_] = True
        self.Waiting_.append(self.s0_)

    # Propagate the energy along e
    # Returns if energy of dst was changed

    ## Propagates the energy along an edge
    # @param en (int): Edge number
    # @param opt (bool): Whether the optimal energy for dst is chosen or energy is always propagated
    # @return Whether the energy of dst changed
    def prop_(self, en: "edge number", opt: bool):
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
            if not ((len(self.Pred_[dst]) >= 2)
                    and (self.Pred_[dst][-1] == en)
                    and (self.Pred_[dst][-2] == en)):
                self.Pred_[dst].append(en)
            return en_prime != en_dst
        return False

    ## Mark state s as waiting
    # @param s (int): state to mark
    def mark_(self, s: "state"):
        if not self.isWaiting_[s]:
            self.isWaiting_[s] = True
            self.Waiting_.append(s)


    ## Helper function to iterate over loops
    # Must be constructed with a state on a cycle.\n
    # Will eventually raise an error otherwise
    # or loop indefinitely otherwise
    # @param si (int): initial state
    # @return yields a state till done
    def loop_(self, si: "init state"):
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
            loopItems.append((s, en))
            if s == si:
                break
        # loopItems.rotate(1)
        loopItems.reverse()
        return loopItems


    ## Helper to pump the simple positive loop containing s
    # @param s (int): initial state
    def pumpLoop_(self, s: "state"):
        __bench_stats__["n_pump_loop"] += 1

        for (sprime, _) in self.loop_(s):
            self.E_[sprime] = -2  # Special marker
            self.onLoop_[sprime] = 2  # Mark it as old
            # All of these might get their values changed
            self.mark_(sprime)
            # Ensure that the predecessor causing the loop appears twice
            if not ((len(self.Pred_[sprime]) >= 2)
                    and (self.Pred_[sprime][-1] == self.Pred_[sprime][-2])):
                self.Pred_[sprime].append(self.Pred_[sprime][-1])
        self.E_[s] = self.wup_

        counter = 0
        while True:
            counter += 1
            for (_, en) in self.loop_(s):
                if not self.prop_(en, False):
                    assert counter <= 2, "fixpoint found too late"
                    return  # fixpoint

    # State s is a candidate for a loop state that
    # needs to be pumped. It could however
    # be either on the loop, or the postfix or
    # the postfix of a loop already pumped
    # @param s (int): State to be checked
    def checkLoop(self, s: "state"):
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


    ## Pump all (energy positive) loops of the current iteration
    def pumpAll(self):
        # Reset who is on a loop
        self.onLoop_ = array('b', self.N_*[0])

        # Check for each state if loop candidate
        for s in range(self.N_):
            if not self.changedE_[s]:
                continue
            if self.onLoop_[s] != 0:
                continue  # State belongs to some other loop or postfix
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
                # Loop candidate
                self.checkLoop(dst)

    def ensureLoopPred(self):
        """
        Search for loops induced by the last predecessors in the graph,
        if necessary, make them appear twice in the list.
        #todo This should probable be factor with pumpAll at some point

        Returns:

        """

        self.onLoop_ = array('I', self.N_*[0])
        searchIdx = 0
        for s in range(self.N_):
            searchIdx += 1
            if self.onLoop_[s] != 0:
                continue  # State belongs to some other loop or postfix

            # Check this state
            # Note that here, it does not necessarily have to loop,
            # but it might also simply return to the source state
            # (However the source state might also be on a loop...)
            sprime = s

            while (self.onLoop_[sprime] == 0) and self.Pred_[sprime]:
                # Continue searching along the loop.
                # If the current state has no predecessor, he can not be part of a loop
                self.onLoop_[sprime] = searchIdx
                sprime = self.g_.edge_storage(self.Pred_[sprime][-1]).src

            if self.onLoop_[sprime] == searchIdx:
                # We have found a loop
                for (sloop, _) in self.loop_(sprime):
                    if not ((len(self.Pred_[sloop]) >= 2)
                            and (self.Pred_[sloop][-1] == self.Pred_[sloop][-2])):
                        self.Pred_[sloop].append(self.Pred_[sloop][-1])
            else:
                # This was just some prefix from the initial state
                pass

    def BF1(self):
        __bench_stats__["n_bf_iter"] += 1

        isWaiting2_ = array('b', self.N_ * [False])
        Waiting2_ = array('L')
        # Swap
        # self.isWaiting_, isWaiting2_ = isWaiting2_, self.isWaiting_
        # self.Waiting_, Waiting2_ = Waiting2_, self.Waiting_

        for _ in range(self.N_):
            if not self.isWaiting_:
                break  # Early exit

            isWaiting2_ = array('b', self.N_ * [False])
            # There is no "fill" for a base array

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


    ## Computes for each state the maximal energy for which it can be reached from s0 with initial credit c0 given the weak upper bound wup
    # @param s0 (int): the initial state
    # @param wup (int): the weak upper bound
    # @param c0 (int): the initial credit
    # :param asGen: If set to true, it will yield the current energy levels and predecessors at every iteration
    def FindMaxEnergy_(self,
                       s0: "state",
                       wup: "weak upper bound",
                       c0: "Initial credit"):
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

        # Ensure that even loops that did not get pumped/
        # traversed multiple times, the loop predecessors still appear twice
        self.ensureLoopPred()

    def FindMaxEnergyGen(self,
                         s0: "state",
                         wup: "weak upper bound",
                         c0: "Initial credit"):
        return self.FindMaxEnergy_(s0, wup, c0)

    def FindMaxEnergy(self,
                      s0: "state",
                      wup: "weak upper bound",
                      c0: "Initial credit"):
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

## Solve an ɷ-regular energy game in an energy ɷ-automaton.
#
# @param aut (HOA automaton): automaton as twa_graph
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @param do_display: 0 No information is displayed at all\n
#                    1 Only text is shown\n
#                    2 The (sub)-graphs are shown as well, only works from jupyter
# @return True if there is a (wup, c0) accepting Büchi path in hoa, False otherwise.
def OmegaEnergy(aut: "HOA automaton",
                s0: "state",
                wup: "weak upper bound",
                c0: "initial credit",
                do_display: "show iterations and info" = 0) -> BuechiResult:
    """Searches for energy feasible lasso in the given automaton from the initial state
    with a weak upper bound of \a wup and an initial credit of \a c0

    Returns a BuechiResult allowing to extract the trace.
    """
    if isinstance(aut, str):
        hoa = spot.automaton(aut)
    else:
        hoa = aut


    ipy_utils.set_display_mode(do_display)
    acc_cond = hoa.acc()

    # Empty automaton or f
    if hoa.num_states() == 0 or acc_cond.is_f():
        ipy_utils.print_c("This automaton is empty or its condition is False!")
        return BuechiResult()

    # Condition is t
    if acc_cond.is_t():
        ipy_utils.print_c("True condition detected.")
        return TrueEnergy(hoa, s0, wup, c0)

    # Büchi (can be generalized)
    if acc_cond.is_generalized_buchi():
        ipy_utils.print_c("(Generalized) Büchi condition detected.")
        return BuechiEnergy(hoa, s0, wup, c0, None, None)

    # Co-Büchi
    if acc_cond.is_co_buchi():
        ipy_utils.print_c("(Generalized) co-Büchi condition detected.")
        return CoBuechiEnergy(hoa, s0, wup, c0)

    # Parity
    # Implements the algorithm presented in Section 7
    if acc_cond.is_parity()[0]:
        ipy_utils.print_c("Parity condition detected.")
        return ParityEnergy(hoa, s0, wup, c0)

    # Rabin
    p = acc_cond.is_rabin()
    if p != 1:
        ipy_utils.print_c("Rabin condition detected.")
        return RabinEnergy(hoa, p, s0, wup, c0)

    # TODO other automata types
    ipy_utils.print_c("Unknown automaton type. Assuming acceptance condition is t.")
    ipy_utils.print_c("This will lead to errors in trace extraction if the acceptance condition is not t.")
    return TrueEnergy(hoa, s0, wup, c0)

## Remove every transition with maximal priority in an automaton.
#
# @param aut (HOA automaton): automaton as twa_graph
# @param is_max (bool): is this a parity-max automaton? (defaults to True for non-parity automata)
# @return a new automaton without highest-priority transitions, or an empty automaton if there is only one color
def PrunePriority(aut: "HOA automaton",
                  is_max: "bool" = True):
    aut_new = spot.make_twa_graph(aut, spot.twa_prop_set.all())
    aut_new.copy_named_properties_of(aut)

    color = MaxColor(aut) if is_max else MinColor(aut)

    # Remove relevant edges using the method provided in twagraph-internals.ipynb
    # TODO might have problems with automata using non-conventional state numbering
    for s in range(aut_new.num_states()):
        it = aut_new.out_iteraser(s)
        while it:
            e = it.current()
            if e.acc.has(color):
                it.erase()
            else:
                it.advance()

    return aut_new


## Return the highest priority in an automaton.
#
# @param aut (HOA automaton)
# @return the highest priority used in aut, -1 if no priorities were found
def MaxColor(aut: "HOA automaton"):
    try:
        return max([acc_set
                    for e in aut.edges()
                    for acc_set in list(e.acc.sets())])
    except ValueError:
        return -1


## Return the lowest priority in an automaton.
#
# @param aut (HOA automaton)
# @return the lowest priority used in aut, -1 if no priorities were found
def MinColor(aut: "HOA automaton"):
    try:
        return min([acc_set
                    for e in aut.edges()
                    for acc_set in list(e.acc.sets())])
    except ValueError:
        return -1


## Solve an ɷ-regular energy game in a parity automaton.
#
# @param hoa (HOA automaton): parity automaton as twa_graph
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @return True if there is a (wup, c0) accepting Büchi path in hoa, False otherwise.
def ParityEnergy(hoa: "parity automaton",
                 s0: "state",
                 wup: "weak upper bound",
                 c0: "initial credit"
                 ):
    def scc_one_parity(acc_set, look_for_odd):
        ## Return True if acc_set contains only odd sets (even sets if look_for_odd is set to False), False otherwise.
        # @param acc_set(mark_t): set of all acceptance sets in a SCC
        # @param look_for_odd(bool): are we looking for odd sets?

        acc_map = [col for col in acc_set.sets() if col % 2 != look_for_odd]
        return acc_map == []

    parity_status = hoa.acc().is_parity()
    if not parity_status[2]:
        raise ValueError("ParityEnergy must be called with a pure parity automaton.")

    is_max = parity_status[1]
    is_odd = parity_status[2]

    bf = mod_BF_iter(hoa)
    # whole automaton
    # Finds optimal prefix energy for each
    # state, disregarding the colors
    assert s0 == hoa.get_init_state_number()
    en, pred = bf.FindMaxEnergy(hoa.get_init_state_number(), wup, c0)
    ipy_utils.print_c("Prefix energy per state", format_energie_(en),
             "\nCurrent optimal predecessor", format_pred_aut_(hoa, pred), sep='\n')
    ipy_utils.print_c("""State names are: "state number, max energy"\nOptimal predecessor is highlighted in pink""")
    hoa.set_state_names([f"{i},{ei}" for i, ei in enumerate(en)])

    ipy_utils.highlight_c(hoa, pred, opt="tsbrg")

    ssi = spot.scc_info(hoa)
    # Loop over all SCCs
    for i in range(ssi.scc_count()):
        so = ssi.states_of(i)
        new_states = dict()
        for j, s in enumerate(so):
            new_states[s] = j

        scc = spot.make_twa_graph(hoa.get_dict())
        scc.copy_ap_of(hoa)
        scc.new_states(len(so))

        # Skip SCCs with all-odd acceptance sets in the even case
        if scc_one_parity(ssi.acc_sets_of(i), not is_odd):
            continue

        for e in ssi.inner_edges_of(i):
            ne = scc.new_edge(
                new_states[e.src],
                new_states[e.dst],
                e.cond,
                e.acc
                )
            spot.set_weight(scc, ne, spot.get_weight(hoa, e))

        ipy_utils.display_c(scc)
        ipy_utils.print_c("Checking SCC", i)
        ipy_utils.display_c(scc)

        # TODO use simplified acceptance condition
        current_color = MaxColor(scc) if is_max else MinColor(scc)
        if current_color == -1:
            return BuechiResult()

        # TODO current implementation allocates a LOT of memory
        if (current_color % 2 == 0 and is_odd) or (current_color % 2 == 1 and not is_odd):
            ipy_utils.print_c(f"Acceptance set {current_color} will be pruned from this SCC")
            scc_copy = PrunePriority(scc, is_max)
            ipy_utils.display_c(scc_copy)
            return ParityEnergy(scc_copy, s0, wup, c0)
        else:
            ipy_utils.print_c(f"Solving Büchi reduction of this SCC for acceptance set {current_color}")
            # Create a copy of the current automaton
            scc_copy = spot.make_twa_graph(scc, spot.twa_prop_set.all())
            scc_copy.copy_named_properties_of(scc)

            # Recolor and set the acceptance condition to Büchi
            for e in scc_copy.edges():
                e.acc = spot.mark_t({0}) if e.acc.has(current_color) else spot.mark_t()
            scc_copy.set_buchi()

            ipy_utils.display_c(scc_copy)

            # Solve in this new automaton
            buchi_res = BuechiEnergy(scc_copy, s0, wup, c0, None, None)
            return buchi_res if buchi_res else ParityEnergy(PrunePriority(scc, is_max), s0, wup, c0)


## Solve an ɷ-regular energy game in a co-Büchi automaton (legacy algorithm).
#
# @param hoa (HOA automaton): generalized weighted co-büchi automaton as twa_graph
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @return True if there is a (wup, c0) accepting Büchi path in hoa, False otherwise.
def LEGACY_CoBuechiEnergy(hoa: "co-Büchi automaton",
                   s0: "state",
                   wup: "weak upper bound",
                   c0: "initial credit"
                   ):
    # Algorithm:
    # First, calculate the prefixes in the original automaton.
    # Then, try to find an accepting loop in the automaton
    # created by removing every edge accepting a color k for every color
    # and setting every other edge's acceptance to 0.
    # TODO find a more efficient algorithm

    bf = mod_BF_iter(hoa)
    # whole automaton
    # Finds optimal prefix energy for each
    # state, disregarding the colors
    assert s0 == hoa.get_init_state_number()
    en, pred = bf.FindMaxEnergy(hoa.get_init_state_number(), wup, c0)
    ipy_utils.print_c("Prefix energy per state", format_energie_(en),
             "\nCurrent optimal predecessor", format_pred_aut_(hoa, pred), sep='\n')
    ipy_utils.print_c("""State names are: "state number, max energy"\nOptimal predecessor is highlighted in pink""")
    hoa.set_state_names([f"{i},{ei}" for i, ei in enumerate(en)])
    ipy_utils.highlight_c(hoa, pred, opt="tsbrg")

    # "Smart" approach
    # Algorithm:
    # we first calculate the prefixes regardless of acceptance sets
    # For each color, we then remove the associated edges
    # and try to find >= 0 loops using a dfs
    # We then check if at least one of these loops is energy accepting
    # (Example: this won't work
    #     5        5
    # a -----> b -----> c -\
    # ^                    |
    # \--------------------/
    #         -8
    # if the wup is 5, even though it is a positive loop)

    for col in range(hoa.acc().num_sets()):
        ipy_utils.print_c(f"Examining color {str(col)}")
        sub_hoa = spot.make_twa_graph(hoa, spot.twa_prop_set.all())
        sub_hoa.copy_named_properties_of(hoa)
        sub_hoa.set_acceptance(spot.acc_cond("t"))

    for i in range(sub_hoa.num_states()):
        it = sub_hoa.out_iteraser(i)
        while it:
            e = it.current()
            if e.acc.has(col):
                it.erase()
            else:
                it.advance()

    ipy_utils.display_c(sub_hoa)
    ipy_utils.print_c("Finding potential loops in this automaton")

    # TODO this might not work if there are jumps in state numbering
    V = sub_hoa.num_states()

    # "Stack" of states being processed
    # Note that we push couples composed of the state and the energy attained
    succ = [([], None, (s0, c0))]
    # List of already seen couples
    discovered = []
    examined_loops = []

    while succ != []:
        (path, current_edge, (current_state, current_energy)) = succ.pop(0)
        ipy_utils.print_c(f"Now processing state {current_state} with inbound energy {current_energy}")
        ipy_utils.print_c([f"{e.src} > {e.dst}" for e in path])

        # Check if there's a loop
        # By definition of path, there won't be nested loops
        # Also by definition, the last element of path will close the loop
        loop_already_seen = False
        if len(path) != 0:
            closing_state = path[-1].dst
            for start_index in range(-1, -len(path) - 1, -1):
                if path[start_index].src == closing_state:
                    # We found a loop
                    # Check if the loop is energy-feasible
                    prefix_edges = path[:start_index]
                    loop_edges = path[start_index:]
                    loop_length = len(loop_edges)
                    ipy_utils.print_c(f"Found a candidate loop from state {closing_state} (loop length: {len(loop_edges)} edges)")

                    # We won't process already examined loops
                    # We need to take into account shifted loops
                    for loop in examined_loops:
                        for _ in range(len(loop)):
                            loop.append(loop.pop(0))
                            if loop == loop_edges:
                                ipy_utils.print_c("We know this is a negative loop")
                                loop_already_seen = True
                    if loop_already_seen:
                        break

                    examined_loops.append(loop_edges)
                    energy = en[closing_state]

                    # We need to rotate the loop loop_length times.
                    # See many_iterations_co_buechi_flattened.hoa,
                    #     co_buechi_shifted_loop.hoa
                    # for cases where this procedure is necessary to find a valid loop
                    # Also see lemma 4.4 in the paper
                    for _ in range(loop_length):
                        # Idea: the loop is feasible if, starting from closing_state, the resulting energy (w.r.t. the WUP) is >= to the starting energy.
                        for seg in range(loop_length):
                            loop_segment = loop_edges[seg]
                            energy = min(wup,
                                         energy + spot.get_weight(sub_hoa, loop_segment)
                                         )
                        ipy_utils.print_c(f"Final energy is {energy} (initial was {en[closing_state]})")
                        if energy < en[closing_state] or energy < 0:
                            # Non-accepting loop (energy loss)
                            ipy_utils.print_c("This is not an accepting loop")
                            # Shift the loop
                            loop_edges.append(loop_edges.pop(0))
                            closing_state = loop_edges[-1].dst
                            ipy_utils.print_c(f"Shifting the loop, now starting at {closing_state}")
                            energy = en[closing_state]
                        else:
                            # Feasible loop
                            # TODO return a BuechiResult
                            ipy_utils.print_c("Found an accepting loop")
                            return True
                    ipy_utils.print_c("This loop has been entirely shifted, proceeding to next candidate loop")

        # Continue dfs if no accepting loop was found earlier
        if (current_state, current_energy) not in discovered and not loop_already_seen:
            discovered.append((current_state, current_energy))
            # Get successors and edges leading to them
            # TODO this is suboptimal
            for e in sub_hoa.edges():
                if e.src != current_state:
                    continue
                next_energy = min(wup, current_energy + spot.get_weight(sub_hoa, e))
                if next_energy >= 0:
                    ipy_utils.print_c(f"Pushing next state {e.dst} with target energy {next_energy} (reached from ({current_state}, {current_energy}))")
                    succ.append((path + [e], e, (e.dst, next_energy)))

        ipy_utils.print_c(f"End processing ({current_state}, {current_energy})")

    return BuechiResult()


## Solve an ɷ-regular energy game in a co-Büchi automaton.
#
# @param hoa (HOA automaton): generalized weighted co-büchi automaton as twa_graph
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @return True if there is a (wup, c0) accepting Büchi path in hoa, False otherwise.
def CoBuechiEnergy(hoa: "co-Büchi automaton",
                   s0: "state",
                   wup: "weak upper bound",
                   c0: "initial credit"
                   ):
    # Algorithm:
    # First, calculate the prefixes in the original automaton.
    # Then, try to find an accepting loop in the automaton
    # created by removing every edge accepting a color k for every color
    # and setting every other edge's acceptance to 0.
    # TODO find a more efficient algorithm

    bf = mod_BF_iter(hoa)
    # whole automaton
    # Finds optimal prefix energy for each
    # state, disregarding the colors
    assert s0 == hoa.get_init_state_number()
    en, pred = bf.FindMaxEnergy(hoa.get_init_state_number(), wup, c0)
    ipy_utils.print_c("Prefix energy per state", format_energie_(en),
             "\nCurrent optimal predecessor", format_pred_aut_(hoa, pred), sep='\n')
    ipy_utils.print_c("""State names are: "state number, max energy"\nOptimal predecessor is highlighted in pink""")
    hoa.set_state_names([f"{i},{ei}" for i, ei in enumerate(en)])
    ipy_utils.highlight_c(hoa, pred, opt="tsbrg")

    # "Smart" approach
    # Algorithm:
    # we first calculate the prefixes regardless of acceptance sets
    # For each color, we then remove the associated edges
    # and try to find >= 0 loops using a dfs
    # We then check if at least one of these loops is energy accepting
    # (Example: this won't work
    #     5        5
    # a -----> b -----> c -\
    # ^                    |
    # \--------------------/
    #         -8
    # if the wup is 5, even though it is a positive loop)

    for col in range(hoa.acc().num_sets()):
        ipy_utils.print_c(f"Examining color {str(col)}")
        sub_hoa = spot.make_twa_graph(hoa, spot.twa_prop_set.all())
        sub_hoa.copy_named_properties_of(hoa)
        sub_hoa.set_acceptance(spot.acc_cond("t"))

        for i in range(sub_hoa.num_states()):
            it = sub_hoa.out_iteraser(i)
            while it:
                e = it.current()
                if e.acc.has(col):
                    it.erase()
                else:
                    it.advance()

    ipy_utils.display_c(sub_hoa)
    ipy_utils.print_c("Finding potential loops in this automaton")

    # TODO this might not work if there are jumps in state numbering
    V = sub_hoa.num_states()

    # "Stack" of states being processed
    # Note that we push couples composed of the state and the energy attained
    succ = [([], None, (s0, c0))]
    # List of already seen states along with their predecessors
    discovered = []

    while succ != []:
        (path, current_edge, (current_state, predecessor)) = succ.pop(0)
        ipy_utils.print_c(f"Now processing state {current_state} reached from {predecessor}")
        ipy_utils.print_c([f"{e.src} > {e.dst}" for e in path])

        # Check if there's a loop
        # By definition of path, there won't be nested loops
        # Also by definition, the last element of path will close the loop
        if len(path) != 0:
            closing_state = path[-1].dst
            for start_index in range(-1, -len(path) - 1, -1):
                if path[start_index].src == closing_state:
                    # We found a loop
                    # Check if the loop is energy-feasible
                    prefix_edges = path[:start_index]
                    loop_edges = path[start_index:]
                    loop_length = len(loop_edges)
                    ipy_utils.print_c(f"Found a candidate loop from state {closing_state} (loop length: {len(loop_edges)} edges)")

                    # Pump this loop twice
                    E = {}
                    for e in loop_edges:
                        E[e.src] = -1
                    E[closing_state] = en[closing_state]
                    ipy_utils.print_c(f"Initial energy at {closing_state} is {E[closing_state]}")
                    ipy_utils.print_c("Pumping loop")
                    loop_ok = False

                    for k in range(2):
                        while 1:
                            for e in loop_edges:
                                eprime = max(0, min(
                                    wup,
                                    E[e.src] + spot.get_weight(sub_hoa, e)
                                )
                                             )
                                if eprime == E[e.dst]:
                                    loop_ok = True
                                    break
                                E[e.dst] = eprime
                            if loop_ok:
                                break

                        ipy_utils.print_c(f"Final energy after pumping at {closing_state} is {E[closing_state]}")
                        ipy_utils.print_c(E)

                    final_energy = E[closing_state]

                    # Backtrack the loop
                    # We don't have to worry about energy being less than 0,
                    # since we wouldn't be able to loop back to this state otherwise
                    energy = E[closing_state]
                    for i in range(len(loop_edges) - 1, -1, -1):
                        e = loop_edges[i]
                        energy = min(
                            wup,
                            energy - spot.get_weight(sub_hoa, e)
                            )
                    ipy_utils.print_c(f"Starting energy after backtracking the loop at {closing_state} is {energy}")

                    if energy > final_energy:
                        ipy_utils.print_c("This is not an accepting loop")
                    else:
                        # Feasible loop
                        # TODO return a BuechiResult
                        ipy_utils.print_c("Found an accepting loop")
                        return True
                    
        # Continue dfs if no accepting loop was found earlier
        if (current_state, predecessor) not in discovered:
            discovered.append((current_state, predecessor))
            # Get successors and edges leading to them
            # TODO this is suboptimal
            for e in sub_hoa.edges():
                if e.src != current_state:
                    continue
                if (e.dst, e.src) in discovered:
                    continue
                ipy_utils.print_c(f"Pushing next state {e.dst} (reached from {current_state})")
                succ.append((path + [e], e, (e.dst, e.src)))

        ipy_utils.print_c(f"End processing {current_state}")

    return BuechiResult()


## Solve an ɷ-regular energy game in a Rabin automaton.
#
# @param hoa (HOA automaton): generalized weighted Rabin automaton as twa_graph
# @param p (int): number of acceptance set pairs
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @return True if there is a (wup, c0) accepting Büchi path in hoa, False otherwise.
def RabinEnergy(hoa: "Rabin automaton",
                p: int,
                s0: "state",
                wup: "weak upper bound",
                c0: "initial credit"
                ):
    # TODO Try to find a more efficient algorithm
    # Algorithm:
    # for each accepting state pair (f, i), check if hoa with Büchi condition Inf(i) has a Büchi accepting path when removing every edge that accepts f
    # Else move on to the next pair
    for k in range(p):
        f = 2 * k
        i = 2 * k + 1

        buchi_hoa = spot.make_twa_graph(hoa, spot.twa_prop_set.all())
        buchi_hoa.copy_named_properties_of(hoa)
        buchi_hoa.set_buchi()

        # First iteration: remove f-accepting edges
        for i in range(buchi_hoa.num_states()):
            it = buchi_hoa.out_iteraser(i)
            while it:
                e = it.current()
                if e.acc.has(f):
                    it.erase()
                else:
                    it.advance()

        # Second iteration: set acceptance to Büchi
        for buchi_e in buchi_hoa.edges():
            buchi_e.acc = spot.mark_t({0}) if buchi_e.acc.has(i) else spot.mark_t()

        energy = BuechiEnergy(buchi_hoa, s0, wup, c0, None, None)
        if energy:
            return energy

    return BuechiResult()


## Solve an ɷ-regular energy game in an automaton with an acceptance condition of t.
#
# @param hoa (HOA automaton): automaton as twa_graph
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @return True if there is a (wup, c0) accepting Büchi path in hoa, False otherwise.
def TrueEnergy(hoa: "automaton",
               s0: "state",
               wup: "weak upper bound",
               c0: "initial credit"
               ):
    # Algorithm: promote every edge to back edge and run BuechiEnergy on the new automaton
    buchi_hoa = spot.make_twa_graph(hoa, spot.twa_prop_set.all())
    buchi_hoa.copy_named_properties_of(hoa)
    buchi_hoa.set_buchi()

    for e in buchi_hoa.edges():
        e.acc = spot.mark_t({0})

    return BuechiEnergy(buchi_hoa, s0, wup, c0, None, None)


## Solve an ɷ-regular energy game in a Büchi automaton.
#
# @param hoa (HOA automaton): generalized weighted büchi automaton as twa_graph
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @param en: array containing the prefix energies for each state.
# If it is not provided, the prefixes are calculated instead.
# @param pred: array containing the optimal predecessors for each state
# @return True if there is a (wup, c0) accepting Büchi path in hoa, False otherwise.
def BuechiEnergy(aut: "Büchi automaton",
                 s0: "state",
                 wup: "weak upper bound",
                 c0: "initial credit",
                 en: "prefix energies array" = None,
                 pred: "optimal predecessors array" = None
                 ):
    if not (aut.acc().num_sets() >= 1) and aut.acc().is_generalized_buchi():
        raise RuntimeError("Automaton does not have a generalized buechi acceptance.")

    opts = {"wup": wup, "ic": c0, "s0": aut.get_init_state_number()}

    ipy_utils.print_c("Original automaton")
    ipy_utils.display_c(aut, "tsbrg")

    if not en:
        bf = mod_BF_iter(aut)
        # whole automaton
        # Finds optimal prefix energy for each
        # state, disregarding the colors
        assert s0 == aut.get_init_state_number()
        en, pred = bf.FindMaxEnergy(aut.get_init_state_number(), wup, c0)
        ipy_utils.print_c("Prefix energy per state", format_energie_(en),
            "\nCurrent optimal predecessor", format_pred_aut_(aut, pred), sep='\n')
        ipy_utils.print_c("""State names are: "state number, max energy"\nOptimal predecessor is highlighted in pink""")
        aut.set_state_names([f"{i},{ei}" for i, ei in enumerate(en)])
        ipy_utils.highlight_c(aut, pred, opt="tsbrg")

    ssi = spot.scc_info(aut)
    # Loop over all SCCs
    for i in range(ssi.scc_count()):
        if not ssi.is_accepting_scc(i):
            continue
        __bench_stats__["n_scc"] += 1
        ipy_utils.print_c("Checking SCC", i)
        aut_degen, acc_edge, rename = degen_counting(aut, ssi, i)
        ipy_utils.print_c(f"Degeneralized SCC has: {aut_degen.num_states()} states, {aut_degen.num_edges()} edges and {len(acc_edge)} back-edges.")

        revrename = {v: k for k, v in rename.items()}

        # renaming of states
        names = ["" for _ in range(len(rename))]
        for old, new in rename.items():
            names[new] = str(old)
        names = names * aut.get_acceptance().used_sets().max_set()
        for i in range(len(names)):
            names[i] = names[i]+":"+str(i//len(rename))
        aut_degen.set_state_names(names)

        ipy_utils.print_c(f"Current SCC with: {aut_degen.num_states()} states and {len(acc_edge)} back-edges")
        ipy_utils.print_c("""Associating states in the original automaton to the corresponding states in lvl 0 of the degeneralised SCC""",
                rename, sep="\n")
        # Update names
        aut_degen.set_state_names([f"{i}" for i in range(aut_degen.num_states())])
        ipy_utils.display_c(aut_degen, "tsbrg")

        # current degeneralized SCC
        bf2 = mod_BF_iter(aut_degen)

        # Loop over each (accepting) backedge
        # of the degeneralized current SCC
        for be_num in acc_edge:
            __bench_stats__["n_backedges"] += 1
            be = aut_degen.edge_storage(be_num)
            ipy_utils.print_c("Analysing backedge " + names[be.src],
                    "->", names[be.dst] + ".")

            start_energy = en[revrename[be.dst]]
            if start_energy < 0:
                continue
            ipy_utils.print_c("We start with " + str(start_energy)
                    + " energy in state " + names[be.dst] + ".")

            # look from backedge->destination
            (en3, pred3) = bf2.FindMaxEnergy(be.dst, wup, start_energy)
            ipy_utils.print_c("Energy starting in backedge dst", format_energie_(en3),
                    "Corresponding predecessors", format_pred_aut_(aut_degen, pred3), sep="\n")
            if en3[be.src] >= 0:
                new_energy = min(en3[be.src]+spot.get_weight(aut_degen, be_num), wup)
            else:
                new_energy = -1
            if new_energy >= start_energy:
                ipy_utils.print_c("We found a non-negative loop using edge", names[be.src],
                        "->", names[be.dst]+" directly.")
                ipy_utils.highlight_c(aut_degen, pred3, opt="tsbrg")
                return BuechiResult(aut, aut_degen, rename, opts, en, pred, be_num, en3, pred3, -1, None, None)
            else:
                # restart with the new energy
                if new_energy < 0:
                    continue
                ipy_utils.print_c("We restart with " + str(new_energy)
                        + " energy in state " + names[be.dst] + ".")

                # look again from backedge->destination but with lower start energy
                en3, pred3 = bf2.FindMaxEnergy(be.dst, wup, new_energy)
                ipy_utils.print_c(en3, pred3)
                if en3[be.src] >= 0:
                    even_newer_energy = min(en3[be.src] + spot.get_weight(aut_degen, be_num), wup)
                else:
                    even_newer_energy = -1
                ipy_utils.print_c("We arrived with " + str(even_newer_energy)
                        + " energy in state " + names[be.dst] + ".")
                if even_newer_energy >= new_energy:
                    ipy_utils.print_c("We found a non-negative loop using edge", names[be.src],
                            "->", names[be.dst] + " in the second iteration.")
                    ipy_utils.highlight_c(aut_degen, pred3, opt="tsbrg")
                    return BuechiResult(aut, aut_degen, rename, opts, en, pred, be_num, en3, pred3, -1, None, None)
                else:
                    for node, energy in enumerate(en3):
                        if energy == wup:
                            ipy_utils.print_c("we should check also from " + str(names[node])+".")
                            en4, pred4 = bf2.FindMaxEnergy(node, wup, wup)
                            ipy_utils.print_c(en4, pred4)
                            if en4[be.src] >= 0:
                                newest_energy = min(en4[be.src] + spot.get_weight(aut_degen, be_num), wup)
                                ipy_utils.print_c("We arrived with ", newest_energy,
                                        " energy in state ", names[be.dst], ".")
                                en5, pred5 = bf2.FindMaxEnergy(be.dst, wup, newest_energy)
                                ipy_utils.print_c(en5, pred5)
                                ipy_utils.print_c("We arrived with ", en5[node],
                                        " energy in state ", names[node], ".")
                                if en5[node] == wup:
                                    ipy_utils.print_c("We found a non-negative loop using node",
                                            names[node], "in the third iteration.")
                                    # TODO: look at those highlights, I have no idea
                                    ipy_utils.highlight_c(aut_degen, pred4, opt="tsbrg")
                                    ipy_utils.highlight_c(aut_degen, pred5, opt="tsbrg")
                                    # en/pred4 : WUP to be.src; en/pred5 : be.dst to WUP
                                    return BuechiResult(aut, aut_degen, rename, opts, en, pred, be_num, en4, pred4, node, en5, pred5)

    ipy_utils.print_c("No feasible Büchi run detected!")
    return BuechiResult()


@dataclass
## Class for representing energy functions segments.
class EnergySegment:
    # The segment on which this function is defined, which is included in [0, wup]
    lowerBound: int
    upperBound: int

    # Get the optimal predecessor on longer paths
    pred: int

    # We can demonstrate that in our case the equation of this segment will always be of the form e_out = a * e_in + b where a is either 0 or 1
    a: int
    b: int = -1

    @property
    def domain(self):
        return range(self.lowerBound, self.upperBound)

    def __str__(self):
        str_def = f"[{self.lowerBound}, {self.upperBound}] -> IR"
        str_a = 'e_in' if self.a else ''
        str_b = '' if self.a and not self.b else str(self.b) if not self.a else f" + {self.b}"
        return f"{str_def} ; e_in |---> {str_a}{str_b}" + f" (from {self.pred})"

    def evaluate(self, e_in):
        # TODO This works while our weights are integers
        if e_in not in range(self.lowerBound, self.upperBound + 1):
            raise ValueError()
        return self.a * e_in + self.b

    def restriction(self, low, upp):
        return EnergySegment(low, upp, self.pred, self.a, self.b)

    @staticmethod
    def nil(low, upp, pred):
        return EnergySegment(low, upp, pred, 0, -1)

    @staticmethod
    def identity(low, upp, pred):
        return EnergySegment(low, upp, pred, 1, 0)

    @staticmethod
    def const(low, upp, pred, k):
        return EnergySegment(low, upp, pred, 0, k)

    @staticmethod
    def incr(low, upp, pred, b):
        return EnergySegment(low, upp, pred, 1, b)


@dataclass
## Class for representing an energy function.
class EnergyFunction:
    segments: List[EnergySegment]
    # TODO maybe put the wup somewhere else, class property maybe?
    wup: int

    @staticmethod
    def nil(low, upp, wup):
        return EnergyFunction([EnergySegment.nil(low, upp, None)], wup)

    @property
    def domain(self):
        # Assuming segments are ordered
        return range(self.segments[0].lowerBound,
                     self.segments[-1].upperBound + 1)

    @property
    def discontinuities(self):
        # Assuming segments are ordered
        # IMPORTANT: Global lower and upper bounds are also considered as discontinuities!
        return [seg.lowerBound for seg in self.segments] + [self.segments[-1].upperBound]

    def is_in_domain(self, x):
        if x not in self.domain:
            raise ValueError(f"x = {x} is out of the domain of the function: {self}")

    ## Return the segment that is used when evaluating this function at x.
    def get_segment(self, x):
        self.is_in_domain(x)
        
        for seg in self.segments:
            # We assume that if there is a discontinuity at x, the used segment will be the one that has maximal energy.
            # This means that a segment is only usable on [lowerBound, upperBound-1] unless it is the last segment (since there is no next segment to use)
            corrected_upper = seg.upperBound if seg != self.segments[-1] else seg.upperBound + 1
            if x in range(seg.lowerBound, corrected_upper):
                return seg

    def evaluate(self, x):
        self.is_in_domain(x)

        return self.get_segment(x).evaluate(x)

    @staticmethod
    def clean(f):
        print(f"to clean: {f}")
        new_segs = []
        next_seg = None
        for old_seg in f.segments:
            print(f"processing: {old_seg}")
            if next_seg is None:
                next_seg = old_seg
                continue
            # Remove zero-length segments
            if old_seg.lowerBound == old_seg.upperBound:
                continue

            # Merge segments with the same equation
            if old_seg.a == next_seg.a and old_seg.b == next_seg.b:
                next_seg.upperBound = old_seg.upperBound
                print(f"merging segments, new segment: {next_seg}")
            else:
                print(f"next segment: {next_seg}")
                new_segs.append(next_seg)
                next_seg = old_seg
        new_segs.append(next_seg)
        return EnergyFunction(new_segs, f.wup)

    @staticmethod
    def max(f1, f2):
        new_segs = []
        # The discontinuities of the max are the union of those of the 2 functions
        discs = list(set(f1.discontinuities + f2.discontinuities))
        discs.sort()
        for i in range(len(discs) - 1):
            lower = discs[i]
            upper = discs[i+1]

            seg1 = f1.get_segment(lower)
            seg2 = f2.get_segment(lower)

            if seg1.a == seg2.a:
                next_seg = seg1 if seg1.b > seg2.b else seg2
                new_segs.append(next_seg.restriction(lower, upper))
            else:
                # Segments might be intersecting
                # Use the IVT (xor)
                if (seg1.evaluate(lower) > seg2.evaluate(lower)) != (seg1.evaluate(upper) > seg2.evaluate(upper)):
                    # found by manipulating the functions' equations
                    # again, this only works if our weights are ints
                    # intersect is guaranteed to be in ]lower, upper[ with the initial condition
                    intersect = int((seg2.b - seg1.b) / (seg1.a - seg2.a))
                    first_on_top = seg1 if seg1.evaluate(lower) > seg2.evaluate(lower) else seg2
                    new_segs.append(first_on_top.restriction(lower, intersect))
                    second_on_top = seg2 if first_on_top == seg1 else seg1
                    new_segs.append(second_on_top.restriction(intersect, upper))
                else:
                    # Segments do not intersect, we use upper because its image is guaranteed to be not equal (unless seg1 == seg2)
                    next_seg = seg1 if seg1.evaluate(upper) > seg2.evaluate(upper) else seg2
                    new_segs.append(next_seg.restriction(lower, upper))

        f = EnergyFunction(new_segs, f1.wup)
        return EnergyFunction.clean(f)

    def __add__(self, other):
        new_segs = []
        for i in range(len(self.discontinuities) - 1):
            lower = self.discontinuities[i]
            upper = self.discontinuities[i+1]
            this_seg = self.get_segment(lower)
            other_seg = other.get_segment(lower)

            # New equation is f(x) = a2(a1*x + b1) + b2
            # This is not commutative
            new_a = other_seg.a * this_seg.a
            new_b = other_seg.a * this_seg.b + other_seg.b
            if new_a == 0:
                new_segs.append(
                    EnergySegment.const(lower,
                                        upper,
                                        this_seg.pred,
                                        min(new_b, self.wup))
                )
            else:
                # Create new discontinuities if the result is < 0 or > wup
                disc1 = -new_b
                disc2 = self.wup - new_b
                if disc1 > 0:
                    new_segs.append(
                        EnergySegment.const(lower,
                                            disc1,
                                            this_seg.pred,
                                            -1)
                        )
                new_segs.append(
                    EnergySegment.incr(max(lower, disc1),
                                       min(disc2, upper),
                                       this_seg.pred,
                                       new_b)
                    )
                if disc2 < self.wup:
                    new_segs.append(
                        EnergySegment.const(disc2,
                                            self.wup,
                                            this_seg.pred,
                                            self.wup)
                        )

        f = EnergyFunction(new_segs, self.wup)
        print(str(f))
        return EnergyFunction.clean(f)

    def __str__(self):
        return " U ".join([str(seg) for seg in self.segments])


## Solve an ɷ-regular energy game in a co-Büchi automaton using Floyd-Warshall on energy functions.
def CoBuechi_FW(hoa: "co-Büchi automaton",
                s0: "state",
                wup: "weak upper bound",
                c0: "initial credit"
                ):
    hoa = spot.automaton(hoa)
    V = hoa.num_states()
    M = [
        [EnergyFunction([EnergySegment.const(0, wup, None, -1)], wup) for _ in range(V)] for _ in range(V)]

    for e in hoa.edges():
        # 3 cases:
        # edge weight is 0 -> identity
        # edge weight is > 0 -> increasing function + constant
        # edge weight is < 0 -> undefined + increasing
        weight = spot.get_weight(hoa, e)
        segs = []
        if weight > 0:
            segs = [
                EnergySegment.incr(0, wup - weight, e.src, weight),
                EnergySegment.const(wup - weight, wup, e.src, wup)
                ]
        elif weight < 0:
            segs = [
                EnergySegment.const(0, -weight, e.src, -1),
                EnergySegment.incr(-weight, wup, e.src, weight)
                ]
        else:
            segs = [
                EnergySegment.identity(0, wup, e.src)
                ]
        f = EnergyFunction(segs, wup)
        print(f"building function {f} @ ({e.src}, {e.dst})")
        M[e.src][e.dst] = EnergyFunction.clean(f)
    for v in range(V):
        M[v][v] = EnergyFunction.clean(EnergyFunction(
            [
                EnergySegment.identity(0, wup, v)
            ],
            wup))
    for k in range(V):
        for i in range(V):
            for j in range(V):
                M[i][j] = EnergyFunction.max(M[i][j], M[i][k] + M[k][j]) if M[i][k] != EnergyFunction.nil(0, wup, wup) else M[i][j]

    for li in range(len(M)):
        print(f"====== From {li} ======")
        for col in range(len(M[li])):
            print(f"to {col}: {str(M[li][col])}")

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
