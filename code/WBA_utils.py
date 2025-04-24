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
from copy import deepcopy as deepcopy
import array

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
                c2 = (c + 1)%n_color
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
            assert self.Pred_[sprime],"Has no predecessor -> Can not be on a loop"
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
def OmegaEnergy(hoa: "HOA automaton",
                s0: "state",
                wup: "weak upper bound",
                c0: "initial credit",
                do_display: "show iterations and info" = 0) -> BuechiResult:
    """Searches for energy feasible lasso in the given automaton from the initial state
    with a weak upper bound of \a wup and an initial credit of \a c0

    Returns a BuechiResult allowing to extract the trace.
    """
    # TODO boilerplate, move this function somewhere else
    def print_c(*args, **kwargs):
        if do_display > 0:
            print(*args, **kwargs)
        return

    acc_cond = hoa.acc()

    # Empty automaton or f
    if hoa.num_states() == 0 or acc_cond.is_f():
        print_c("This automaton is empty or its condition is False!")
        return BuechiResult()

    # Condition is t
    if acc_cond.is_t():
        print_c("True condition detected.")
        return TrueEnergy(hoa, s0, wup, c0, do_display)

    # Büchi (can be generalized)
    if acc_cond.is_generalized_buchi():
        print_c("(Generalized) Büchi condition detected.")
        return BuechiEnergy(hoa, s0, wup, c0, do_display)

    # Co-Büchi
    if acc_cond.is_co_buchi():
        print_c("(Generalized) co-Büchi condition detected.")
        return CoBuechiEnergy(hoa, s0, wup, c0, do_display)

    # Parity
    # Implements the algorithm presented in Section 7
    if acc_cond.is_parity()[0]:
        print_c("Parity condition detected.")
        return ParityEnergy(hoa, s0, wup, c0, do_display)

    # Rabin
    p = acc_cond.is_rabin()
    if p != 1:
        print_c("Rabin condition detected.")
        return RabinEnergy(hoa, p, s0, wup, c0, do_display)

    # TODO other automata types
    print_c("Unknown automaton type. Assuming acceptance condition is t.")
    print_c("This will lead to errors in trace extraction if the acceptance condition is not t.")
    return TrueEnergy(hoa, s0, wup, c0, do_display)

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
# @param pau (HOA automaton): parity automaton as twa_graph
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @param do_display: 0 No information is displayed at all\n
#                    1 Only text is shown\n
#                    2 The (sub)-graphs are shown as well, only works from jupyter
# @return True if there is a (wup, c0) accepting Büchi path in hoa, False otherwise.
def ParityEnergy(pau: "parity automaton",
                 s0: "state",
                 wup: "weak upper bound",
                 c0: "initial credit",
                 do_display: "show iterations and info" = 0
                 ):
    parity_status = pau.acc().is_parity()
    if not parity_status[2]:
        raise ValueError("ParityEnergy must be called with a pure parity automaton.")

    is_max = parity_status[1]
    is_odd = parity_status[2]

    current_color = MaxColor(pau) if is_max else MinColor(pau)
    if current_color == -1:
        return BuechiResult()

    # TODO current implementation allocates a LOT of memory
    if (current_color % 2 == 0 and is_odd) or (current_color % 2 == 1 and not is_odd):
        pau_copy = PrunePriority(pau, is_max)
        return ParityEnergy(pau_copy, s0, wup, c0, do_display)
    else:
        # Create a copy of the current automaton
        pau_copy = spot.make_twa_graph(pau, spot.twa_prop_set.all())
        pau_copy.copy_named_properties_of(pau)

        # Recolor and set the acceptance condition to Büchi
        for e in pau_copy.edges():
            e.acc = spot.mark_t({0}) if e.acc.has(current_color) else spot.mark_t()
        pau_copy.set_buchi()
        display(pau_copy.show())

        # Solve in this new automaton
        buchi_res = BuechiEnergy(pau_copy, s0, wup, c0, do_display)
        return buchi_res if buchi_res else ParityEnergy(PrunePriority(pau, is_max),
                                                        s0, wup, c0, do_display)


## Solve an ɷ-regular energy game in a co-Büchi automaton.
#
# @param hoa (HOA automaton): generalized weighted co-büchi automaton as twa_graph
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @param do_display: 0 No information is displayed at all\n
#                    1 Only text is shown\n
#                    2 The (sub)-graphs are shown as well, only works from jupyter
# @return True if there is a (wup, c0) accepting Büchi path in hoa, False otherwise.
def CoBuechiEnergy(hoa: "co-Büchi automaton",
                   s0: "state",
                   wup: "weak upper bound",
                   c0: "initial credit",
                   do_display: "show iterations and info" = 0
                   ):
    # TODO more visual output
    # TODO this actually doesn't work
    # Algorithm:
    # For every accepting set a, set the acceptance of every edge to a
    # if it is not accepting a, or None if it is accepting a.
    # Also set the accepting condition of this new automaton to buchi.
    # Run BuechiEnergy on this new automaton

    # TODO boilerplate, move this function somewhere else
    def print_c(*args, **kwargs):
        if do_display > 0:
            print(*args, **kwargs)
        return

    for col in range(hoa.acc().num_sets()):
        print_c(f"Building Büchi automaton for color {str(col)}")
        co_hoa = spot.make_twa_graph(hoa, spot.twa_prop_set.all())
        co_hoa.copy_named_properties_of(hoa)
        co_hoa.set_buchi()

        for e in co_hoa.edges():
            e.acc = spot.mark_t() if e.acc.has(col) else spot.mark_t({0})
        display(co_hoa.show())

        if BuechiEnergy(co_hoa, s0, wup, c0, do_display):
            # TODO use BuechiResult
            return True
        
    return BuechiResult()


## Solve an ɷ-regular energy game in a Rabin automaton.
#
# @param hoa (HOA automaton): generalized weighted Rabin automaton as twa_graph
# @param p (int): number of acceptance set pairs
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @param do_display: 0 No information is displayed at all\n
#                    1 Only text is shown\n
#                    2 The (sub)-graphs are shown as well, only works from jupyter
# @return True if there is a (wup, c0) accepting Büchi path in hoa, False otherwise.
def RabinEnergy(hoa: "Rabin automaton",
                p: int,
                s0: "state",
                wup: "weak upper bound",
                c0: "initial credit",
                do_display: "show iterations and info" = 0
                ):
    # TODO Try to find a more efficient algorithm
    # Algorithm:
    # for each accepting state pair (f, i), run two tests:
    # - CoBuechiEnergy when considering only f
    # - BuechiEnergy when considering only i
    # Return True if the two tests were successful, else move on to the next pair
    for k in range(p):
        f = 2 * k
        i = 2 * k + 1

        buchi_hoa = spot.make_twa_graph(hoa, spot.twa_prop_set.all())
        buchi_hoa.copy_named_properties_of(hoa)
        buchi_hoa.set_buchi()

        cobuchi_hoa = spot.make_twa_graph(hoa, spot.twa_prop_set.all())
        cobuchi_hoa.copy_named_properties_of(hoa)
        cobuchi_hoa.set_co_buchi()

        for buchi_e in buchi_hoa.edges():
            buchi_e.acc = spot.mark_t({0}) if buchi_e.acc.has(i) else spot.mark_t()
        # cobuchi_hoa will already be ready for Büchi analysis
        for cobuchi_e in cobuchi_hoa.edges():
            cobuchi_e.acc = spot.mark_t() if cobuchi_e.acc.has(f) else spot.mark_t({0})

        if BuechiEnergy(buchi_hoa, s0, wup, c0, do_display) and BuechiEnergy(cobuchi_hoa, s0, wup, c0, do_display):
            # TODO use BuechiResult
            return True

    return BuechiResult()


## Solve an ɷ-regular energy game in an automaton with an acceptance condition of t.
#
# @param hoa (HOA automaton): automaton as twa_graph
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @param do_display: 0 No information is displayed at all\n
#                    1 Only text is shown\n
#                    2 The (sub)-graphs are shown as well, only works from jupyter
# @return True if there is a (wup, c0) accepting Büchi path in hoa, False otherwise.
def TrueEnergy(hoa: "automaton",
               s0: "state",
               wup: "weak upper bound",
               c0: "initial credit",
               do_display: "show iterations and info" = 0
               ):
    # Algorithm: promote every edge to back edge and run BuechiEnergy on the new automaton
    buchi_hoa = spot.make_twa_graph(hoa, spot.twa_prop_set.all())
    buchi_hoa.copy_named_properties_of(hoa)
    buchi_hoa.set_buchi()

    for e in buchi_hoa.edges():
        e.acc = spot.mark_t({0})

    return BuechiEnergy(buchi_hoa, s0, wup, c0, do_display)


## Solve an ɷ-regular energy game in a Büchi automaton.
#
# @param hoa (HOA automaton): generalized weighted büchi automaton as twa_graph
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @param do_display: 0 No information is displayed at all\n
#                    1 Only text is shown\n
#                    2 The (sub)-graphs are shown as well, only works from jupyter
# @return True if there is a (wup, c0) accepting Büchi path in hoa, False otherwise.
def BuechiEnergy(hoa: "Büchi automaton",
                 s0: "state",
                 wup: "weak upper bound",
                 c0: "initial credit",
                 do_display: "show iterations and info" = 0
                 ):
    def print_c(*args, **kwargs):
        if do_display > 0:
            print(*args, **kwargs)
        return

    def display_c(aut, opt=""):
        if do_display > 1:
            display(aut.show(opt))
        return

    def highlight_c(aut: spot.twa_graph,
                    pred: List[List[int]],
                    predColors: List[int] = [1, 2, 3, 4, 5],
                    opt="") -> None:
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

    if not (aut.acc().num_sets() >= 1) and aut.acc().is_generalized_buchi():
        raise RuntimeError("Automaton does not have a generalized buechi acceptance.")

    opts = {"wup": wup, "ic": c0, "s0": aut.get_init_state_number()}

    print_c("Original automaton")
    display_c(hoa, "tsbrg")

    bf = mod_BF_iter(hoa)
    # whole automaton
    # Finds optimal prefix energy for each
    # state, disregarding the colors
    assert s0 == aut.get_init_state_number()
    en, pred = bf.FindMaxEnergy(aut.get_init_state_number(), wup, c0)
    print_c("Prefix energy per state", format_energie_(en),
             "\nCurrent optimal predecessor", format_pred_aut_(aut, pred), sep='\n')
    print_c("""State names are: "state number, max energy"\nOptimal predecessor is highlighted in pink""")
    aut.set_state_names([f"{i},{ei}" for i, ei in enumerate(en)])
    highlight_c(aut, pred, opt="tsbrg")

    ssi = spot.scc_info(hoa)
    # Loop over all SCCs
    for i in range(ssi.scc_count()):
        if not ssi.is_accepting_scc(i):
            continue
        __bench_stats__["n_scc"] += 1
        print_c("Checking SCC", i)
        aut_degen, acc_edge, rename = degen_counting(hoa, ssi, i)
        print_c(f"Degeneralized SCC has: {aut_degen.num_states()} states, {aut_degen.num_edges()} edges and {len(acc_edge)} back-edges.")

        revrename = {v: k for k, v in rename.items()}

        # renaming of states
        names = ["" for _ in range(len(rename))]
        for old, new in rename.items():
            names[new] = str(old)
        names = names * hoa.get_acceptance().used_sets().max_set()
        for i in range(len(names)):
            names[i] = names[i]+":"+str(i//len(rename))
        aut_degen.set_state_names(names)

        print_c(f"Current SCC with: {aut_degen.num_states()} states and {len(acc_edge)} back-edges")
        print_c("""Associating states in the original automaton to the corresponding states in lvl 0 of the degeneralised SCC""",
                rename, sep="\n")
        # Update names
        aut_degen.set_state_names([f"{i}" for i in range(aut_degen.num_states())])
        display_c(aut_degen, "tsbrg")

        # current degeneralized SCC
        bf2 = mod_BF_iter(aut_degen)

        # Loop over each (accepting) backedge
        # of the degeneralized current SCC
        for be_num in acc_edge:
            __bench_stats__["n_backedges"] += 1
            be = aut_degen.edge_storage(be_num)
            print_c("Analysing backedge " + names[be.src],
                    "->", names[be.dst] + ".")

            start_energy = en[revrename[be.dst]]
            if start_energy < 0:
                continue
            print_c("We start with " + str(start_energy)
                    + " energy in state " + names[be.dst] + ".")

            # look from backedge->destination
            (en3, pred3) = bf2.FindMaxEnergy(be.dst, wup, start_energy)
            print_c("Energy starting in backedge dst", format_energie_(en3),
                    "Corresponding predecessors", format_pred_aut_(aut_degen, pred3), sep="\n")
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
                # restart with the new energy
                if new_energy < 0:
                    continue
                print_c("We restart with " + str(new_energy)
                        + " energy in state " + names[be.dst] + ".")

                # look again from backedge->destination but with lower start energy
                en3, pred3 = bf2.FindMaxEnergy(be.dst, wup, new_energy)
                print_c(en3, pred3)
                if en3[be.src] >= 0:
                    even_newer_energy = min(en3[be.src] + spot.get_weight(aut_degen, be_num), wup)
                else:
                    even_newer_energy = -1
                print_c("We arrived with " + str(even_newer_energy)
                        + " energy in state " + names[be.dst] + ".")
                if even_newer_energy >= new_energy:
                    print_c("We found a non-negative loop using edge", names[be.src],
                            "->", names[be.dst] + " in the second iteration.")
                    highlight_c(aut_degen, pred3, opt="tsbrg")
                    return BuechiResult(aut, aut_degen, rename, opts, en, pred, be_num, en3, pred3, -1, None, None)
                else:
                    for node, energy in enumerate(en3):
                        if energy == wup:
                            print_c("we should check also from " + str(names[node])+".")
                            en4, pred4 = bf2.FindMaxEnergy(node, wup, wup)
                            print_c(en4, pred4)
                            if en4[be.src] >= 0:
                                newest_energy = min(en4[be.src] + spot.get_weight(aut_degen, be_num), wup)
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
