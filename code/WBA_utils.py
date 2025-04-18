## @package WBA_utils
# Utility functions for solving energy problems in weighted Büchi automata.
#
# This file contains our main contributions, notably\n
# The main algorithm, algorithm 1 in the paper\n
# The helper algorithms for energy computations,\n
# subsumed in algorithm 2 in the paper\n

from dataclasses import dataclass
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



## Class allowing to run iterations of the modified bellman-ford algorithm.
#
# Holds all necessary variables and member functions described in algorithm 2.
# Most of them have additional optimiations    
class mod_BF_iter:
    def __init__(self, g:spot.twa_graph):
        self.g_ = g

    def init(self):
        self.N_ = self.g_.num_states()
        # Base values
        self.E_ = array('q', self.N_*[-1])
        self.Pred_ = array('Q', self.N_*[0])
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
    
    ## Propagates the energy along an edge
    # @param en (int): Edge number
    # @param opt (bool): Whether the optimal energy for dst is chosen or energy is always propagated
    # @return Whether the energy of dst changed
    def prop_(self, en:"edge number", opt:bool):
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
            self.Pred_[dst] = en
            return en_prime != en_dst
        return False


        
    ## Mark state s as waiting
    # @param s (int): state to mark    
    def mark_(self, s:"state"):
        if not self.isWaiting_[s]:
            self.isWaiting_[s] = True
            self.Waiting_.append(s)


    ## Helper function to iterate over loops
    # Must be constructed with a state on a cycle.\n
    # Will eventually raise an error otherwise
    # or loop indefinitely otherwise
    # @param si (int): initial state
    # @return yields a state till done    
    def loop_(self, si:"init state"):
        from collections import deque
        s = si
        loopItems = deque()
        def pred_(s):
            en = self.Pred_[s]
            assert en != 0, "No valid Predecessor!"
            return en, self.g_.edge_storage(en)
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


        
    ## Helper to pump the simple positive loop containing s
    # @param s (int): initial state    
    def pumpLoop_(self, s:"state"):
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


        
    # State s is a candidate for a loop state that
    # needs to be pumped. It could however
    # be either on the loop, or the postfix or
    # the postfix of a loop already pumped
    # @param s (int): State to be checked
    def checkLoop(self, s:"state"):
        sprime = s

        while self.onLoop_[sprime] == 0:
            self.onLoop_[sprime] = 1
            assert self.Pred_[sprime] != 0
            sprime = self.g_.edge_storage(self.Pred_[sprime]).src

        if self.onLoop_[sprime] == 1:
            # Found a new simple positive loop
            self.pumpLoop_(sprime)
        # Mark the postfix if necessary
        sprime = s
        while self.onLoop_[sprime] == 1:
            self.onLoop_[sprime] = 2
            sprime = self.g_.edge_storage(self.Pred_[sprime]).src
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
                continue # State belongs to some other loop or postfix
            # Check if energy can increase
            # Todo Code duplication :(
            en = self.Pred_[s]
            if en == 0:
                # Unreachable
                if s != self.s0_:
                    assert self.E_[s] == -1
                continue
            e = self.g_.edge_storage(en)

            src = e.src
            dst = e.dst
            ew = spot.get_weight(self.g_, en)
            en_src = self.E_[src]
            en_dst = self.E_[dst]

            en_prime = min(self.wup_, en_src + ew)
            if (en_prime > en_dst):
                #Loop candidate
                self.checkLoop(dst)

    ## Perform one round of modified, optimised Bellman-Ford
    def BF1(self):
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


    ## Computes for each state the maximal energy for which it can be reached from s0 with initial credit c0 given the weak upper bound wup
    # @param s0 (int): the initial state
    # @param wup (int): the weak upper bound
    # @param c0 (int): the initial credit
    # :param asGen: If set to true, it will yield the current energy levels and predecessors at every iteration
    def FindMaxEnergy_(self, s0:"state", wup:"weak upper bound", c0:"Initial credit"):
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


# Whole picture
# This is algorithm 1

## Solve an ɷ-regular energy game.
#
# @param hoa (HOA automaton): automaton with one of the following acceptance conditions: parity, Büchi. filename (cast to twa_graph) or twa_graph
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @param do_display: 0 No information is displayed at all\n
#                    1 Only text is shown\n
#                    2 The (sub)-graphs are shown as well, only works from jupyter
# @return True if there is a (wup, c0) accepting path in hoa, False otherwise.
def OmegaEnergy(hoa: "HOA automaton",
                s0: "state",
                wup: "weak upper bound",
                c0: "initial credit",
                do_display: "show iterations and info" = 0
                ):
    
    # TODO cases where hoa is not a Büchi-accepting automaton (parity, co-Büchi, etc)
    
    # We manipulate a twa_graph from now on
    if isinstance(hoa, str):
        aut = spot.automaton(hoa)
    else:
        aut = hoa

    acc_cond = aut.acc()

    # Empty automaton
    if aut.num_states() == 0:
        return False
    
    # Büchi (can be generalized)
    if acc_cond.is_generalized_buchi():
        return BuechiEnergy(aut, s0, wup, c0, do_display)
    
    # Parity
    # Implements the algorithm presented in Section 7
    # TODO test this
    if acc_cond.is_parity()[0]:
        return ParityEnergy(aut, s0, wup, c0, do_display)

    # TODO other automata types
    return False

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
        return False
    
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
        if BuechiEnergy(pau_copy, s0, wup, c0, do_display):
            return True
        else:
            return ParityEnergy(PrunePriority(pau, is_max),
                                s0, wup, c0, do_display)


## Solve an ɷ-regular energy game in a Büchi automaton.
#
# @param bau (HOA automaton): generalized weighted büchi automaton as twa_graph
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @param do_display: 0 No information is displayed at all\n
#                    1 Only text is shown\n
#                    2 The (sub)-graphs are shown as well, only works from jupyter
# @return True if there is a (wup, c0) accepting Büchi path in bau, False otherwise.
def BuechiEnergy(bau: "Büchi automaton",
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

    def highlight_c(aut, pred, opt=""):
        if do_display > 1:
            aut.highlight_edges([i for i in pred if i != 0], 1)
        display_c(aut, opt)
        
    print_c("Original automaton")
    display_c(bau, "tsbrg")

    bf = mod_BF_iter(bau)
    # whole automaton
    # Finds optimal prefix energy for each
    # state, disregarding the colors
    en, pred = bf.FindMaxEnergy(bau.get_init_state_number(), wup, c0)
    print_c(f"Prefix energy per state\n{en}\nCurrent optimal predescessor\n{pred}")
    print_c("""State names are: "state number, max energy"\nOptimal predescessor is highlighted in pink""");
    bau.set_state_names([f"{i},{ei}" for i, ei in enumerate(en)])
    highlight_c(bau, pred, "tsbrg")

    ssi = spot.scc_info(bau)
    # Loop over all SCCs
    for i in range(ssi.scc_count()):
        if not ssi.is_accepting_scc(i):
            continue
        __bench_stats__["n_scc"] += 1
        print_c("Checking SCC", i)
        aut_degen, acc_edge, rename = degen_counting(bau, ssi, i)
        print_c(f"Degeneralized SCC has: {aut_degen.num_states()} states, {aut_degen.num_edges()} edges and {len(acc_edge)} back-edges.")

        revrename = {v: k for k, v in rename.items()}

        # renaming of states
        names = ["" for _ in range(len(rename))]
        for old, new in rename.items():
            names[new] = str(old)
        names = names * bau.get_acceptance().used_sets().max_set()
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
                highlight_c(aut_degen, pred3, "tsbrg")
                return True
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
                if  even_newer_energy >= new_energy:
                    print_c("We found a non-negative loop using edge", names[be.src],
                            "->", names[be.dst]+" in the second iteration.")
                    highlight_c(aut_degen, pred3, "tsbrg")
                    return True
    print_c("No feasible Büchi run detected!")
    return False
