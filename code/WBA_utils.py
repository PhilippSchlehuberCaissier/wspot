# This file contains our main contributions, notably
# The main algorithm, algorithm 1 in the paper
# The helper algorithms for energy computations,
# subsumed in algorithm 2 in the paper

from dataclasses import dataclass
import spot, buddy
from copy import deepcopy as deepcopy
import array
from collections import deque

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
        self.Pred_ = [[] for _ in range(self.N_)]
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
            if en not in self.Pred_[dst]:
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
            assert self.Pred_[s], "No valid Predecessor!"
            en = self.Pred_[s][-1]
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
            assert self.Pred_[sprime]
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
            en = self.Pred_[s][-1]
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

    
    def trace(self, src, dst, c0, names):
        next_pred = [0] * self.N_
        
        #grabbing the initial path 
        path = deque()

        en = self.Pred_[dst][0] 
        e = self.g_.edge_storage(en)
        while e.src != src:
            path.append(en)
            next_pred[e.dst] += 1
            en = self.Pred_[e.src][0]
            e = self.g_.edge_storage(en)
        path.append(en)
        next_pred[e.dst] += 1

        #building the trace
        energy = deque()
        energy.append((src, c0))

        res = []
        infix = []
        cycle = []

        while path:
            en = path.pop()
            e = self.g_.edge_storage(en)
            ew = spot.get_weight(self.g_, en)
            print("taking edge", names[e.src], names[e.dst])

            (_, v) = energy[-1]
            new_energy = v + ew

            print(energy)
            
            if new_energy < 0:
                #going back to a point that has a cycle
                path.append(en)
                while next_pred[e.src] >= len(self.Pred_[e.src]):
                    print("checking edge ", names[e.src], names[e.dst], next_pred[e.src])
                    if not infix:
                        break
                    en = infix.pop()
                    e = self.g_.edge_storage(en)
                    #here
                    next_pred[e.dst] -= 1
                    energy.pop()
                    path.append(en)
                if energy :
                    energy.pop()
                
                #getting cycles
                print("suffix of cycle ", names[e.src],"-", names[e.dst], "next pred", next_pred[e.src])
                current = self.Pred_[e.src][next_pred[e.src]]
                c = self.g_.edge_storage(current)
                print("first state of cycle ", c.src ,"-", c.dst)
                while c.src != e.src:
                    cycle.append(current)
                    c = self.g_.edge_storage(current)
                    current = self.Pred_[c.src][next_pred[c.src]]
                    next_pred[c.src] += 1
                next_pred[c.src] += 1
                
                cycle.reverse()
                res.append((infix, cycle))
                
                #PUMP AND ADD ENERGY OF CYCLE STATES
                new_energy = self.wup_
                for tn in cycle:
                    tw = spot.get_weight(self.g_, tn)
                    new_energy = min(self.wup_, new_energy + tw)
                
                (infix, cycle) = ([], [])
                #next_pred[e.dst] += 1

                energy.append((e.src, new_energy))
                infix.append(en)

            else:
                energy.append((e.dst, new_energy))
                infix.append(en)
                

        if infix:
            res.append((infix, []))

        #DISPLAY

        print(res)
        for (infix, cycle) in res:
            print("INFIX")
            for ei in infix:
                e = self.g_.edge_storage(ei)
                print(names[e.src], " to ", names[e.dst], " weight : ", spot.get_weight(self.g_, ei))
            print("CYCLE")
            for ec in cycle:
                e = self.g_.edge_storage(ec)
                print(names[e.src], " to ", names[e.dst], " weight : ", spot.get_weight(self.g_, ec))
            
        return res

                

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


# Whole picture
# This is algorithm 1

def BuechiEnergy(hoa:"HOA automaton", s0:"state", wup:"weak upper bound", c0:"initial credit", do_display:"show iterations and info"=0):
    """_summary_

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

    def highlight_c(aut, pred, opt=""):
        if do_display > 1:
            to_highlight = []
            for subpred in pred:
                for e in subpred:
                    to_highlight.append(e)
            aut.highlight_edges(to_highlight, 1)
        display_c(aut, opt)

    if isinstance(hoa, str):
        aut = spot.automaton(hoa)
    else:
        aut = hoa

    print_c("Original automaton")
    display_c(aut, "tsbrg")

    bf = mod_BF_iter(aut)
    # whole automaton
    # Finds optimal prefix energy for each
    # state, disregarding the colors
    en, pred = bf.FindMaxEnergy(aut.get_init_state_number(), wup, c0)
    print_c(f"Prefix energy per state\n{en}\nCurrent optimal predescessor\n{pred}")
    print_c("""State names are: "state number, max energy"\nOptimal predescessor is highlighted in pink""");
    aut.set_state_names([f"{i},{ei}" for i, ei in enumerate(en)])
    highlight_c(aut, pred, "tsbrg")

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
                new_energy = min(en3[be.src] + spot.get_weight(aut_degen, be_num), wup)
            else:
                new_energy = -1
            if new_energy >= start_energy:
                print_c("We found a non-negative loop using edge", names[be.src],
                        "->", names[be.dst]+" directly.")
                highlight_c(aut_degen, pred3, "tsbrg")
                #bf2.trace1(be_num, names)
                bf2.trace(bf2.s0_, be.src, bf2.c0_, names)
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
                    #bf2.trace1(be_num, names)
                    bf2.trace(bf2.s0_, be.src, bf2.c0_, names)
                    return True
    print_c("No feasible Büchi run detected!")
    return False
