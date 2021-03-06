# This file contains our main contributions, notably
# The main algorithm, algorithm 1 in the paper
# The helper algorithms for energy computations,
# subsumed in algorithm 2 in the paper

from dataclasses import dataclass
import spot, buddy



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
    print(f"Degeneralized SCC has: {aut_degen.num_states()} states, {aut_degen.num_edges()} edges and {len(acc_edge)} back-edges.")
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
        self.Pred_ = array('Q', self.N_*[0])
        self.isWaiting_ = array('b', self.N_*[False])
        self.Waiting_ = array('B')
        # For Loop searching
        #-1: Postfix of a loop, 0: "Free", 1: the current loop, 2: old loop or postfix
        self.onLoop_ = array('b', self.N_*[0])
        # From initial state)
        self.E_[self.s0_] = self.c0_
        self.isWaiting_[self.s0_] = True
        self.Waiting_.append(self.s0_)
        # Fixpoint attained?
        self.isFixPoint_ = False

    # Propagate the energy along e
    # Returns if energy of dst was changed
    def prop_(self, en:"edge number", opt:bool):
        """
        Propagates the energy along an edge
        :param en: Edge number
        :param opt: Whether the optimal energy for dst is chosen or energy is always propagated
        :return: Whether the energy of dst changed
        """
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

    def pumpLoop_(self, s:"state"):
        """
        Helper to pump the simple positive loop containing s
        :param s: initial state
        """

        for (sprime, _) in self.loop_(s):
            self.E_[sprime] = -1
            self.onLoop_[sprime] = 2 #Mark it as old
            # All of these might get their values changed
            self.mark_(sprime)
        self.E_[self.g_.edge_storage(self.Pred_[s]).src] = self.wup_

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

    def pumpAll(self):
        """
        Pump all (energy positive) loops of the current iteration
        :return:
        """
        # Reset who is on a loop
        self.onLoop_ = array('b', self.N_*[0])

        # Check for each state if loop candidate
        for s in range(self.N_):
            if not self.isWaiting_[s]:
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

    def BF1(self):
        """
        Perform one round of modified, optimised Bellman-Ford
        :return:
        """

        isWaiting2_ = array('b', self.N_ * [False])
        Waiting2_ = array('B')
        #Swap
        #self.isWaiting_, isWaiting2_ = isWaiting2_, self.isWaiting_
        #self.Waiting_, Waiting2_ = Waiting2_, self.Waiting_

        for _ in range(self.N_ - 1):
            if not self.isWaiting_:
                break  # Early exit
            self.isFixPoint_ = True

            isWaiting2_ = array('b', self.N_ * [False])  #There is no "fill" for a base array
            while self.Waiting_:
                s = self.Waiting_.pop()
                for e in self.g_.out(s):
                    en = self.g_.edge_number(e)
                    changed = self.prop_(en, True)
                    if changed:
                        self.isFixPoint_ = False
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

        while not self.isFixPoint_:
            self.isFixPoint_ = True
            self.BF1()  # One round of (modified) Bellman-Ford
            yield self.E_, self.Pred_
            self.pumpAll()  # Who have got to pump it up!
            yield self.E_, self.Pred_

    def FindMaxEnergyGen(self, s0:"state", wup:"weak upper bound", c0:"Initial credit"):
        return self.FindMaxEnergy_(s0, wup, c0)

    def FindMaxEnergy(self, s0:"state", wup:"weak upper bound", c0:"Initial credit"):
        for (En, Pred) in self.FindMaxEnergy_(s0, wup, c0):
            continue
        return (En, Pred)


# Whole picture
# This is algorithm 1

def BuechiEnergy(hoa:"HOA automaton", s0:"state", wup:"weak upper bound", c0:"initial credit", do_display:"show iterations"=False):
    """_summary_

    Args:
        hoa (HOA automaton): generalized weighted b??chi automaton,  filename or twa_graph
        s0 (state): initial state
        wup (weak upper bound):
        c0 (initial credit):
    """

    if isinstance(hoa, str):
        aut = spot.automaton(hoa)
    else:
        aut = hoa
    if do_display:
        print("Original automaton")
        display(aut.show("tsbrg"))
    bf = mod_BF_iter(aut)
    # whole automaton
    # Finds optimal prefix energy for each
    # state, disregarding the colors
    en, pred = bf.FindMaxEnergy(aut.get_init_state_number(), wup, c0)
    if do_display:
        print(f"Prefix energy per state\n{en}\nCurrent optimal predescessor\n{pred}")
        print("""State names are: "state number, max energy"\nOptimal predescessor is highlighted in pink""");
        aut.set_state_names([f"{i},{ei}" for i, ei in enumerate(en)])
        aut.highlight_edges([i for i in pred if i!=0], 1)
        display(aut.show("tsbrg"))

    ssi = spot.scc_info(aut)
    # Loop over all SCCs
    for i in range(ssi.scc_count()):
        if not ssi.is_accepting_scc(i):
            continue
        print("Checking SCC", i)
        aut_degen, acc_edge, rename = degen_counting(aut, ssi, i)
        revrename = {v: k for k, v in rename.items()}

        # renaming of states
        names = ["" for _ in range(len(rename))]
        for old, new in rename.items():
            names[new] = str(old)
        names = names * aut.get_acceptance().used_sets().max_set()
        for i in range(len(names)):
            names[i] = names[i]+":"+str(i//len(rename))
        aut_degen.set_state_names(names)
        if do_display:
            print(f"Current SCC with: {aut_degen.num_states()} states and {len(acc_edge)} back-edges")
            print(rename)
            display(aut_degen.show("tsbrg"))

        # current degeneralized SCC
        bf2 = mod_BF_iter(aut_degen)

        # Loop over each (accepting) backedge
        # of the degeneralized current SCC
        for be_num in acc_edge:
            be = aut_degen.edge_storage(be_num)
            print("Analysing backedge "+ names[be.src],"->", names[be.dst]+".")

            start_energy = en[revrename[be.dst]]
            if start_energy < 0:
                continue
            print("We start with "+ str(start_energy) + " energy in state "+names[be.dst] + ".")

            # look from backedge->destination
            (en3, pred3) = bf2.FindMaxEnergy(be.dst, wup, start_energy)
            print(en3, pred3)
            if en3[be.src] >= 0:
                new_energy = min(en3[be.src]+spot.get_weight(aut_degen, be_num), wup)
            else:
                new_energy = -1
            if new_energy >= start_energy:
                print("We found a non-negative loop using edge", names[be.src],
                      "->", names[be.dst]+" directly.")
                return True
            else:
                #restart with the new energy
                if new_energy < 0:
                    continue
                print("We restart with "+ str(new_energy) + " energy in state "+names[be.dst] + ".")

                # look again from backedge->destination but with lower start energy
                en3, pred3 = bf2.FindMaxEnergy(be.dst, wup, new_energy)
                print(en3, pred3)
                if en3[be.src] >= 0:
                    even_newer_energy = min(en3[be.src]+spot.get_weight(aut_degen, be_num), wup)
                else:
                    even_newer_energy = -1
                if  even_newer_energy >= new_energy:
                    print("We found a non-negative loop using edge", names[be.src],
                          "->", names[be.dst]+" in the second iteration.")
                    return True
    print("No feasible B??chi run detected!")
    return False
