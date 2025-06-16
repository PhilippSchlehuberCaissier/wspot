from array import array
from copy import deepcopy
import spot

__bench_stats__ = {"n_backedges": 0, "n_bf_iter": 0, "n_scc": 0,
                   "n_pump_loop": 0, "n_propagate": 0}


def reset_stats():
    for k in __bench_stats__.keys():
        __bench_stats__[k] = 0


def get_stats():
    return __bench_stats__


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
