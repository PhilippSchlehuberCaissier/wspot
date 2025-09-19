import sys
sys.path.append("/usr/local/lib/python3.12/site-packages")

import spot, buddy
import random
import time
from math import ceil

T = buddy.bddtrue

def maxStretchEnergy(N: int, NM: int):
    """
    Computes the maximal obtainable energy for a circle element of size \a N
    with an incremental energy of \a NM
    Args:
        N:
        NM:

    Returns:

    """
    assert N >= 3

    if N == 3:
        return 2*NM
    else:
        return (N-1)*NM+N*maxStretchEnergy(N-1,NM)

def cstr_(aut:"spot.twa_graph", sState: int, oState: int, N: int, NM: int):
    """
    Adds a circle component between \a sState and \a oState in the
    automauton \a aut of length \a N with an incremental energy of \a NM
    """

    assert N <= NM

    # A circle element of size N adds 2*N states
    s0 = aut.new_states(2*N)
    newStates = list(range(s0, s0+2*N))

    aut.new_edge(sState, newStates[0], T)

    # Interconnections
    for i in range(N):
        en = 1
        for j in range(N-1, i+1, -1):
            t = aut.new_edge(newStates[i], newStates[j], T)
            spot.set_weight(aut, t, en)
            en += 1

    # Outer ring
    for i in range(N):
        aut.new_edge(newStates[i], newStates[N+i], T) # Unweighted
        # Positive
        if i == 0:
            t = aut.new_edge(newStates[-1], newStates[i], T)
            spot.set_weight(aut, t, -maxStretchEnergy(N,NM)-1)
        else:
            t = aut.new_edge(newStates[N+i-1], newStates[i], T)
            spot.set_weight(aut, t, NM)

    # Recurse
    if N > 3:
        for i in range(N):
            cstr_(aut, newStates[i], newStates[N+i], N-1, NM)

    # Outgoing
    aut.new_edge(newStates[-1], oState, T)

def permuteAutWith(aut:spot.twa_graph, seed: int = 1234567):
    """
    Create a randomize version of \a aut (internal transition order is permuted) using
    \a seed
    Args:
        aut:
        seed:

    Returns:

    """
    random.seed(seed)

    autP = spot.make_twa_graph(aut.get_dict())
    autP.copy_ap_of(aut)
    autP.copy_acceptance_of(aut)
    autP.copy_named_properties_of(aut)

    autP.new_states(aut.num_states())

    for s in range(aut.num_states()):
        allT = [t for t in aut.out(s)]
        random.shuffle(allT)
        for t in allT:
            autP.new_edge(t.src, t.dst, t.cond, t.acc)

    return autP


class DevilBuilder:
    def __init__(self, k):
        assert k > 2
        self.k = k
        self.NM = 2*k
        self.wup = maxStretchEnergy(k, self.NM)
        self.name = "circling"
        self.output = "../tests/devil_auto.hoa"

    def update(self, k):
        self.__init__(k)

    def build(self):
        aut = spot.make_twa_graph()
        s = list(range(aut.new_states(3), 3))

        t = aut.new_edge(s[0], s[1], T)
        spot.set_weight(aut, t, 2*maxStretchEnergy(self.k, self.NM))

        cstr_(aut, s[1], s[2], self.k, self.NM)

        print(f"Building circling automaton with circles of size {self.k} and increasing energy of {self.NM} at {self.output}")
        # Original automaton
        with open(self.output, 'w') as f:
            print(aut.to_str("hoa"), file=f)

        # autP = permuteAutWith(aut, ceil(time.time()))
        # print("Permuted automaton")
        # print(autP.to_str("hoa"))


# if __name__ == "__main__":
#     N = 4
#     NM = 6

#     aut = spot.make_twa_graph()
#     s = list(range(aut.new_states(3), 3))

#     t = aut.new_edge(s[0], s[1], T)
#     spot.set_weight(aut, t, 2*maxStretchEnergy(N, NM))

#     cstr_(aut, s[1], s[2], N, NM)

#     output = "../tests/devil.hoa"
#     # Original automaton
#     with open(output, 'w') as f:
#         print(aut.to_str("hoa"), file=f)

#     autP = permuteAutWith(aut, ceil(time.time()))
#     print("Permuted automaton")
#     print(autP.to_str("hoa"))
