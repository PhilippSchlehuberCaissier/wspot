## @package WBA_FW
# Generalized Floyd-Warshall algorithm implementation for automata weighted on a semiring (Mohri, 2002).

import spot

from buechi import BuechiResult
from energy import WUP

## Generalized Floyd-Warshall on automata weighted on a semiring sr.
# @param aut (HOA automaton): generalized weighted co-büchi automaton as twa_graph
# @param sr (class): a semiring class
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @param check (matrix of sr -> None): optional function to call after a pass in the triple loop
# @return True if there is a (wup, c0) accepting Büchi path in hoa, False otherwise.
def FWhoa(aut: "HOA automaton",
          sr,
          s0: int,
          check=lambda M, i, j: None
          ) -> BuechiResult:
    if isinstance(aut, str):
        hoa = spot.automaton(aut)
    else:
        hoa = aut

    # Initial matrix filling
    V = hoa.num_states()
    M = [
        [sr.zero() for _ in range(V)] for _ in range(V)]

    # Set diagonal
    for v in range(V):
        M[v][v] = sr.one()

    # Set edges (eventually overwriting the diagonal)
    for e in hoa.edges():
        weight = spot.get_weight(hoa, e)
        M[e.src][e.dst] = sr.transition_to_sr(e, weight)

    # The usual triple loop
    for k in range(V):
        for i in range(V):
            for j in range(V):
                M[i][j] = M[i][j] + (M[i][k] * M[k][j])
                # TODO do we need to return something?
                res = check(M, i, j)
                if res:
                    return res

    for i in range(V):
        print(f"From {i}")
        for j in range(V):
            print(str(M[i][j]))
        print("")
