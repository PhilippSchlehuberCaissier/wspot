from typing import List
from buechi import BuechiResult
import BF1
from BF1 import mod_BF_iter
from energy import EnergyFunction

import spot

import ipython_utils as ipy
import WBA_FW as wf
ipy_utils = ipy.IPythonUtils()


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


## Remove a specified color (acceptance set) in an automaton.
# @param aut (HOA automaton): automaton as twa_graph
# @param col (int): the color to remove
# @return a new automaton without transitions accepting col
def RemoveColor(aut: "HOA automaton",
                col: int):
    aut_new = spot.make_twa_graph(aut, spot.twa_prop_set.all())
    aut_new.copy_named_properties_of(aut)
    for s in range(aut_new.num_states()):
        it = aut_new.out_iteraser(s)
        while it:
            e = it.current()
            if e.acc.has(col):
                it.erase()
            else:
                it.advance()

    return aut_new

    
## Remove every transition with maximal priority in an automaton.
# TODO refactor with RemoveColor
# @param aut (HOA automaton): automaton as twa_graph
# @param is_max (bool): is this a parity-max automaton? (defaults to True for non-parity automata)
# @return a new automaton without highest-priority transitions, or an empty automaton if there is only one color
def PrunePriority(aut: "HOA automaton",
                  is_max: bool = True):
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
def MaxColor(aut: "HOA automaton") -> int:
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
def MinColor(aut: "HOA automaton") -> int:
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
                 s0: int,
                 wup: int,
                 c0: int
                 ) -> BuechiResult:
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


## DEPRECATED
# Solve an ɷ-regular energy game in a co-Büchi automaton using a naive algorithm.
def NaiveCoBuechi(hoa, s0, wup, c0):
    # For every color col, prune every edge accepting col, promote to backedge (color 0) every other edge and solve for Büchi
    bf = mod_BF_iter(hoa)
    assert s0 == hoa.get_init_state_number()

    en, pred = bf.FindMaxEnergy(hoa.get_init_state_number(), wup, c0)
    ipy_utils.print_c("Prefix energy per state", format_energie_(en),
             "\nCurrent optimal predecessor", format_pred_aut_(hoa, pred), sep='\n')
    ipy_utils.print_c("""State names are: "state number, max energy"\nOptimal predecessor is highlighted in pink""")
    hoa.set_state_names([f"{i},{ei}" for i, ei in enumerate(en)])

    ipy_utils.highlight_c(hoa, pred, opt="tsbrg")
    for col in range(hoa.acc().num_sets()):
        ipy_utils.print_c(f"Building Büchi automaton for color {str(col)}")
        co_hoa = spot.make_twa_graph(hoa, spot.twa_prop_set.all())
        co_hoa.copy_named_properties_of(hoa)
        co_hoa.set_buchi()

        # TODO temporary, edges to be removed are marked with no acceptance sets
        for e in co_hoa.edges():
            e.acc = spot.mark_t() if e.acc.has(col) else spot.mark_t({0})

        # TODO can the edges be remove directly in the previous loop?
        for i in range(co_hoa.num_states()):
            it = co_hoa.out_iteraser(i)
            while it:
                e = it.current()
                if e.acc == spot.mark_t():
                    it.erase()
                else:
                    it.advance()

        ipy_utils.display_c(co_hoa)
        res = BuechiEnergy(co_hoa, s0, wup, c0, en, pred)
        if res:
            return res


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
                   s0: int,
                   wup: int,
                   c0: int
                   ) -> BuechiResult:
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


# ## Solve an ɷ-regular energy game in a co-Büchi automaton using Floyd-Warshall on energy functions.
# def CoBuechi_FW(aut: "co-Büchi automaton",
#                 s0: int,
#                 wup: int,
#                 c0: int
#                 ) -> BuechiResult:
#     if isinstance(aut, str):
#         hoa = spot.automaton(aut)
#     else:
#         hoa = aut

#     for col in range(hoa.acc().num_sets()):
#         ipy_utils.print_c(f"Examining color {str(col)}")
#         sub_hoa = RemoveColor(hoa, col)

#         V = sub_hoa.num_states()
#         M = [
#             [EnergyFunction.zero(0, wup) for _ in range(V)] for _ in range(V)]

#         for e in sub_hoa.edges():
#             # 3 cases:
#             # edge weight is 0 -> identity
#             # edge weight is > 0 -> increasing function + constant
#             # edge weight is < 0 -> undefined + increasing
#             weight = spot.get_weight(sub_hoa, e)
#             segs = []
#             if weight > 0:
#                 segs = [
#                     EnergySegment.incr(0, wup - weight, e.src, weight),
#                     EnergySegment.const(wup - weight, wup, e.src, wup)
#                     ]
#             elif weight < 0:
#                 segs = [
#                     EnergySegment.const(0, -weight, e.src, -1),
#                     EnergySegment.incr(-weight, wup, e.src, weight)
#                     ]
#             else:
#                 segs = [
#                     EnergySegment.one(0, wup, e.src)
#                     ]
#             f = EnergyFunction(segs)
#             M[e.src][e.dst] = EnergyFunction.clean(f)
#         for v in range(V):
#             if M[v][v] == EnergyFunction.zero(0, wup):
#                 M[v][v] = EnergyFunction.clean(EnergyFunction(
#                     [
#                         EnergySegment.one(0, wup, None)
#                     ]))
#         for k in range(V):
#             for i in range(V):
#                 for j in range(V):
#                     M[i][j] = M[i][j] + (M[i][k] * M[k][j]) if M[i][k] != EnergyFunction.zero(0, wup) else M[i][j]
#                     if i == j:
#                         if M[i][j].is_above_one:
#                             ipy_utils.print_c(f"There is a positive loop starting from {i}! ({M[i][j]})")
#                             # TODO return BuechiResult
#                             return True

#         # Check the diagonal
#         for k in range(V):
#             fun = M[k][k]
#             if fun.is_above_one:
#                 ipy_utils.print_c(f"There is a positive loop starting from {k}! ({fun})")
#                 # TODO return BuechiResult
#                 return True

#     for i in range(V):
#         print(f"==== color {i} ====")
#         for j in range(V):
#             print(f"to {j}: {M[i][j]}")
#         print("\n")
#     ipy_utils.print_c("There is no positive loop")
#     return BuechiResult()


## Solve an ɷ-regular energy game in a co-Büchi automaton using Floyd-Warshall on energy functions.
def CoBuechi_FW_new(aut: "co-Büchi automaton",
                    s0: int,
                    wup: int,
                    c0: int
                    ) -> BuechiResult:
    if isinstance(aut, str):
        hoa = spot.automaton(aut)
    else:
        hoa = aut

    def diag_is_above_one(M, i, j):
        if i == j:
            if M[i][j].is_above_one:
                ipy_utils.print_c(f"There is a positive loop starting from {i}! ({M[i][j]})")
                # TODO return BuechiResult
                return True

    ncolors = hoa.acc().num_sets()
    for col in range(ncolors):
        ipy_utils.print_c(f"Examining color {str(col)}")
        sub_hoa = RemoveColor(hoa, col)

        res = wf.FWhoa(sub_hoa, EnergyFunction, diag_is_above_one)
        # TODO return a BuechiResult and not the result matrix
        if res:
            return res

    ipy_utils.print_c("There is no positive loop")
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
                s0: int,
                wup: int,
                c0: int
                ) -> BuechiResult:
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


## Solve an ɷ-regular energy game in an automaton with an acceptance condition of t using the Floyd-Warshall algorithm.
#
# @param hoa (HOA automaton): automaton as twa_graph
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @return True if there is a (wup, c0) accepting Büchi path in hoa, False otherwise.
def TrueEnergy(hoa: "automaton",
               s0: int,
               wup: int,
               c0: int
               ) -> BuechiResult:
    # Algorithm: promote every edge to back edge and run BuechiEnergy on the new automaton
    return CoBuechi_FW_new(hoa, s0, wup, c0)
    # buchi_hoa = spot.make_twa_graph(hoa, spot.twa_prop_set.all())
    # buchi_hoa.copy_named_properties_of(hoa)
    # buchi_hoa.set_buchi()

    # for e in buchi_hoa.edges():
    #     e.acc = spot.mark_t({0})

    # return BuechiEnergy(buchi_hoa, s0, wup, c0, None, None)


## Solve an ɷ-regular energy game in a Büchi automaton.
#
# @param aut (HOA automaton): generalized weighted büchi automaton as twa_graph
# @param s0 (int): initial state
# @param wup (int): weak upper bound
# @param c0 (int): initial credit
# @param en: array containing the prefix energies for each state.
# If it is not provided, the prefixes are calculated instead.
# @param pred: array containing the optimal predecessors for each state
# @return True if there is a (wup, c0) accepting Büchi path in hoa, False otherwise.
def BuechiEnergy(aut,
                 s0: int,
                 wup: int,
                 c0: int,
                 en: "prefix energies array" = None,
                 pred: "optimal predecessors array" = None
                 ) -> BuechiResult:
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
        BF1.__bench_stats__["n_scc"] += 1
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
            BF1.__bench_stats__["n_backedges"] += 1
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
