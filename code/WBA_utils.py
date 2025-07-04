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
from array import array

import WBA_solvers as solv
from buechi import BuechiResult
import ipython_utils as ipy
ipy_utils = ipy.IPythonUtils()


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
                s0: int,
                wup: int,
                c0: int,
                do_display: int = 0
                ) -> BuechiResult:
    """Searches for energy feasible lasso in the given automaton from the initial state
    with a weak upper bound of \a wup and an initial credit of \a c0

    Returns a BuechiResult allowing to extract the trace.
    """
    if isinstance(aut, str):
        hoa = spot.automaton(aut)
    else:
        hoa = aut

    wup_t = wup
    ipy_utils.set_display_mode(do_display)
    acc_cond = hoa.acc()

    # Empty automaton or f
    if hoa.num_states() == 0 or acc_cond.is_f():
        ipy_utils.print_c("This automaton is empty or its condition is False!")
        return BuechiResult()

    # Condition is t
    if acc_cond.is_t():
        ipy_utils.print_c("True condition detected.")
        return solv.TrueEnergy(hoa, s0, wup_t, c0)

    # Büchi (can be generalized)
    if acc_cond.is_generalized_buchi():
        ipy_utils.print_c("(Generalized) Büchi condition detected.")
        return solv.BuechiEnergy(hoa, s0, wup_t, c0, None, None)

    # Co-Büchi
    if acc_cond.is_co_buchi():
        ipy_utils.print_c("(Generalized) co-Büchi condition detected.")
        return solv.CoBuechi_FW_new(hoa, s0, wup_t, c0)

    # Parity
    # Implements the algorithm presented in Section 7
    if acc_cond.is_parity()[0]:
        ipy_utils.print_c("Parity condition detected.")
        return solv.ParityEnergy(hoa, s0, wup_t, c0)

    # Rabin
    p = acc_cond.is_rabin()
    if p != 1:
        ipy_utils.print_c("Rabin condition detected.")
        return solv.RabinEnergy(hoa, p, s0, wup_t, c0)

    # TODO other automata types
    ipy_utils.print_c("Unknown automaton type. Assuming acceptance condition is t.")
    ipy_utils.print_c("This will lead to errors in trace extraction if the acceptance condition is not t.")
    return solv.TrueEnergy(hoa, s0, wup_t, c0)
