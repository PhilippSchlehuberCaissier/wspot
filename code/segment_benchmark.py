import spot, buddy, copy, subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import time
import math
import cProfile
import pstats
import io

import WBA_solvers as ws
import WBA_FW as wf
from energy import EnergyFunction
from nested_loops_builder import NestedLoopsBuilder, AltNestedLoopsBuilder


# We will use the nested loops automata with an increasing number k of nested loop.
loops = 11

builder = AltNestedLoopsBuilder(loops)
WUP = builder.wup
EnergyFunction.set_wup(WUP)

# subprocess.run(["python3", "stairs_builder.py", str(loops)])
builder.build()
hoa = spot.automaton("../tests/nested_loops_auto.hoa")
# hoa = spot.automaton("../tests/devil.hoa")
# hoa = spot.automaton("../tests/stairs_auto.hoa")
# hoa = spot.automaton("../tests/many_iterations_co_buechi_flattened.hoa")

ef_class = EnergyFunction


def valid(M, i, j):
    if i == j:
        if M[i][j].is_above_one:
            return True


with cProfile.Profile() as cpr:
    cpr.disable()
    cpr.run('wf.FWhoa(hoa, ef_class, valid)')
    # cpr.run('wf.FWhoa(hoa, ef_class, valid)', 'restats')

    cpr.disable()
    cpr.print_stats()
    cpr.dump_stats('restats')

    stats = pstats.Stats('restats')
    stats.print_callees()
    # stats.dump_stats("restats")

exit()

# ============

M = wf.FWhoa(hoa, EnergyFunction, lambda M, i, j: True)
n = len(M[0])
M_seg = [[len(f.segments) for f in li] for li in M]

print(f"Result matrix has {sum([sum(li) for li in M_seg])} energy segments")

fig, ax = plt.subplots()
im = ax.imshow(M_seg)
ax.set_xticks(range(n), range(n))
plt.xlabel("to")
ax.set_yticks(range(n), range(n))
plt.ylabel("from")

for i in range(n):
    for j in range(n):
        text = ax.text(j, i, M_seg[i][j], ha="center", va="center", color='w')

ax.set_title(f"Number of segments in energy functions in the FW matrix for {loops} loops")
fig.tight_layout()
plt.show()
