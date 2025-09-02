## Benchmark to measure the number of energy segments for each energy function in the Floyd-Warshall matrix.

import spot, buddy, copy, subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import time
import math
import cProfile

import WBA_solvers as ws
import WBA_FW as wf
from energy import EnergyFunction
from nested_loops_builder import NestedLoopsBuilder


# We will use the nested loops automata with an increasing number k of nested loops.
loops = int(sys.argv[1])
assert loops > 1
WUP = 150

# subprocess.run(["python3", "nested_loops_builder.py", str(loops)])
builder = NestedLoopsBuilder(loops)
builder.build()
hoa = spot.automaton(builder.output)
# hoa = spot.automaton("../tests/devil.hoa")
# hoa = spot.automaton("../tests/many_iterations_co_buechi_flattened.hoa")

EnergyFunction.set_wup(builder.wup)
ef_class = EnergyFunction


def diag_is_above_one(M, i, j):
    return i == j and M[i][j].is_above_one


M = wf.FWhoa(hoa, ef_class, diag_is_above_one)
n = len(M[0])
M_seg = [[len(f.segments) for f in li] for li in M]

print(f"Result matrix has {sum([sum(li) for li in M_seg])} energy segments")
exit()

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
# plt.show()
