import spot, buddy, copy, subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import time
import math
import cProfile

import WBA_solvers as ws
import WBA_FW as wf
from energy import EnergyFunctionWup


# We will use the nested loops automata with an increasing number k of nested loops.
loops = 10

subprocess.run(["python3", "nested_loops_builder.py", str(loops)])
hoa = spot.automaton("../tests/nested_loops_auto.hoa")
# hoa = spot.automaton("../tests/many_iterations_co_buechi_flattened.hoa")

ef_class = EnergyFunctionWup(100)
cProfile.run('wf.FWhoa(hoa, ef_class, lambda M, i, j: True)')
cProfile.run('wf.FWhoa(hoa, ef_class, lambda M, i, j: True)', 'restats')
M = [] #wf.FWhoa(hoa, EnergyFunctionWup(100), lambda M, i, j: True)
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
