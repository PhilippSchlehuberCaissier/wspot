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
from energy import EnergyFunction


# We will use the nested loops automata with an increasing number k of nested loops.
loops = 20
WUP = 150
# WUP = loops
EnergyFunction.set_wup(WUP)

# subprocess.run(["python3", "nested_loops_builder.py", str(loops)])
# subprocess.run(["python3", "stairs_builder.py", str(loops)])
# hoa = spot.automaton("../tests/nested_loops_auto.hoa")
hoa = spot.automaton("../tests/devil.hoa")
# hoa = spot.automaton("../tests/stairs_auto.hoa")
# hoa = spot.automaton("../tests/many_iterations_co_buechi_flattened.hoa")

ef_class = EnergyFunction

def valid(M, i, j):
    if i == j:
        if M[i][j].is_above_one:
            return True

cProfile.run('wf.FWhoa(hoa, ef_class, valid)')
cProfile.run('wf.FWhoa(hoa, ef_class, valid)', 'restats')
# exit()
M = wf.FWhoa(hoa, EnergyFunction, lambda M, i, j: True)
exit()
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
