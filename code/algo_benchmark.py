import spot, buddy, copy, subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import time
import math

import WBA_solvers as ws
from energy import EnergyFunction
# TODO refactor these
from stairs_builder import StairsBuilder
from nested_loops_builder import NestedLoopsBuilder
from devilCircles import DevilBuilder


## Benchmark to assess the efficiency of our algorithms with automata of varying sizes.
# We will use the nested loops automata with an increasing number k of nested loops.
start = 3
max_loops = 11
builder = NestedLoopsBuilder(start)
# builder = DevilBuilder(start)
wup = builder.wup

times_naive = []
times_cycles = []
times_backtrack = []
times_FW = []

solvers = [
    # naive
    ws.NaiveCoBuechi,
    # cycles
    ws.LEGACY_CoBuechiEnergy,
    # backtrack
    ws.CoBuechiEnergy,
    # FW
    ws.CoBuechi_FW_new
]
times = [[0 for _ in range(len(solvers))] for _ in range(start, max_loops + 1)]

for k in range(start, max_loops + 1):
    builder.update(k)
    builder.build()
    wup = builder.wup
    EnergyFunction.set_wup(wup)

    # hoa = spot.automaton("../tests/nested_loops_auto.hoa")
    # hoa = spot.automaton("../tests/stairs_auto.hoa")
    hoa = spot.automaton(builder.output)

    for i in range(len(solvers)):
        solver = solvers[i]
        start_time = time.time()
        solver(hoa, 0, wup, 0)
        times[k-start][i] = time.time() - start_time
        print(f"{solver.__name__} ...done")

for s in times:
    print(s)

xs = range(start, max_loops + 1)

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, len(solvers))))
plt.plot(xs, times)
plt.legend(["naive", "cycle storage", "backtracking", "Floyd-Warshall"], loc="upper left")
plt.title(f"Co-Büchi solving in the {builder.name} automaton")
plt.xlabel("Number of loops")
plt.xticks(xs, xs)
plt.ylabel("Execution time in s")
plt.yscale("log")
plt.show()
