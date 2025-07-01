import spot, buddy, copy, subprocess
import matplotlib.pyplot as plt
import os
import time
import math

import WBA_solvers as ws


## Benchmark to assess the efficiency of our algorithms with automata of varying sizes.
# We will use the nested loops automata with an increasing number k of nested loops.
max_loops = 11

times_naive = []
times_cycles = []
times_backtrack = []
times_FW = []

solvers = [
    ws.NaiveCoBuechi,
    ws.LEGACY_CoBuechiEnergy,
    ws.CoBuechiEnergy,
    ws.CoBuechi_FW_new
]
times = [[0 for _ in range(len(solvers))] for _ in range(2, max_loops + 1)]

for k in range(2, max_loops + 1):
    # Create the nested loops automaton
    subprocess.run(["python3", "nested_loops_builder.py", str(k)])
    hoa = spot.automaton("../tests/nested_loops_auto.hoa")

    for i in range(len(solvers)):
        solver = solvers[i]
        start = time.time()
        solver(hoa, 0, 10, 0)
        times[k-2][i] = time.time() - start

for s in times:
    print(s)

plt.plot(range(2, max_loops + 1), times)
plt.legend(["naive", "cycle storage", "backtracking", "Floyd-Warshall"], loc="upper left")
plt.title("Co-Büchi solving in the nested loops automaton")
plt.xlabel("Number of loops")
plt.ylabel("Execution time in s")
plt.yscale("log")
plt.show()
