## Benchmark to check the memory used by the new (backtrack) co-Büchi algorithm.

import spot, buddy, WBA_utils as wu, WBA_solvers as ws
import subprocess
from memory_profiler import memory_usage
import numpy as np
import matplotlib.pyplot as plt

from nested_loops_builder import NestedLoopsBuilder


solvers = [ws.NaiveCoBuechi, ws.LEGACY_CoBuechiEnergy, ws.CoBuechiEnergy]#, ws.CoBuechi_FW_new]
memory = [[] for _ in range(len(solvers))]
loops = 92
builder = NestedLoopsBuilder(loops)

def to_benchmark(what):
    builder.build()
    this = "../tests/nested_loops_auto.hoa"
    hoa = spot.automaton(this)

    what(hoa, 0, 10, 0)
    del hoa
    del this
    # ws.NaiveCoBuechi(hoa, 0, 10, 0)
    # ws.LEGACY_CoBuechiEnergy(hoa, 0, 10, 0)
    # ws.CoBuechiEnergy(hoa, 0, 10, 0)
    # ws.CoBuechi_FW_new(hoa, 0, 10, 0)

# if __name__ == '__main__':
#     to_benchmark()


for i in range(len(solvers)):
    solver = solvers[i]
    memory[i] = memory_usage((to_benchmark, (solver,)))

n = max([len(mem) for mem in memory])
for i in range(len(memory)):
    memory[i] = np.pad(memory[i], (0, n-len(memory[i])), 'constant', constant_values=np.nan)

time = np.linspace(0, n * 0.2, n)
plt.plot(time, np.transpose(memory))
plt.legend(["naive", "cycle storage", "backtracking"], loc="lower right")
plt.title("Co-Büchi solving in the nested loops automaton")
plt.xlabel("Execution time in s")
plt.ylabel("Memory usage in MiB")
plt.show()
