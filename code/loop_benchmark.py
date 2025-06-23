## Benchmark to assess the efficiency of co-Büchi energy solving. Uses Python time.

import spot, buddy, WBA_solvers as ws
import time

files = [
    "../tests/nested_loops.hoa",
    "../tests/large_co_buechi_flattened.hoa"
    ]

for this in files:
    hoa = spot.automaton(this)

    print(f"Solving {this} using the naive algorithm")
    start = time.time()
    for _ in range(5000):
        ws.NaiveCoBuechi(hoa, 0, 10, 0)
    print(f"Time: {time.time() - start}")

    print(f"Solving {this} using the new algorithm")
    start = time.time()
    for _ in range(5000):
        ws.CoBuechiEnergy(hoa, 0, 10, 0)
    print(f"Time: {time.time() - start}")

    print("======================================")
    print(f"Solving {this} using the legacy algorithm")
    start = time.time()
    for _ in range(5000):
        ws.LEGACY_CoBuechiEnergy(hoa, 0, 10, 0)
    print(f"Time: {time.time() - start}")
