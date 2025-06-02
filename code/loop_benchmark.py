## Benchmark to assess the efficiency of co-Büchi energy solving.

import spot, buddy, WBA_utils as wu
import time

files = [
    "../tests/nested_loops.hoa",
    "../tests/large_co_buechi_flattened.hoa"
    ]

for this in files:
    hoa = spot.automaton(this)

    print(f"Solving {this} using the new algorithm")
    start = time.time()
    for _ in range(1000):
        wu.OmegaEnergy(hoa, 0, 10, 0, 0)
    print(f"Time: {time.time() - start}")

    print("======================================")
    print(f"Solving {this} using the legacy algorithm")
    start = time.time()
    for _ in range(1000):
        wu.LEGACY_CoBuechiEnergy(hoa, 0, 10, 0)
    print(f"Time: {time.time() - start}")
