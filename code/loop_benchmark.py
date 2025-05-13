## Benchmark to assess the efficiency of co-Büchi energy solving.

import spot, buddy, WBA_utils as wu
import time

hoa = spot.automaton("../tests/large_co_buechi_flattened.hoa")

print("Solving using the new algorithm")
start = time.time()
wu.OmegaEnergy(hoa, 0, 10, 0, 0)
print(f"Time: {time.time() - start}")

print("======================================")
print("Solving using the legacy algorithm")
start = time.time()
wu.LEGACY_CoBuechiEnergy(hoa, 0, 10, 0)
print(f"Time: {time.time() - start}")
