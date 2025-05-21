## Benchmark to check the memory used by the new (backtrack) co-Büchi algorithm.

import spot, buddy, WBA_utils as wu

this = "../tests/large_co_buechi_flattened.hoa"
hoa = spot.automaton(this)

for _ in range(5000):
    wu.OmegaEnergy(hoa, 0, 10, 0, 0)
