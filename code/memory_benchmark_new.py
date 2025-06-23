## Benchmark to check the memory used by the new (backtrack) co-Büchi algorithm.

import spot, buddy, WBA_utils as wu, WBA_solvers as ws

this = "../tests/nested_loops.hoa"
hoa = spot.automaton(this)

for _ in range(100):
    ws.NaiveCoBuechi(hoa, 0, 10, 0)
