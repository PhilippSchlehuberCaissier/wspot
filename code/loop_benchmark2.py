## Benchmark to assess the efficiency of co-Büchi energy solving. Use with GNU time.

import spot, buddy, WBA_solvers as ws

# files = [
#     "../tests/nested_loops.hoa",
#     "../tests/large_co_buechi_flattened.hoa"
#     ]

#this = "../tests/nested_loops.hoa"
this = "../tests/nested_loops_auto.hoa"

hoa = spot.automaton(this)
for _ in range(1):
    ws.CoBuechi_FW(hoa, 0, 10, 0)

# print(f"Solving {this} using the naive algorithm")
# start = time.time()
# for _ in range(5000):
#     ws.NaiveCoBuechi(hoa, 0, 10, 0)
# print(f"Time: {time.time() - start}")

# print("======================================")
# print(f"Solving {this} using the new algorithm")
# start = time.time()
# for _ in range(5000):
#     ws.CoBuechiEnergy(hoa, 0, 10, 0)
# print(f"Time: {time.time() - start}")

# print("======================================")
# print(f"Solving {this} using the legacy algorithm")
# start = time.time()
# for _ in range(5000):
#     ws.LEGACY_CoBuechiEnergy(hoa, 0, 10, 0)
# print(f"Time: {time.time() - start}")
