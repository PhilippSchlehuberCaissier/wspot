## Utility to build a stair automata with an arbitrary amount of loops.
# A stair automaton with k loops is an automaton that has k negative loops composed of four transitions: -i, +i, +k, -k-1+i. This type of automaton results in a stair-shaped energy function on the diagonal
# Each loop adds 3 new states to the automaton.
# These kind of automata are useful for benchmarking our algos since they can be as large as we want

import sys


# Quick and dirty Fibonacci
def fib(n):
    dic = { 0: 1, 1: 1 }
    for i in range(2, n+1):
        dic[i] = dic[i-1] + dic[i-2]
    return dic[n]


k = int(sys.argv[1])
assert k > 1
output = "../tests/stairs_auto.hoa"

print(f"Building stair automaton with wup of {k} at {output}")
f = open(output, 'w')

# Usual HOA headers
print("HOA: v1", file=f)
print(f"States: {2 + 3 * k}", file=f)
print("Start: 0", file=f)
print("AP: 1 \"a\"", file=f)
print("acc-name: co-Buchi 1", file=f)
print("Acceptance: 1 Fin(0)", file=f)
print("properties: trans-labels explicit-labels trans-acc weak", file=f)
print("--BODY--", file=f)

# Initial state is always this
print(f"State: 0\n[t] 0 <-5> {{0}}\n[t] 1 <{k}>", file=f)

# "Hub" state
print("State: 1", file=f)
for i in range(k):
    print(f"[t] {2 + 3*i} <{-i}>", file=f)

# Build every loop
for lp in range(k):
    print(f"State: {2 + 3*lp}\n[t] {3 + 3*lp} <{lp}>", file=f)
    print(f"State: {3 + 3*lp}\n[t] {4 + 3*lp} <{k}>", file=f)
    print(f"State: {4 + 3*lp}\n[t] 1 <{-k+1+lp}>", file=f)

print("--END--", file=f)
