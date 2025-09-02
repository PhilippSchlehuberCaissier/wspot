## Utility to build nested loop automata with an arbitrary amount of loops.
# A nested loop automaton with k loops is an automaton that has k-1 negative loops of strictly increasing length and one final positive loop of length greater than the largest negative loop.
# These kind of automata are useful for benchmarking our algos since they can be as large as we want

# wup doesn't matter for nesteds
# we can stress test our algos by using a large wup
NESTED_LOOPS_WUP = 1000


# Quick and dirty Fibonacci
def fib(n):
    dic = {0: 1, 1: 1}
    for i in range(2, n+1):
        dic[i] = dic[i-1] + dic[i-2]
    return dic[n]


# TODO factorize this (superclass HOABuilder)
class NestedLoopsBuilder:
    def __init__(self, k):
        assert k > 1
        self.k = k
        self.wup = NESTED_LOOPS_WUP
        self.name = "nested loops"
        self.output = "../tests/nested_loops_auto.hoa"

    def update(self, k):
        self.__init__(k)

    def build(self):

        print(f"Building nested loop automaton with {self.k} loops at {self.output}")
        f = open(self.output, 'w')

        # Usual HOA headers
        print("HOA: v1", file=f)
        print(f"States: {2 + sum(range(self.k))}", file=f)
        print("Start: 0", file=f)
        print("AP: 1 \"a\"", file=f)
        print("acc-name: co-Buchi 1", file=f)
        print("Acceptance: 1 Fin(0)", file=f)
        print("properties: trans-labels explicit-labels trans-acc weak", file=f)
        print("--BODY--", file=f)

        # Initial state is always this
        print("State: 0\n[t] 0 <-5> {0}\n[t] 1 <10>", file=f)

        # "Hub" state
        print("State: 1\n[t] 1 <-1>", file=f)
        for i in range(2, self.k+1):
            print(f"[t] {2 + sum(range(i-1))} <0>", file=f)

        next_state = 2

        # Build every loop
        # Loop lp has lp new states
        for lp in range(self.k):
            for st in range(lp):
                print(f"State: {next_state}", file=f)
                if st == lp - 1:
                    print("[t] 1 <1>", file=f) if lp == self.k-1 else print("[t] 1 <-1>", file=f)
                else:
                    print(f"[t] {next_state+1} <0>", file=f)

                next_state += 1

        print("--END--", file=f)
