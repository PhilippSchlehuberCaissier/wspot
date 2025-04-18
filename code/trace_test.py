import spot, buddy
import WBA_utils


def test1():
    g = spot.make_twa_graph()

    g.new_states(50) # Much more states to increase the duration of BF

    for (s, w, d) in [(0, -5, 1), (0, -5, 3),
                      (1,-5, 2),
                      (3, -5, 2), (3, 1, 4), (3, 0, 1),
                      (4, 1, 5), (4, 0, 1),
                      (5, 1, 3), (5, 0, 1)]:
        en = g.new_edge(s, d, buddy.bddtrue)
        spot.set_weight(g, en, w)



    wup = 50
    ic = 10

    mBF = WBA_utils.mod_BF_iter(g)

    en, pred = mBF.FindMaxEnergy(0, wup, ic)
    print("Energy\n", en)
    print("predecessors")
    for s, p in enumerate(pred):
        print(s, ": ", p)

    t = WBA_utils.searchTrace(g, pred, 0, 2, ic, wup//2, wup)
    print("trace")
    print(t)


def test2():
    g = spot.make_twa_graph()

    g.new_states(50)
    g.set_buchi()

    for (s, w, acc, d) in [(0, 0, [], 1), (0, 0, [], 5),  # 0
                           (1, 2, [], 2),  # 1
                           (2, -1, [], 3),  # 2
                           (3, 0, [], 4), (3, 0, [], 5),  # 3
                           (4, 0, [], 1),  # 4
                           (5, -1, [0], 0),  # 5 # 6
                           ]:
        en = g.new_edge(s, d, buddy.bddtrue, acc)
        spot.set_weight(g, en, w)

    print(g.to_str("hoa"))

    br = WBA_utils.BuechiEnergy(g, 0, wup=50, c0=0, do_display=1)
    print(br)

    t = WBA_utils.traceExctraction(br, False)
    print(t)

    tproj = WBA_utils.traceExctraction(br, True)
    print(tproj)

def test2bis():
    g = spot.make_twa_graph()

    g.new_states(50)
    g.set_generalized_buchi(2)

    for (s, w, acc, d) in [(0, 0, [], 1), (0, 0, [], 5),  # 0
                           (1, 2, [], 2),  # 1
                           (2, -1, [], 3),  (2, -10, [], 6),  # 2
                           (3, 0, [], 4), (3, 0, [], 5),  # 3
                           (4, 0, [], 1),  # 4
                           (5, -1, [0], 0),  # 5
                           (6, -10, [1], 1),  # 6
                           ]:
        en = g.new_edge(s, d, buddy.bddtrue, acc)
        spot.set_weight(g, en, w)

    print(g.to_str("hoa"))

    br = WBA_utils.BuechiEnergy(g, 0, wup=50, c0=0, do_display=1)
    print(br)

    t = WBA_utils.traceExctraction(br, False)
    print(t)

    tproj = WBA_utils.traceExctraction(br, True)
    print(tproj)


def test3():
    g = spot.make_twa_graph()

    g.new_states(50)
    g.set_buchi()

    for (s, w, acc, d) in [(0, 5, [], 1),  # 0
                           (1, -5, [], 2), (1, -3, [], 6),  # 1
                           (2, 3, [], 3), (2, -1, [], 5),  # 2
                           (3, 0, [], 4), (3, -1, [], 2),  # 3
                           (4, 0, [], 5), (4, -1, [], 3), (4, -5, [], 6),  # 4
                           (5, 0, [], 2), (5, -1, [], 4),  # 5
                           (6, -1, [0], 1),  # 6
                           ]:
        en = g.new_edge(s, d, buddy.bddtrue, acc)
        spot.set_weight(g, en, w)

    print(g.to_str("hoa"))

    br = WBA_utils.BuechiEnergy(g, 0, wup=50, c0=50, do_display=1)
    print(br)

    t = WBA_utils.traceExctraction(br, False)
    print(t)

    tproj = WBA_utils.traceExctraction(br, True)
    print(tproj)


def testXX():
    g = spot.automaton("""HOA: v1
States: 14
Start: 1
AP: 0
acc-name: Buchi
Acceptance: 1 Inf(0)
properties: trans-labels explicit-labels state-acc complete
--BODY--
State: 0 "0:0"
[t] 1 <0>
[t] 5 <0>
State: 1 "1:0"
[t] 2 <2>
State: 2 "2:0"
[t] 3 <-1>
[t] 6 <-10>
State: 3 "3:0"
[t] 4 <0>
[t] 5 <0>
State: 4 "4:0"
[t] 1 <0>
State: 5 "5:0"
[t] 7 <-1>
State: 6 "6:0"
[t] 1 <-10>
State: 7 "0:1"
[t] 8 <0>
[t] 12 <0>
State: 8 "1:1"
[t] 9 <2>
State: 9 "2:1"
[t] 10 <-1>
[t] 13 <-10>
State: 10 "3:1"
[t] 11 <0>
[t] 12 <0>
State: 11 "4:1"
[t] 8 <0>
State: 12 "5:1"
[t] 7 <-1>
State: 13 "6:1" {0}
[t] 1 <-10>
--END--""")
    bf = WBA_utils.mod_BF_iter(g)
    for i, (en, pred) in enumerate(bf.FindMaxEnergyGen(1, 50, 0)):
        print(i, en, pred)




if __name__ == '__main__':
    test1()
    test2()
    testXX()
    test2bis()
    test3()