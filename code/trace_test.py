import spot, buddy
import WBA_utils


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