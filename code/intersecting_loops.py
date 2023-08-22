import spot, buddy 

def bench_(n, aut, s0, wup):
    if n < 3:
        s = aut.new_state()
        idx = aut.new_edge(s0, s, buddy.bddtrue)
        spot.set_weight(aut, idx, 1)
        idx = aut.new_edge(s, s0, buddy.bddtrue)
        spot.set_weight(aut, idx, 1)
        return 
    
    tmp = aut.new_state()
    idx = aut.new_edge(s0, tmp, buddy.bddtrue)
    spot.set_weight(aut, idx, 0)
    bench_(n - 1, aut, tmp, wup)

    for _ in range(2, n):
        s = aut.new_state()
        idx = aut.new_edge(tmp, s, buddy.bddtrue)
        spot.set_weight(aut, idx, -wup)
        tmp = s
        bench_(n - 1, aut, s, wup)

    idx = aut.new_edge(s, s0, buddy.bddtrue)
    spot.set_weight(aut, idx, 0)


def bench(n, wup):
    if n < 2:
        return None
    
    if n == 2:
        aut = spot.make_twa_graph()
        aut.new_states(2)
        aut.set_buchi()
        
        idx = aut.new_edge(0, 1, buddy.bddtrue)
        spot.set_weight(aut, idx, 1)
        
        aut.new_edge(1, 0, buddy.bddtrue, [0])
        spot.set_weight(aut, idx, 1)
    
        return aut
    
    aut = spot.make_twa_graph()
    aut.new_states(n)
    aut.set_buchi()

    idx = aut.new_edge(0, 1, buddy.bddtrue)
    if n == 3:
        spot.set_weight(aut, idx, -wup)
    else:
        spot.set_weight(aut, idx, 0)
    bench_(n - 1, aut, 1, wup)

    for i in range(2, n):
        idx = aut.new_edge(i - 1, i, buddy.bddtrue)
        spot.set_weight(aut, idx, -wup)
        bench_(n - 1, aut, i, wup)

    idx = aut.new_edge(n - 1, 0, buddy.bddtrue, [0])
    if n == 3:
        spot.set_weight(aut, idx, -wup)
    else:
        spot.set_weight(aut, idx, 0)
    bench_(n - 1, aut, 0, wup)

    return aut
