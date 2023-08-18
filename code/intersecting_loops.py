import spot, buddy 

"""
Bench1 recursively creates cycles with -wup weights 
backedges have a weight of 0 
cycles with 2 states have 1 weights
the buchi condition is on the backedges
"""

def bench1_(n, aut, s0, wup):
    if n < 3:
        s = aut.new_state()
        idx = aut.new_edge(s0, s, buddy.bddtrue)
        spot.set_weight(aut, idx, 1)
        idx = aut.new_edge(s, s0, buddy.bddtrue)
        spot.set_weight(aut, idx, 1)
        return 
    else:
        tmp = s0

        for _ in range(1, n):
            s = aut.new_state()
            idx = aut.new_edge(tmp, s, buddy.bddtrue)
            spot.set_weight(aut, idx, -wup)
            tmp = s
            bench1_(n - 1, aut, s, wup)
        
        idx = aut.new_edge(s, s0, buddy.bddtrue, [0])
        spot.set_weight(aut, idx, 0)


def bench1(n, wup):
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

    for i in range(1, n):
        idx = aut.new_edge(i - 1, i, buddy.bddtrue)
        spot.set_weight(aut, idx, -wup)
        bench1_(n - 1, aut, i, wup)
    
    aut.new_edge(n - 1, 0, buddy.bddtrue, [0])
    spot.set_weight(aut, idx, 0)
    bench1_(n - 1, aut, 0, wup)

    return aut


"""
Bench2 recursively creates cycles with 0 weights 
on cycles with 3 states, the weights are respectively 0, -wup, -wup
cycles with 2 states have 1 weights  
the buchi condition is on the cycles with 3 states
"""

def bench2_(n, aut, s0, wup):
    if n < 3:
        s = aut.new_state()
        idx = aut.new_edge(s0, s, buddy.bddtrue)
        spot.set_weight(aut, idx, 1)
        idx = aut.new_edge(s, s0, buddy.bddtrue)
        spot.set_weight(aut, idx, 1)
        return 
    
    if n == 3:
        s = aut.new_state()
        idx = aut.new_edge(s0, s, buddy.bddtrue)
        spot.set_weight(aut, idx, 0)
        bench2_(2, aut, s, wup)

        tmp = s
        for _ in range(2,n):
            s = aut.new_state()
            idx = aut.new_edge(tmp, s, buddy.bddtrue)
            spot.set_weight(aut, idx, -wup)
            tmp = s 
            bench2_(2, aut, s, wup)

        idx = aut.new_edge(s, s0, buddy.bddtrue, [0])
        spot.set_weight(aut, idx, -wup)

    else:
        tmp = s0
        for _ in range(1, n):
            s = aut.new_state()
            idx = aut.new_edge(tmp, s, buddy.bddtrue)
            spot.set_weight(aut, idx, 0)
            tmp = s
            bench2_(n - 1, aut, s, wup)

        idx = aut.new_edge(s, s0, buddy.bddtrue)
        spot.set_weight(aut, idx, 0)


def bench2(n, wup):
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

    w1 = 0    
    if n == 3:
        w1 = -3

    for i in range(1, n):
        idx = aut.new_edge(i - 1, i, buddy.bddtrue)
        spot.set_weight(aut, idx, w1)
        bench2_(n - 1, aut, i, wup)
    
    if n == 3:
        aut.new_edge(n - 1, 0, buddy.bddtrue, [0])
    else:
        aut.new_edge(n - 1, 0, buddy.bddtrue)

    spot.set_weight(aut, idx, 0)
    bench2_(n - 1, aut, 0, wup)

    return aut