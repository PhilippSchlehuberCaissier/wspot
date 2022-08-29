import spot
import WBA_utils
import to_weighted_twa
import time
import sys
import os

from copy import deepcopy as deepcopy


def n_states(zgFile:str):
    with open(zgFile, "r") as f:
        c = 0
        f.readline()
        while("intval" in f.readline()):
            c += 1
    return c

if __name__ == "__main__":
    assert(len(sys.argv) == 4)

    tchkFile = sys.argv[1]
    spec_ltl = sys.argv[2]
    benchfile = sys.argv[3]

    WBA_utils.reset_stats()

    t = time.time()
    wba = to_weighted_twa.translate(tchkFile)
    tchk2cpatime = time.time() - t
    zgStates = n_states(tchkFile + ".zg")

    t = time.time()
    spec = spot.translate(spec_ltl, dict=wba.get_dict())
    spectranstime = time.time() - t

    t = time.time()
    tot_wba = spot.product(wba, spec)
    prodtime = time.time() - t
    prodstates = tot_wba.num_states()

    # A energy wup of 650 ensures that the problem is feasible
    # till i > 20
    t = time.time()
    feas = WBA_utils.BuechiEnergy(tot_wba, 0, 650, 350, 0)
    solvetime = time.time() - t

    det_bench = deepcopy(WBA_utils.get_stats())

    if not os.path.isfile(benchfile):
        with open(benchfile, "w") as f:
            f.write("file,spec,tchk2cpatime,zgStates,cpaStates,spectranstime,prodtime,prodstates,feas,solvetime,n_backedges,n_bf_iter,n_scc,n_pump_loop\n")
    with open(benchfile, "a") as f:
        f.write(f'{tchkFile},{spec_ltl},{tchk2cpatime},{zgStates},{wba.num_states()},{spectranstime},{prodtime},{prodstates},{1 * feas},{solvetime},{det_bench["n_backedges"]},{det_bench["n_bf_iter"]},{det_bench["n_scc"]},{det_bench["n_pump_loop"]}\n')
    print("Done")
