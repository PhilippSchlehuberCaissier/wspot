import argparse
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

parser = argparse.ArgumentParser(description="Run one instance given as tcheckerfile and ltl spec")

parser.add_argument("tchkFile", help="Name of the tchecker file", type=str)
parser.add_argument("spec", help="Specification given in ltl", type=str)
parser.add_argument("file", help="CSV output file", type=str)
parser.add_argument('initial', help="Starting energy", type=int)
parser.add_argument('wup', help="Weak upper bound for energy", type=int)
parser.add_argument('nMod', help="Number of work modules", type=int)



if __name__ == "__main__":
    args = parser.parse_args()

    tchkFile = args.tchkFile
    spec_ltl = args.spec
    benchfile = args.file

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

    # for the base model we need at least
    # 350 + 10*#N modules wup
    t = time.time()
    feas = WBA_utils.BuechiEnergy(tot_wba,
                                  tot_wba.get_init_state_number(),
                                  args.wup,
                                  args.initial, 0)
    solvetime = time.time() - t

    det_bench = deepcopy(WBA_utils.get_stats())

    if not os.path.isfile(benchfile):
        with open(benchfile, "w") as f:
            f.write("file,spec,numModules,tchk2cpatime,zgStates,cpaStates,spectranstime,prodtime,prodstates,feas,solvetime,n_backedges,n_bf_iter,n_scc,n_pump_loop,n_propagate\n")
    with open(benchfile, "a") as f:
        f.write(f'{tchkFile},{spec_ltl},{args.nMod:d},{tchk2cpatime},{zgStates},{wba.num_states()},{spectranstime},{prodtime},{prodstates},{1 * feas},{solvetime},{det_bench["n_backedges"]},{det_bench["n_bf_iter"]},{det_bench["n_scc"]},{det_bench["n_pump_loop"]},{det_bench["n_propagate"]}\n')
    print("Done")
