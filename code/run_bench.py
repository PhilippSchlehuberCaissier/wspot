# Benchmarks
import os
import subprocess
import argparse

from copy import deepcopy as deepcopy

parser = argparse.ArgumentParser(description='Run satellite plus work modules benchmarks')

parser.add_argument('--Nmax', default=10, help="Maximal number of workmodules", type=int)
parser.add_argument('--Nstep', default=1, help="Interval of instances to check", type=int)
parser.add_argument('--initial', default=350, help="Starting energy", type=int)
parser.add_argument('--wup', default=650, help="Weak upper bound for energy", type=int)
parser.add_argument('--Nrep', default=3, help="How often to repeat each instance", type=int)
parser.add_argument('--file', default="bench.csv", help="Name of the csv output file", type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    Nall = list(range(1, args.Nmax+1, args.Nstep))

    subprocess.call(f"cd tchecker_examples && ./build_examples.sh {args.Nmax+1} {args.Nstep}", shell=True)

    print(f"Benchmarking for # modules:\n{Nall}")
    spec_ltl_base = "GF(dT)"

    print(os.getcwd())
    os.chdir("/media/philipp/cc069263-05da-48d7-bc4c-5fa6918260464/git/wspot/code")

    try:
        os.remove(args.file)
    except FileNotFoundError:
        pass

    for i in Nall:
        tchkFiles = [f"tchecker_examples/satellite_work_base_{i}.tchk", f"tchecker_examples/satellite_work_{i}.tchk"]
        spec_ltl = deepcopy(spec_ltl_base)
        for j in range(1, i+1, 1):
            spec_ltl += f"&&GF(tr{j})"

        for tchkFile in tchkFiles:
            for _ in range(args.Nrep):
                subprocess.call(f"""./spot/tests/run one_exec.py {tchkFile} "{spec_ltl}" {args.file} {args.initial} {args.wup} {i}""", shell=True)
