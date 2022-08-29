# Benchmarks
import os
import subprocess

from copy import deepcopy as deepcopy

if __name__ == "__main__":
    import sys
    assert(len(sys.argv) == 3)

    Nmax = int(sys.argv[1])
    Nstep = int(sys.argv[2])
    Nall = list(range(1, Nmax, Nstep))

    subprocess.call(f"cd tchecker_examples && ./build_examples.sh {Nmax} {Nstep}", shell=True)

    print(Nall)
    spec_ltl_base = "GF(dT)"

    file = "bench.csv"

    print(os.getcwd())
    os.chdir("/media/philipp/cc069263-05da-48d7-bc4c-5fa6918260464/git/wspot/code")

    Nrep = 3

    try:
        os.remove(file)
    except FileNotFoundError:
        pass

    for i in Nall:
        tchkFiles = [f"tchecker_examples/satellite_work_base_{i}.tchk", f"tchecker_examples/satellite_work_{i}.tchk"]
        spec_ltl = deepcopy(spec_ltl_base)
        for j in range(1, i+1, 1):
            spec_ltl += f"&&GF(tr{j})"

        for tchkFile in tchkFiles:
            for _ in range(Nrep):
                subprocess.call(f"""./spot/tests/run one_exec.py {tchkFile} "{spec_ltl}" {file}""", shell=True)
