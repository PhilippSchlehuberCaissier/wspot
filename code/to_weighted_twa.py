# Helper function to perform the following three steps in a row
# 1) Read a tchecker input file
# 2) Compute the zone graph
# 3) Compute the corner point abstraction
# 4) Save the result as a hoa file

import subprocess
import argparse

import simple_1CTA

import os, sys

GLOBOPTS = {"tchecker":"./tchecker/build/src/tck-reach"}

def set_tchecker_path(path:str) -> None:
    """
    Update the path for tchecker. No check-ups are performed
    Args:
        path: New path

    Returns: None

    """
    GLOBOPTS["tchecker"] = path

def translate(tcheckerFile:"tchecker Input file",
              bddDict:"Dict for AP" = None):
    """_summary_
        Create the corner point astraction of a given (1-clock) timed system
    Args:
        tcheckerFile (tchecker Input file): Tchecker input file with one clock called "x"
        bddDict ("Dict for AP"): spot.bdd_dict or None
    """


    zgFile = f"{tcheckerFile}.zg"
    hoaFile = f"{tcheckerFile}.hoa"
    namedHoaFile = f"{tcheckerFile}_named.hoa"

    # Call tchecker
    print(" ".join([GLOBOPTS["tchecker"], "-a", "reach",  "-C",  zgFile, "-l",  "Init", tcheckerFile]))
    subprocess.check_call([GLOBOPTS["tchecker"], "-a", "reach",  "-C",  zgFile, "-l",  "Init", tcheckerFile])

    sCTA = simple_1CTA.simple_1CTA(tcheckerFile)

    cpaAbstr = simple_1CTA.ZG2HOABuilder(sCTA, zgFile, bddDict)

    names = []
    for locE, scpas in cpaAbstr.locE2scpa.items():
        for az, scpa in zip(cpaAbstr.sCTA.all_zones, scpas):
            if scpa != -1:
                names.append(f"{locE[1]}, {cpaAbstr.locE2info[locE]['rate']}, {az}".replace(" ", ""))

    with open(hoaFile, "w") as f:
        f.write(cpaAbstr.wTWA.to_str("hoa"))

    cpaAbstr.wTWA.set_state_names(names)

    with open(namedHoaFile, "w") as f:
        f.write(cpaAbstr.wTWA.to_str("hoa"))

    return cpaAbstr.wTWA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translates a "weighted" one clock TA to HOA')
    parser.add_argument('tfile', type=str,
                        help='Tchecker file')

    args = parser.parse_args()

    tcheckerFile = args.tfile

    translate(tcheckerFile)

    sys.exit(0)



