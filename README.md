# wspot
Extending spot to weighted and timed automata

The tool relies on two main projects: Spot and Tchecker

# Spot
To represent weighted automata we are currently using a branch of SPOT:
https://gitlab.lrde.epita.fr/spot/spot/-/tree/sven/weighted

# Tchecker
To work with timed automata, we use a fork of the tchecker project

# Usage

## Installation

Two possibilities: Either install locally or use the provided docker image

### Local installation

In order to clone and compile the code, you can simply
cd into the code repository and call build_all.sh

Note that spot and tchecker have several dependencies.
Notably spot needs autotools and tchecker libboost and catch2.

#### Using the local installation
Once everything is compiled, simply do (while still being in the code
directory):

./spot/tests/run jupyter notebook energy_buechi.ipynb

This should launch a notebook containing several test cases
as well as the example presented in the paper.
The benchmarks can be found in energy_buechi_bench.ipynb

### Dockerfile

If you have docker installed, simply use
docker run -it --rm -p 8888:8888 elfuius/wspot:0.2 jupyter notebook --NotebookApp.default_url=/lab/ --ip=0.0.0.0 --port=8888
to download and launch the latest "stable" image.
This will give you a jupyterlab interface to the project.
Note that wspot:0.2 is build using code/Dockerfile.

## Mybinder

The jupyerlab interface is also available under 
https://mybinder.org/v2/gh/PhilippSchlehuberCaissier/wspot/latest

### Files
The two main notebooks are 
energy_buechi.ipynb which showcases the algorithms
and
energy_buechi_bench.ipynb which is used to launch the benchmarks

The actual code can be found in WBA_utils.py
simple_1CTA.py and to_weighted_twa.py provided utility functions
for treating weighted timed automata with one clock.

wspot
  - LICENSE
  - README.md
  - Dockerfile simplified Dockerfile for mybinder
  code Main folder
    - WBA_utils.py Main implementation of modified Bellman-Ford
    - simple_1CTA.py Translation tchecker zone-graph -> corner point abstraction
    - energy_buechi.ipynb Usage examples
    - energy_buechi_bench.py Runs the benchmark
    - Dockerfile "main" Dockerfile used to actually build the image
    - tchecker_examples tchecker input files
    - Rest: Helper and convenience scripts

### Known issues
The compilation of spot might not terminate with gcc-9.4