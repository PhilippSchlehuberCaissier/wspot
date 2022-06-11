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

### Dockerfile

If you have docker installed, simply use
docker run -it --rm -p 8888:8888 elfuius/wspot:latest jupyter notebook --NotebookApp.default_url=/lab/ --ip=0.0.0.0 --port=8888
to download and launch the latest "stable" image.
This will give you a jupyterlab interface to the project

## Mybinder

The jupyerlab interface is also available under 
https://mybinder.org/v2/gh/PhilippSchlehuberCaissier/wspot/latest
