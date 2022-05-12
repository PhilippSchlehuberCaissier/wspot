# wspot
Extending spot to weighted and timed automata

# Spot
To represent weighted automata we are currently using a branch of SPOT:
https://gitlab.lrde.epita.fr/spot/spot/-/tree/sven/weighted

# Tchecker
To work with timed automata, we use a fork of the tchecker project

# Installation
In order to clone and compile the code, you can simply
cd into the code repository and call build_all.sh

Note that spot and tchecker have several dependencies.
Notably spot needs autotools and tchecker libboost and catch2.

# Usage
Once everything is compiled, simply do (while still being in the code
directory):

./spot/tests/run jupyter notebook energy_buechi.ipynb

This should launch a notebook containing several test cases
as well as the example presented in the paper.