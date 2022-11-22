#!/bin/bash
set -e

CMAKEOPT=${1:""}
CONFOPT=${2:""}
DOINSTALL=${3:"No"}

git clone https://github.com/PhilippSchlehuberCaissier/tchecker.git
cd tchecker
git checkout energyBA
mkdir build
cd build
cmake $CMAKEOPT ..
make -j 8
cd ../..

git clone https://gitlab.lrde.epita.fr/spot/spot.git
cd spot
git checkout sven/weighted
autoreconf -vfi
./configure $CONFOPT
make -j 8
if [[ $DOINSTALL == "Yes" ]]
then
  make install -j 8
  make mostlyclean
fi

