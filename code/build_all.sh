#!/bin/bash
set -e

git clone https://github.com/PhilippSchlehuberCaissier/tchecker.git
cd tchecker
git checkout energyBA
mkdir build
cd build
cmake $1 ..
make -j 8
cd ../..

git clone https://gitlab.lrde.epita.fr/spot/spot.git
cd spot
git checkout sven/weighted
autoreconf -vfi
./configure $2
make -j 8
make install -j 8
make mostlyclean

