#!/bin/sh

set -e  # Abort on any error
set -x  # Show each instruction as it is run

NB_USER=jovyan
HOME=/home/${NB_USER}

#VERSION=`dpkg-query -W -f='${Version}' spot`
#DATE=`date -I`
#sed -i "s/DATE/$DATE/g;s/VERSION/$VERSION/g" \
#    ${HOME}/.jupyter/custom/custom.js
chown ${NB_USER}:${NB_USER} ${HOME}/.jupyter

## LTL2BA
#V=1.3
#wget https://www.lrde.epita.fr/dload/spot/deps/ltl2ba-$V.tar.gz
#tar xvf ltl2ba-$V.tar.gz
#cd ltl2ba-$V
#make -j4
#mv ltl2ba /usr/local/bin/
#mkdir -p /usr/local/share/doc/ltl2ba
#cp LICENSE README /usr/local/share/doc/ltl2ba
#cd ..
#rm -rf ltl2ba-$V ltl2ba-$V.tar.gz
#
## LTL3BA
#V=1.1.3
#wget https://www.lrde.epita.fr/dload/spot/deps/ltl3ba-$V.tar.gz
#tar xvf ltl3ba-$V.tar.gz
#cd ltl3ba-$V
#make -j4
#mv ltl3ba /usr/local/bin/
#mkdir -p /usr/local/share/doc/ltl3ba
#cp LICENSE README /usr/local/share/doc/ltl3ba
#cd ..
#rm -rf ltl3ba-$V ltl3ba-$V.tar.gz
#
## LTL3DRA
#V=0.3.0
#wget https://github.com/xblahoud/ltl3dra/archive/v$V.tar.gz -O ltl3dra-$V.tar.gz
#tar xvf ltl3dra-$V.tar.gz
#cd ltl3dra-$V
#make -j4
#mv ltl3dra /usr/local/bin/
#mkdir -p /usr/local/share/doc/ltl3dra
#cp LICENSE README /usr/local/share/doc/ltl3dra
#cd ..
#rm -rf ltl3dra-$V ltl3dra-$V.tar.gz
#
## LTL3TELA
#V=2.1.1
#cat >ltl3tela.patch <<\EOF
#diff --git a/Makefile b/Makefile
#index 044f7b4..65d62c4 100644
#--- a/Makefile
#+++ b/Makefile
#@@ -21 +21 @@
#-	g++ -std=c++14 -o ltl3tela $(FILES) -lspot -lbddx
#+	g++ -std=c++17 -O2 -o ltl3tela $(FILES) -lspot -lbddx
#diff --git a/automaton.cpp b/automaton.cpp
#index 1669882..87a73e9 100644
#--- a/automaton.cpp
#+++ b/automaton.cpp
#@@ -111 +111 @@ template<typename T> void Automaton<T>::add_edge(unsigned from, bdd label, std::
#-			if (spot_id_to_slaa_set == nullptr) {
#+			if constexpr (! std::is_same<T, unsigned>::value) {
#@@ -153 +153 @@ template<typename T> void Automaton<T>::add_edge(unsigned from, bdd label, std::
#-			if (spot_id_to_slaa_set == nullptr) {
#+			if constexpr (! std::is_same<T, unsigned>::value) {
#EOF
#wget https://github.com/jurajmajor/ltl3tela/archive/v$V.tar.gz
#tar xvf v$V.tar.gz
#cd ltl3tela-$V
#patch -p1 < ../ltl3tela.patch
#make -j4
#mv ltl3tela /usr/local/bin || mv ltl3hoa /usr/local/bin
#mkdir -p /usr/local/share/doc/ltl3tela
#cp LICENSE README.md /usr/local/share/doc/ltl3tela
#cd ..
#rm -rf ltl3tela-$V v$V.tar.gz
#
## Seminator
#V=2.0
#wget https://github.com/mklokocka/seminator/releases/download/v$V/seminator-$V.tar.gz
#tar xvf seminator-$V.tar.gz
#cd seminator-$V
#sed -i 's/c[+][+]14/c++17/g' configure
#./configure
#make -j4
#make install
#ldconfig
#mkdir -p /usr/local/share/doc/seminator
#cp COPYING README.md /usr/local/share/doc/seminator
#mv notebooks /usr/local/share/doc/seminator
#cd ..
#rm -rf seminator-$V seminator-$V.tar.gz
#
## ltl2dstar
#V=0.5.4
#wget https://www.lrde.epita.fr/dload/spot/deps/ltl2dstar-$V.tar.gz
#tar xvf ltl2dstar-$V.tar.gz
#cd ltl2dstar-$V/src
#make -j4
#mv ltl2dstar /usr/local/bin/
#cd ..
#mkdir -p /usr/local/share/doc/ltl2dstar
#cp NEWS LICENSE README /usr/local/share/doc/ltl2dstar
#cd ..
#rm -rf ltl2dstar-$V ltl2dstar-$V.tar.gz
#
## jhoafparser 1.1.1
#V=1.1.1
#mkdir -p /usr/local/share/jhoaf
#wget http://automata.tools/hoa/jhoafparser/down/jhoafparser-$V.jar -O /usr/local/share/jhoaf/jhoafparser-$V.jar
#wget http://automata.tools/hoa/jhoafparser/down/jhoafparser-$V.zip -O /tmp/jhoafparser-$V.zip
#unzip /tmp/jhoafparser-$V.zip jhoafparser-$V/LICENSE jhoafparser-$V/README -d /usr/local/share/doc/
#mv /usr/local/share/doc/jhoafparser-$V /usr/local/share/doc/jhoafparser
#rm -f /tmp/jhoafparser-$V.zip
#cat >/usr/local/bin/jhoaf <<EOF
##!/bin/sh
#exec java -jar /usr/local/share/jhoaf/jhoafparser-$V.jar "\$@"
#EOF
#chmod +x /usr/local/bin/jhoaf
#
#
## Spin already installed as Debian package
#ln -s /usr/share/doc/spin /usr/local/share/doc
## Divine already installed as Debian package
#ln -s /usr/share/doc/divine-ltsmin  /usr/local/share/doc/
## Spins already installed as Debian package
#ln -s /usr/share/doc/spins  /usr/local/share/doc/
#
##SPINVERSION=`spin -V | sed -n 's/.* \([0-9.]\+\) --.*/\1/p'`
##sed -i "s/@SPINVERSION@/$SPINVERSION/" ${HOME}/README
#
## Owl
#V=20.06.00
#wget http://www.lrde.epita.fr/dload/spot/deps/owl-$V.zip
#mkdir -p /usr/local/share/
#unzip owl-$V.zip -d /usr/local/share/
#rm -f owl-$V.zip
#
#for i in delag dra2dpa ltl2da ltl2dgra ltl2dpa ltl2dra ltl2ldba \
#         ltl2ldgba ltl2na ltl2nba ltl2ngba nba2dpa nba2ldba nbadet \
#         nbasim owl owl-native owl-server
#do
#    ln -s /usr/local/share/owl-$V/bin/$i /usr/local/bin/
#done
#