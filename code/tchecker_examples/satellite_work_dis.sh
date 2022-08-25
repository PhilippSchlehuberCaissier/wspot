#!/bin/bash

NMODULES=${1:-1}
DTDISC=${2:-10}
# How much time and energy it takes to send
WNIGHT=-10
WDAY=40
WENER=-10
WTIME=10

cat << EOF
# This file is a part of the TChecker project.
#
# See files AUTHORS and LICENSE for copyright details.

#clock:size:name
#int:size:min:max:init:name
#process:name
#event:name
#location:process:name{attributes}
#edge:process:source:target:event:{attributes}
#sync:events
#   where
#   attributes is a colon-separated list of key:value
#   events is a colon-separated list of process@event

# event:n0 Dummy
# event:wp starting work is currently possible
# event:tr$i transmission of i-th work module complete
# Added by translation: dT for timepassing
# state n: night
# state d: day
# state p$i: idle state of i-th module
# state s$i: sending state of i-th module
system:satellite_work_${NMODULES}

clock:1:x

event:n0

# Process controlling night and day
process:P
location:P:n{initial::labels:$WNIGHT,Init:invariant:x<=35}
location:P:d{labels:$WDAY:invariant:x>=35&&x<=90}

edge:P:n:d:n0{provided:x>=35&&x<=35}
edge:P:d:n:n0{provided:x>=90&&x<=90:do:x=0}
EOF

for i in $(seq $NMODULES)
do

cat << EOF

# Work module $i
event:tr$i

process:W$i

location:W$i:p$i{initial::labels:0,Init}

EOF

for t0 in $(seq 0 $DTDISC $((90 - $i)))
do

cat << EOF
location:W$i:s${i}x${t0}{labels:$WENER}

edge:W$i:p$i:s${i}x${t0}:n0{provided:x>=$t0&&x<=$t0}
edge:W$i:s${i}x${t0}:p$i:tr$i{provided:x>=$(($t0 + $i))&&x<=$(($t0 + $i))}

EOF
done
done

