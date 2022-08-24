#!/bin/bash

NMODULES=${1:-1}

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
event:wp

# Process controlling night and day
process:P
location:P:n{initial::labels:$WNIGHT,Init:invariant:x<=35}
location:P:d{labels:$WDAY:invariant:x<=55}

edge:P:n:n:wp{}
edge:P:n:d:wp{provided:x>=35&&x<=35:do:x=0}
edge:P:n:d:n0{provided:x>=35&&x<=35:do:x=0}
edge:P:d:n:wp{provided:x>=55&&x<=55:do:x=0}
edge:P:d:n:n0{provided:x>=55&&x<=55:do:x=0}
EOF

for i in $(seq $NMODULES)
do

cat << EOF

# Work module $i
event:tr$i

process:W$i

location:W$i:p$i{initial::labels:0,Init}
location:W$i:s$i{labels:$WENER:invariant:x<=$i}

edge:W$i:p$i:s$i:wp{do:x=0}
edge:W$i:s$i:p$i:tr$i{provided:x>=$i&&x<=$i}

sync:P@wp:W$i@wp
EOF

done

