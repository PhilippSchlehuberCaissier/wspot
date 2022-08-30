#!/bin/bash

NMODULES=${1:-1}

# How much time and energy it takes to send
WNIGHT=-10
WDAY=40
WENER=-20

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

# event:n0 Dummy event
# event:sr sunrise
# event:sd sundown
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
location:P:d{labels:$WDAY:invariant:x<=55}

edge:P:n:d:n0{provided:x>=35&&x<=35:do:x=0}
edge:P:d:n:n0{provided:x>=55&&x<=55:do:x=0}
EOF

for i in $(seq $NMODULES)
do

cat << EOF

# Work module $i
event:str$i
event:tr$i

location:P:w$i{labels:$WENER,Init:invariant:x<=$i}

edge:P:n:w$i:str$i{do:x=0}
edge:P:w$i:n:tr$i{provided:x>=$i&&x<=$i}

EOF

done

