#!/bin/bash

N=${1:-10}
S=${2:-1}

echo "Creating example upto $N with step $S"

for i in $(seq 1 ${S} ${N})
do
  ./satellite_work.sh $i > ./satellite_work_$i.tchk
  ./satellite_work_base.sh $i > ./satellite_work_base_$i.tchk
  ./satellite_work_dis.sh $i > ./satellite_work_dis_$i.tchk
done
