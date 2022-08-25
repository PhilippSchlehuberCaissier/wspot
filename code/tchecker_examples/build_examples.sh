#!/bin/bash

for i in {1..10}
do
  ./satellite_work.sh $i > ./satellite_work_$i.tchk
  ./satellite_work_base.sh $i > ./satellite_work_base_$i.tchk
  ./satellite_work_dis.sh $i > ./satellite_work_dis_$i.tchk
done