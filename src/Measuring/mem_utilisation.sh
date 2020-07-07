#!/bin/bash

now=$(date '+%m_%d_%y_%H_%M_%S')

headers="mem_used%","mem_free%","time"


echo $headers >> ${now}_mem_usage.csv
start=$MILISECONDS
while true; do
  #used %= ((total - available)total) * 100
  mem_use=$(free | awk 'FNR == 2 {print (($2-$7)/$2) *100}')
  
  #free % = (free/total) * 100
  mem_free=$(free | awk 'FNR == 2 {print $4/$2 *100.0}')
  
  #time added for graphing purposes
  timer=$(( SECONDS - start ))
  echo $mem_use,$mem_free,$timer >> ${now}_mem_usage.csv
  sleep 1
done
