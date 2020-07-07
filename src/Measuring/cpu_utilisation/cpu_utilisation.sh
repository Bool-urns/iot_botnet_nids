#!/bin/bash

now=$(date '+%m_%d_%y_%H_%M_%S')

headers="total_cpu, cpu0, cpu1, cpu2, cpu3, time"

#output to csv
echo $headers >> ${now}_cpu_usage.csv
start=$SECONDS
while true; do
  values=$(bash get_cpus.sh | tr "\n" ",") #0.05
  timer=$(( SECONDS - start ))
  echo $values $timer  >> ${now}_cpu_usage.csv
  sleep 0.5
done
