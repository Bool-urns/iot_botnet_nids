#!/usr/bash

#based on: https://askubuntu.com/questions/22021/how-to-log-cpu-load
#and:https://stackoverflow.com/questions/54856201/how-to-store-ps-command-output-in-csv-format 

#Prints the top ten processes in descending order of memory utilisation into a csv file every second

now=$(date '+%m_%d_%y%T')
while true; do
  (ps -e -o %p, -o lstart -o ,%C, -o %mem -o ,%c --sort=%mem | tail) >> ${now}_system_resouces.csv; 
  sleep 1
done
