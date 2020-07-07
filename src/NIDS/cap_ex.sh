#!/bin/bash

time_window=$1

now=$(date '+%m_%d_%y_%H_%M_%S')

directory="time_files/"

file_pcap="$directory${now}.pcap"
file_argus="$directory${now}.argus"
file_csv="$directory${now}.csv"

features="srate drate mean dpkts spkts dbytes sbytes proto flgs sum"

tshark -i wlan0 -a duration:${time_window} -w $file_pcap -F pcap
argus -r $file_pcap -w $file_argus
ra -r $file_argus -s $features -c, > $file_csv

echo $file_csv
