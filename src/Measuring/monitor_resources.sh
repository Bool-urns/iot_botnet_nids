#!/bin/bash

#turning off hdmi
vcgencmd display_power 0
#starting measurement scripts in background
bash cpu_utilisation.sh & bash mem_utilisation.sh &
echo "utilisation tests running"
sleep 3s
#start nids system to test
python nids.py
#stopping measurements scripts
pkill -f cpu_utilisation.sh
pkill -f mem_utilisation.sh
echo "utilisation tests  ended"
#turning hdmi back on
vcgencmd display_power 1
