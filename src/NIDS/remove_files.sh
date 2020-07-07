#!/bin/bash

remove_time=$1

str="-$1 seconds"

find time_files/ -type f -not -newermt "$str" -delete

