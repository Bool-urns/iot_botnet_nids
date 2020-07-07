import subprocess
import datetime
import os

import pandas

window_size = str(30) #adjust to change time window size
time_win = "duration:"+window_size

def remove_files(num_mins):
    
  now = getDate()
  date_time = now.split('_')
  current_min = int(date_time[4])
  current_hour = int(date_time[3])*60
  current_time = current_hour + current_min
  cut_off = current_time - num_mins

  p = subprocess.Popen(["ls", "time_files"], stdout=subprocess.PIPE)
  output = p.stdout.read()
  files = str(output)
  files = output.split("\n")
  files = files[:-1]
  for file in files:
    file_fields = file.split('_')
    
    mins = int(file_fields[4])
    hours = int(file_fields[3])*60
    c_time = mins + hours
    
    if(c_time < cut_off):
      old_file = "time_files/" + file
      subprocess.call(["rm", old_file])
      print("file: " + old_file + " was deleted") 

def getDate():
  now = str(datetime.datetime.now())
  dt = now.replace(":", "_")
  dt = dt.replace(".", "_")
  dt = dt.replace(" ", "_")
  dt = dt.replace("-", "_")
  return dt
  
def capture_and_extract():
  now = getDate()
  directory = "time_files/"
  file_pcap = directory + now + ".pcap"
  file_argus = directory +  now + ".argus"
  file_csv = directory + now + ".csv"
  subprocess.call(["tshark", "-i", "wlan0", "-a", time_win, "-w", file_pcap, "-F", "pcap"])
  subprocess.call(["argus", "-r", file_pcap, "-w", file_argus])
  cmd = "ra -r " + file_argus + " -s saddr daddr proto flgs dur pkts bytes mean rate sum -c, > " + file_csv
  os.system(cmd)
  return file_csv

def preprocess_data(new_csv):
  df = pandas.read_csv(new_csv)
  
  return df
def main(): 
  csv = capture_and_extract()
  #remove_files(10)
  full_df = preprocess_data(csv) 

main()
