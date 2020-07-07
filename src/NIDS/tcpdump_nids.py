import time
start = time.time() #testing execution time
import resource

import subprocess
import datetime
import os

import pandas
from sklearn.preprocessing import LabelEncoder

window_size = str(30) #adjust to change time window size

def getDate():
  now = str(datetime.datetime.now())
  dt = now.replace(":", "_")
  dt = dt.replace(".", "_")
  dt = dt.replace(" ", "_")
  dt = dt.replace("-", "_")
  return dt

def capture_and_extract():
  print("entered capture/extract at: %s seconds" % (time.time() - start))
  now = getDate()
  buff_size = str(4096) #size of tcpdump input buffer - default is 2MB
  directory = "time_files/"
  file_pcap = directory + now + ".pcap"
  file_argus = directory +  now + ".argus"
  file_csv = directory + now + ".csv"
  
  #subprocess.call(["tshark", "-i", "wlan0", "-a", time_win, "-w", file_pcap, "-F", "pcap"])
  subprocess.call(["sudo", "timeout", window_size, "tcpdump", "-i", "wlan0", "-B", buff_size, "-w", file_pcap])
  subprocess.call(["argus", "-r", file_pcap, "-w", file_argus])
  cmd = "ra -r " + file_argus + " -s saddr daddr proto flgs dur pkts bytes mean rate sum -c, > " + file_csv
  os.system(cmd)
  return file_csv

def preprocess_data(new_csv):
  print("entered preprocess at: %s seconds" % (time.time() - start))
  df = pandas.read_csv(new_csv)
  label_encoder = LabelEncoder()
  
  proto_values = df['Proto'].values
  flg_values = df['Flgs'].values

  proto_encoded = label_encoder.fit_transform(proto_values)
  flg_encoded = label_encoder.fit_transform(flg_values)

  df['Proto_num'] = proto_encoded
  df['Flg_num'] = flg_encoded

  df = df.drop(['SrcAddr', 'DstAddr', 'Proto', 'Flgs'], axis=1)
  print("left preprocess at: %s seconds" % (time.time() - start))
  return df

def main():
  csv = capture_and_extract()
  df = preprocess_data(csv)
  print(df.head())

main()
