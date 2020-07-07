import time
start = time.time() #testing execution time

import subprocess
import datetime
import os
import sys

import pandas
from sklearn.preprocessing import LabelEncoder

#from xgb.xgboost_classify import test_xg
from knn import test_knn_classifier

from Super_Fast_Vector_Classifier.sfsvc_preprocess import test_preprocess
from Super_Fast_Vector_Classifier.sfsvc import test_sfsvc_classifier

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

#used when a pcap file is given in the command line arguement instead of capturing the network traffic
def pcap_input_extract(file_pcap):
  now = getDate()
  directory = "time_files/"
  file_argus = directory +  now + ".argus"
  file_csv = directory + now + ".csv"
  subprocess.call(["argus", "-r", file_pcap, "-w", file_argus])
  cmd = "ra -r " + file_argus + " -s saddr daddr proto flgs dur pkts bytes mean rate sum -c, > " + file_csv
  os.system(cmd)
  return file_csv


def preprocess_data(new_csv):
  df = pandas.read_csv(new_csv)
  label_encoder = LabelEncoder()
  
  proto_values = df['Proto'].values
  flg_values = df['Flgs'].values

  proto_encoded = label_encoder.fit_transform(proto_values)
  flg_encoded = label_encoder.fit_transform(flg_values)

  df['Proto_num'] = proto_encoded
  df['Flg_num'] = flg_encoded

  df = df.drop(['SrcAddr', 'DstAddr', 'Proto', 'Flgs'], axis=1)

  return df

def knn_classify(df):
  pred = test_knn_classifier(df)
  return pred

def xg_classify(df):
  pred = test_xg(df)
  return pred

def sfsvc_classify(df):
  test_samples = test_preprocess(df)
  pred = test_sfsvc_classifier(test_samples)
  return pred

def run_once():
  csv = capture_and_extract()
  remove_files(10)
  full_df = preprocess_data(csv)
  results = knn_classify(full_df)
  print(results)
  print("execution time: %s seconds" % (time.time() - start - int(window_size))) #window size taken away to account for predefined time capturing packets

def run_loop():
  while(True):
    csv = capture_and_extract()
    remove_files(10)
    full_df = preprocess_data(csv)
    results = knn_classify(full_df)
    print(results)


def main():
  #run_once()
  #run_loop()
  #csv = "botiot_12c_8f.csv"
  csv = "test.csv"
  full_df = preprocess_data(csv)
  #pred = sfsvc_classify(full_df)
  pred = knn_classify(full_df)
  print(pred)

main()
