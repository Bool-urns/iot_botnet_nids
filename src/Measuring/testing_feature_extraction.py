import subprocess
import datetime
import os

import pandas
from sklearn.preprocessing import LabelEncoder

#mine
#saddr daddr proto flgs dur pkts bytes mean rate sum

#all
#srate drate seq min stddev stime ltime state_number max  dpkts dbytes sbytes spkts 
#features = ["saddr", "daddr", "dur", "pkts", "bytes", "mean", "rate", "sum", "srate", "drate", "seq", "min", "stddev", "stime", "ltime", "max",  "dpkts", "dbytes", "sbytes", "spkts"]

#needs the preprocess_data methood too
#other_features = ["proto", "flgs", "state"]
#for i in features:
#    feat = i
#    print(feat)

def pcap_input_extract(file_pcap, feature):  
  feat = feature
  directory = "feature_test/"
  file_argus = directory +  feat + ".argus"
  file_csv = directory + feat + ".csv"
  subprocess.call(["argus", "-r", file_pcap, "-w", file_argus])
  #cmd = "ra -r " + file_argus + " -s saddr daddr proto flgs dur pkts bytes mean rate sum -c, > " + file_csv
  cmd = "ra -r " + file_argus + " -s " + feat + " -c, > " + file_csv
  os.system(cmd)
  return file_csv


def preprocess_data(new_csv):
  df = pandas.read_csv(new_csv)
  label_encoder = LabelEncoder()
  
  proto_values = df['Proto'].values
  flg_values = df['Flgs'].values
  #state_values = df['State'].values

  proto_encoded = label_encoder.fit_transform(proto_values)
  flg_encoded = label_encoder.fit_transform(flg_values)
  #state_encoded = label_encoder.fit_transform(state_values)

  df['Proto_num'] = proto_encoded
  df['Flg_num'] = flg_encoded
  #df['State_num'] = state_encoded
  df = df.drop(['SrcAddr', 'DstAddr', 'Proto', 'Flgs'], axis=1)
  return df

pcap = "IoT_Dataset_TCP_DoS__00046_20180604134915.pcap"
csv = pcap_input_extract(pcap, "srate")
#df = preprocess_data(csv)
