import subprocess
import pandas
from sklearn.preprocessing import LabelEncoder

import time

#remove time in seconds
def remove_files(remove_time):
  rm_time = str(remove_time)
  subprocess.call(["bash", "remove_files.sh", rm_time])
  #add print of what files were deleted

#time window in seconds
def capture_and_extract(time_window):
  t_win = str(time_window)
  proc = subprocess.Popen(["bash", "cap_ex.sh", t_win], stdout=subprocess.PIPE)
  csv = proc.stdout.read()
  return csv


def main():
  csv = capture_and_extract(30)
  remove_files(10)
  print(csv)

main()
