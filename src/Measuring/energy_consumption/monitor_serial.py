import serial
import datetime

#function to output data and time in file-friendly format
def getDate():
    now = str(datetime.datetime.now())
    dt = now.replace(":", "_")
    dt = dt.replace(".", "_")
    dt = dt.replace(" ", "_")
    dt = dt.replace("-", "_")
    return dt

#the serial port the arduino is using
serial_port = '/dev/cu.usbmodem14412'
#the same baud rate used in the arduino script
baud_rate = 115200
#the csv file to output
path = "energy_consumption_%s.csv" % (getDate())
ser = serial.Serial(serial_port, baud_rate)
with open(path, 'w+') as f:
    while True:
        line = ser.readline()
        #removing some chararters caught during over the serial output
        the_line = str(line.strip()).replace("'", "")
        the_line = the_line.replace("b", "")
        #writing each to file
        f.write(the_line)
        #ensuring a new line is created
        f.write("\n")
