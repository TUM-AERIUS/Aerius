import socket
import json
import serial
import time
from datetime import datetime


TCP_HOST = '192.168.42.1'
TCP_PORT = 5015
BUFFER_SIZE = 1024

try:
    ser = serial.Serial('/dev/ttyACM0', 9600)
except:
    ser = serial.Serial('/dev/ttyACM1', 9600)

time.sleep(1)
ser.setDTR(level=0)
time.sleep(1)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_HOST, TCP_PORT))
s.listen(20)

while 1:
    # Setup Socket
    conn, address = s.accept()
    conn.settimeout(1)
    print("Connection address: ", address)

    # Get Data
    try:
        data = conn.recv(BUFFER_SIZE)
        out = data.decode().split(' ')[1]

        v = out.split(',')[0]
        s = out.split(',')[1][:-1]

        print(out    + str(datetime.now().time()))
        ser.write(bytes(out))
        print('Sent' + str(datetime.now().time()))

    except socket.timeout: # Handle Timeout
        ser.write(bytes("0,0."))
        print("timeout -> 0,0.")

    conn.close()
