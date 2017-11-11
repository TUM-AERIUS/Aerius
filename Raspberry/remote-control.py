import socket
import json
import smbus
from time import sleep
from datetime import datetime

TCP_HOST = '192.168.42.1'
TCP_PORT = 5015
BUFFER_SIZE = 1024
i2c_address = 0x37


def send(bus, out):
    for c in out:
        try:
            bus.write_byte(i2c_address, ord(c))
        except:
            print('Loose Connection!')
            sleep(1)
            send(bus, out)


bus = smbus.SMBus(1)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((TCP_HOST, TCP_PORT))
sock.listen(20)

# Setup Socket
conn, address = sock.accept()
conn.settimeout(1.2)
print("Connection address: ", address)

out = "0,0."
stopped = False

while 1:
    try:
        data = conn.recv(BUFFER_SIZE)
        out = data.decode().split(' ')[1]
        stopped = False
    except socket.timeout:
        if not stopped:
            send(bus, "0,0.")  # Failsafe
        stopped = True
        print('Failsafe activated')

    if stopped:
        time.sleep(3)
        continue

    v = out.split(',')[0]
    s = out.split(',')[1][:-1]

    print(out + " -> " + str(datetime.now().time()))
    send(bus, out)
