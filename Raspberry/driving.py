"""
Once a model is learned, use this to drive it.
"""

import numpy as np
from nn import neural_net
import tensorflow as tf
import socket
import smbus
from time import sleep
import json

TCP_HOST = '192.168.42.1'
CAM_HOST = '169.254.251.205'

TCP_PORT = 5005
CAM_PORT = 8000

BUFFER_SIZE = 1024
NUM_SENSORS = 42

# Goal LL -> L -> D -> R -> RR
LL = -20
L  = -10
D  = 0
R  = +10
RR = +20


def drive(session, state, prediction):
    connect_i2c() # Connect to Arduino

    bus = smbus.SMBus(1) # Connect to I2C Bus

    remote_conn = remote_connect() # TCP Socket for Remote Control Communication
    camera_sock = camera_socket() # Camera Socket for Cross-Pi Communication

    setup_cam(camera_sock) # Setup Cam

    vel, deg = 0, 0

    # Main Loop
    while True:
        vel_new, deg_new = get_data(remote_conn) # Get Data from App

        world_state = state() # Read Sensors -> Construct internal state

        obstacle = obstacles(readings) # Check for obstacle

        if obstacle: # Imminent Collision -> Invoke NeuralNet
            feed_dict = {state: world_state}
            action = np.argmax(session.run([prediction], feed_dict=feed_dict))
        else:
            # Car velocity vector
            dx = vel * Math.cos(deg)
            dy = vel * Math.sin(deg)

            # Given velocity vector
            fx = vel_new * Math.cos(deg_new)
            fy = vel_new * Math.sin(deg_new)

            cos_angle = Math.cos(deg - deg_new)
            sin_angle = Math.sin(deg - deg_new)

            if cos_angle < 0.998:
                action = RR if sin_angle < 0 else LL
            else: action = deg_new

        vel, deg = vel_new, action # Update values

        send(bus, vel, deg) # Take action.

        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)

def connect_serial():
    try:    ser = serial.Serial('/dev/ttyACM0', 9600)
    except: ser = serial.Serial('/dev/ttyACM1', 9600)

    time.sleep(1)
    ser.setDTR(level=0)
    time.sleep(1)

def remote_connect():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((TCP_IP, TCP_PORT))
    sock.listen(20)

    conn, address = sock.accept()
    conn.settimeout(1.4)
    print("Connection address: ", address)

    return conn

def camera_socket():
    sock = socket.socket()
    sock.connect((CAM_HOST, CAM_PORT))

def setup_cam(sock):
    pi_conn = sock.makefile('rwb')     # Make file-like object out of connection
    # Todo: Setup Camera

def get_data(conn):
    while 1:
        try:
            data = conn.recv(BUFFER_SIZE)
            print('Received data: ', data)
            stopped = False
            break;
        except socket.timeout:
            if !stopped: send(bus, '0,0.')
            stopped = true
            print('Failsafe activated')
            sleep(3)

    values = data.decode().split(' ')[1]
    vel_new = int(values.split(',')[0])
    deg_new = int(values.split(',')[1][:-1])

    return vel_new, deg_new

def state():
    readings = []
    # Todo: Add StereoCam values
    # Todo: Add Ultrasound values
    # Todo: Add Driving Values
    return np.array(readings)

def obstacle(state):
    if min(state[0][40:42]) < MIN_SONIC: # Check Ultrasonic Sensors
        return True
    for i in range(40):
        state[0][i]

def send(bus, vel, deg):
    out = deg + "," + vel + "."
    for c in out:
        try: bus.write_byte(i2c_address, ord(c))
        except:
            print('Loose Connection!')
            sleep(1)
            send(bus, out)


if __name__ == "__main__":
    session, update, prediction, state, input, labels = neural_net(NUM_SENSORS, [180, 164])

    drive(session, state, prediction)
