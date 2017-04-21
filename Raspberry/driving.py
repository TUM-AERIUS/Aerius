"""
Once a model is learned, use this to drive it.
"""

import numpy as np
from nn import neural_net
import tensorflow as tf
import socket
import serial
import time
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

# Connect to Arduino
try:
    ser = serial.Serial('/dev/ttyACM0', 9600)
except:
    ser = serial.Serial('/dev/ttyACM1', 9600)


time.sleep(1)
ser.setDTR(level=0)
time.sleep(1)


def drive(session, state, prediction):
    # TCP Socket for Remote Control Communication
    data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    data_sock.bind((TCP_IP, TCP_PORT))
    data_sock.listen(20)

    # Camera Socket for Cross-Pi Communication
    cam_socket = socket.socket()
    cam_socket.connect((CAM_HOST, CAM_PORT))

    # Make file-like object out of connection
    pi_conn = client_socket.makefile('rwb')

    # Setup Camera
    try:
        with picamera.PiCamera() as camera:
            camera.resolution = (640, 480)
            # Start a preview and let the camera warm up for 2 seconds
            camera.start_preview()
            time.sleep(2)


    vel = 0
    deg = 0

    while True:
        # Get Data from App
        rc_conn, rc_address = s.accept()
        print("Connection address: ", address)

        data = rc_conn.recv(BUFFER_SIZE)
        if data:
            print("Received data: ", data)
            values = data.decode().split(' ')[1]

            vel_new = int(values.split(',')[0])
            deg_new = int(values.split('.')[1])
        rc_conn.close()

        # Todo: Get Sensor values



        # obstacle = obst_cams() or obst_son(LEFT) or obst_son(RIGHT)

        obstacle = False # For now so we're not using the Network

        if obstacle:
            feed_dict = {state: worldstate}
            action = np.argmax(session.run([prediction], feed_dict=feed_dict)) # Choose action.

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

        # Update values
        vel, deg = vel_new, action

        # Take action.
        actuate(vel, deg)

        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)

def actuate(v, s):
    ser.write(bytes(str(v) + ',' + str(s) + '.'))

def



if __name__ == "__main__":
    session, update, prediction, state, input, labels = neural_net(NUM_SENSORS, [180, 164])

    drive(session, state, prediction)
