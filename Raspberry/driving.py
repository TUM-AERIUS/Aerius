"""
Once a model is learned, use this to drive it.
"""

import environment
import numpy as np
from nn import neural_net
import tensorflow as tf
import socket

TCP_IP = '127.0.0.1'
TCP_PORT = 5005
BUFFER_SIZE = 1024

NUM_SENSORS = 49

# Goal: LL -> LR -> D -> RL -> RR
L = 0
R = 1
D = 2


def drive(session, state, prediction):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(20)

    vel = 0
    deg = 0

    while True:
        # Get Data from App
        conn, address = s.accept()
        print("Connection address: ", address)

        data = conn.recv(BUFFER_SIZE)
        if data: print("Received data: ", data)
        #Todo: Parse velocity and steering from data
        vel_new =
        deg_new =

        conn.close()


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
                action = R if sin_angle < 0 else L
            else: action = D

        # Update values
        vel, deg = vel_new, deg_new

        # Take action.
        worldstate = actuate(vel, deg)

        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)

def actuate(velocity, steering):
    setMotor(velocity)
    setServo(steering)


if __name__ == "__main__":
    session, update, prediction, state, input, labels = neural_net(NUM_SENSORS, [180, 164])

    drive(session, state, prediction)
