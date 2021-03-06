from __future__ import print_function
import math
from time import sleep
from rplidar import RPLidar
import numpy as np
import traceback
import socket
import io
import struct

port = 7553


def create_server():
    server_socket = socket.socket()
    server_socket.bind(('', port))
    server_socket.listen(1)
    return server_socket


def get_client(server):
    print('Waiting for a connection...')
    client_connection = server.accept()[0].makefile('rwb')
    print('Connected')
    return client_connection


def send_data(connection, state):
    f = io.BytesIO()
    np.save(f, state)
    num_bytes = f.tell()
    connection.write(struct.pack('<L', num_bytes))
    connection.flush()
    f.seek(0)
    connection.write(f.read())
    connection.flush()
    print("Sent")


def rplidar_init():
    try:
        lidar = RPLidar('/dev/ttyUSB1')
    except:
        lidar = RPLidar('/dev/ttyUSB0')

    lidar.start_motor()
    lidar.set_pwm()
    info = lidar.get_info()
    print(info)

    health = lidar.get_health()
    print(health)

    return lidar


def pair(point):
    return (int(math.floor(point[1] + 60) % 360 / 3), point[2] * 1.2)  # This is a hack. Fix it in the simulation.


def transform(points):
    points = [pair(p) for p in points if (p[1] < 60 or p[1] > 300)] # This is a hack. Change it in v2.0!!!!
    readings = [600] * 120
    for point in points:
        readings[point[0]] = min(point[1], readings[point[0]])
    # Take minimum of the 3 points.
    readings_min = [600] * 40
    for i in range(40):
        readings_min = min(readings[3*i], readings[3*i+1], readings[3*i+2])
    # print("points:",points, readings)
    return np.array(readings_min)


if __name__ == "__main__":
    server_socket = create_server()
    client_connection = get_client(server_socket)
    lidar = rplidar_init()

    while True:
        try:
            for i, scan in enumerate(lidar.iter_scans()):  # Read the LiDaR point cloud
                # sleep(1)
                nn_state = transform(scan)  # Transform Point Cloud into suitable NP-Array

                send_data(client_connection, nn_state)

        except Exception as ex:
            print(ex)
            traceback.print_exc()
            lidar.stop_motor()
            lidar.disconnect()
            server_socket.close()
            client_connection.close()
