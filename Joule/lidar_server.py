from __future__ import print_function
import math
from time import sleep
from rplidar import RPLidar
import numpy as np
import traceback
import socket
from io import StringIO
import struct

port = 7555


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
    f = StringIO()
    np.savez_compressed(f, frame=state)
    num_bytes = f.tell()
    connection.write(struct.pack('<L', num_bytes))
    f.seek(0)
    connection.write(f.read())


def rplidar_init():
    try:
        lidar = RPLidar('/dev/ttyUSB1')
    except:
        lidar = RPLidar('/dev/ttyUSB0')

    lidar.start_motor()
    info = lidar.get_info()
    print(info)

    health = lidar.get_health()
    print(health)

    return lidar


def pair(point):
    return int(math.floor(point[1] + 20) % 360), point[2] / 100


def transform(points):
    points = [pair(p) for p in points if p[1] < 20 or p[1] > 340]
    readings = [50] * 40
    for point in points:
        readings[point[0]] = min(point[1], readings[point[0]])

    # print("points:",points, readings)
    return np.array(readings)


if __name__ == "__main__":
    server_socket = create_server()
    client_connection = get_client(server_socket)
    while True:
        lidar = rplidar_init()

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
