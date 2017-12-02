import socket
import numpy as np
import io
import struct


server_address = "192.168.42.1"
port = 7555


def create_connection():
    client_socket = socket.socket()

    try:
        client_socket.connect((server_address, port))
        connection = client_socket.makefile('rwb')
        print('Connected to %s on port %s' % (server_address, port))
    except socket.error as e:
        print('Connection to %s on port %s failed: %s' % (server_address, port, e))

    return connection

def receive_data(connection):
    size = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
    stream = io.BytesIO()
    stream.write(connection.read(size))
    # Transform stream to numpy array
    data = np.load(stream)['frame']
    return data


if __name__ == "__main__":
    connection = create_connection()

    while True:
        data = receive_data(connection)
        print(data)