import socket

TCP_IP = '127.0.0.1'
TCP_PORT = 5005
BUFFER_SIZE = 1024

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

while 1:
    conn, address = s.accept()
    print("Connection address: ", address)

    # Get Data
    data = conn.recv(BUFFER_SIZE)
    if data: print("Received data: ", data)

    conn.close()
