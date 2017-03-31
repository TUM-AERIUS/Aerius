import io
import socket
import struct
from PIL import Image
import picamera
import time

# Connect to StereoCamera.py
cameraSocket = socket.socket()
cameraSocket.connect(('localhost', 8100))

# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)

# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rwb')
try:
    # http://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/
    camera = picamera.PiCamera()
    camera.resolution = (640, 480)
    rawCapture = picamera.PiRGBArray(camera)

    # Start a preview and let the camera warm up for 2 seconds
    camera.start_preview()
    time.sleep(2)

    while True:
        # wait for a request from the StereoCamera controller script
        data = cameraSocket.recv(1)

        # request for the other pi to take a photo and send it over
        connection.write(struct.pack('<L', 1))
        connection.flush()

        stream = io.BytesIO()
        camera.capture(stream, format="png")
        # camera.capture(rawCapture, format="bgr")
        # rawCapture.array

        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]

        if not image_len:
            break

        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))

        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        imageLeft = Image.open(image_stream)
        print('Image is %dx%d' % imageLeft.size)
        imageLeft.verify()
        print('Image left is verified')

        stream.seek(0)
        imageRight = Image.open(stream)
        print('Image is %dx%d' % imageRight.size)
        imageRight.verify()
        print('Image right is verified')

finally:
    camera.close()
    connection.close()
    server_socket.close()
