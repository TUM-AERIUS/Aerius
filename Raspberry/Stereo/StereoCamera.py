from matplotlib import pyplot as plt
import numpy as np
import cv2
import socket
import struct
import io
import json
import os
import sys

# Useful references:
# http://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#reprojectimageto3d
# use StereoMatcher::compute to get the disparity matrix
# get the the points in 3d space, needs disparity and other values we got from calibration

cameraDataFolder = "cameraData.json"
path_to_script = os.path.realpath(__file__)
parent_directory = os.path.dirname(path_to_script)

def getStereoImages(connection):
    # Requesta new photo from the photo server
    connection.write(struct.pack('<L', 1))
    connection.flush()

    # Get lengths
    imageLeftLength = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
    imageRightLength = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]

    # Get images
    # https://picamera.readthedocs.io/en/release-1.10/recipes1.html
    # streams are in jpeg format
    # we have to reformat for openCV to be able to read them
    imageLeftStream = io.BytesIO()
    imageRightStream = io.BytesIO()

    imageLeftStream.write(connection.read(imageLeftLength))
    imageRightStream.write(connection.read(imageRightLength))

    imageLeftStream.seek(0)
    imageRightStream.seek(0)
    # construct a numpy array from the stream
    data = np.fromstring(imageLeftStream.getvalue(), dtype=np.uint8)
    # "Decode" the image from the array, grayscale image
    # bgr order
    imageLeft = cv2.imdecode(data, 0)

    data = np.fromstring(imageRightStream.getvalue(), dtype=np.uint8)
    imageRight = cv2.imdecode(data, 0)

    return imageLeft, imageRight

# start server for PhotoServer.py
photoSocket = socket.socket()
photoSocket.bind(('localhost', 8100))
photoSocket.listen(0)

photoConnection = photoSocket.accept()[0].makefile("rwb")

# Get calibration data from camerData.json
if os.path.isfile(os.path.join(parent_directory, cameraDataFolder)):
    cameraData = json.loads(cameraDataFolder)
else:
    print("No calibration data found")
    sys.exit(0)

Q = np.array(cameraData["Q"])

window_size = 3
min_disp = 16
num_disp = 112-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 16,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32
)

while True:
    img1, img2 = getStereoImages(photoConnection)
    disparity = stereo.compute(img1, img2)
    image3D = cv2.reprojectImageTo3D(disparity, Q)
    plt.imshow(disparity, 'gray')
    plt.show()
