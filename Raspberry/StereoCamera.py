import numpy as np
import cv2
import socket

# start server for PhotoServer.py
photoSocket = socket.socket()
photoSocket.bind(('localhost', 8100))
photoSocket.listen(0)

photoConnection, address = photoSocket.accept()

numBoards = 30  # how many boards would you like to find
board_w = 7
board_h = 6

board_sz = (7, 6)
board_n = board_w*board_h

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
object_points = []  # 3d point in real world space
imagePoints1 = []  # 2d points in image plane.
imagePoints2 = []  # 2d points in image plane.

corners1 = []
corners2 = []

obj = np.zeros((6*7, 3), np.float32)
obj[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# TODO: change here to use the PhotoServer as a client to get photos
photoConnection.send(1)

vidStreamL = cv2.VideoCapture(0)  # index of your camera
vidStreamR = cv2.VideoCapture(1)  # index of your camera
success = 0
k = 0
found1 = False
found2 = False

while success < numBoards:

    retL, img1 = vidStreamL.read()
    height, width, depth = img1.shape
    retR, img2 = vidStreamR.read()
    #resize(img1, img1, Size(320, 280));
    #resize(img2, img2, Size(320, 280));
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    found1, corners1 = cv2.findChessboardCorners(img1, board_sz)
    found2, corners2 = cv2.findChessboardCorners(img2, board_sz)

    if found1:
        cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1),criteria)
        cv2.drawChessboardCorners(gray1, board_sz, corners1, found1)

    if found2:
        cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(gray2, board_sz, corners2, found2)

    cv2.imshow('image1', gray1)
    cv2.imshow('image2', gray2)

    k = cv2.waitKey(100)
    print(k)

    if k == 27:
        break
    if k == 32 and found1 != 0 and found2 != 0:

        imagePoints1.append(corners1);
        imagePoints2.append(corners2);
        object_points.append(obj);
        print("Corners stored\n")
        success += 1

        if success >= numBoards:
            break

cv2.destroyAllWindows()
print("Starting Calibration\n")
cameraMatrix1 = cv2.cv.CreateMat(3, 3, cv2.CV_64FC1)
cameraMatrix2 = cv2.cv.CreateMat(3, 3, cv2.CV_64FC1)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(object_points, imagePoints1, imagePoints2, (width, height))

print("Done Calibration\n")
print("Starting Rectification\n")
R1 = np.zeros(shape=(3,3))
R2 = np.zeros(shape=(3,3))
P1 = np.zeros(shape=(3,3))
P2 = np.zeros(shape=(3,3))

cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,(width, height), R, T, R1, R2, P1, P2, Q=None, flags=cv2.cv.CV_CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0,0))

print("Done Rectification\n")
print("Applying Undistort\n")

map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (width, height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (width, height), cv2.CV_32FC1)

print("Undistort complete\n")

# while True:
#     retL, img1 = vidStreamL.read()
#     retR, img2 = vidStreamR.read()
#     imgU1 = np.zeros((height, width, 3), np.uint8)
#     imgU1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
#     imgU2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
#     cv2.imshow("imageL", img1)
#     cv2.imshow("imageR", img2)
#     cv2.imshow("image1L", imgU1)
#     cv2.imshow("image2R", imgU2)
#     k = cv2.waitKey(5)
#     if k == 27:
#         break

# http://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#reprojectimageto3d

# TODO: use StereoMatcher::compute to get the disparity matrix

# TODO: reprojectImageTo3D mathod to get the the points in 3d space, needs disparity and other values we got from calibration