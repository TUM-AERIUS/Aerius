import cv2
import numpy as np

imagesFolder = "stereo-calibration/calib_imgs/1/"

numBoards = 30  # how many boards would you like to find
board_w = 3
board_h = 3

board_sz = (9, 6)
board_n = board_w*board_h

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
object_points = []  # 3d point in real world space
imagePoints1 = []  # 2d points in image plane.
imagePoints2 = []  # 2d points in image plane.

corners1 = []
corners2 = []

obj = np.zeros((9*6, 3), np.float32)
obj[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

success = 0
k = 0
i = 1
found1 = False
found2 = False

while success < numBoards and i < 30:

    img1 = cv2.imread(imagesFolder + "left" + str(i) + ".jpg", 1)
    img2 = cv2.imread(imagesFolder + "right" + str(i) + ".jpg", 1)

    img1 = cv2.resize(img1, (320, 280))
    img2 = cv2.resize(img1, (320, 280))

    height, width, depth = img1.shape

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    found1, corners1 = cv2.findChessboardCorners(img1, board_sz)
    found2, corners2 = cv2.findChessboardCorners(img2, board_sz)

    if found1:
        cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1),criteria)
        cv2.drawChessboardCorners(gray1, board_sz, corners1, found1)

    if found2:
        cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(gray2, board_sz, corners2, found2)

    # cv2.imshow('image1', img1)
    # cv2.imshow('image2', img2)

    # input("Press Enter...")

    if found1 and found2:
        imagePoints1.append(corners1)
        imagePoints2.append(corners2)
        object_points.append(obj)
        print("Corners stored")
        success += 1

    if success >= numBoards:
        break

    i += 1

print("Starting Calibration\n")

cameraMatrix1 = np.array((3, 4, cv2.CV_64FC1), np.uint8)
cameraMatrix2 = np.array((3, 4, cv2.CV_64FC1), np.uint8)

distCoeffs1 = np.array((8, 1), np.uint8)
distCoeffs2 = np.array((8, 1), np.uint8)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(object_points, imagePoints1, imagePoints2, \
                                                                                                 cameraMatrix1, distCoeffs1, cameraMatrix2, \
                                                                                                 distCoeffs2, (width, height))
print("Done Calibration\n")
print("Starting Rectification\n")
R1 = np.zeros(shape=(3, 3))
R2 = np.zeros(shape=(3, 3))
P1 = np.zeros(shape=(3, 3))
P2 = np.zeros(shape=(3, 3))
Q = np.zeros(shape=(4, 4))

cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,(width, height), R, T, R1, R2, P1, P2, Q, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0,0))

print(Q)

print("Done Rectification\n")
print("Applying Undistort\n")

map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (width, height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (width, height), cv2.CV_32FC1)

print("Undistort complete\n")

cv2.imshow("image1", img1)