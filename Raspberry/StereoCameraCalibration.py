import cv2
import numpy as np
import json
import os
from matplotlib import pyplot as plt
import codecs

imagesFolder = "stereo-photos/"
cameraData = "cameraData.json"

numBoards = 10  # how many boards would you like to find
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

cameraDataFolder = "cameraData.json"
path_to_script = os.path.realpath(__file__)
parent_directory = os.path.dirname(path_to_script)
loaded = False

if os.path.isfile(os.path.join(parent_directory, cameraDataFolder)):
    obj_text = codecs.open(cameraDataFolder, 'r', encoding='utf-8').read()
    cameraData = json.loads(obj_text)
    Q = np.array(cameraData["Q"])
    P1 = np.array(cameraData["P1"])
    P2 = np.array(cameraData["P2"])
    loaded = True

while success < numBoards and i <= 10 and loaded == False:

    if i == 2 or i == 5 or i == 6 or i == 8 or i == 9:
        i += 1
        continue

    print(i)

    img1 = cv2.imread(imagesFolder + "left" + str(i) + ".jpg", 1)
    img2 = cv2.imread(imagesFolder + "right" + str(i) + ".jpg", 1)

    height, width, depth = img1.shape

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    found1, corners1 = cv2.findChessboardCorners(img1, board_sz)
    found2, corners2 = cv2.findChessboardCorners(img2, board_sz)

    print(found1)
    print(found2)

    if found1:
        cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1),criteria)
        cv2.drawChessboardCorners(gray1, board_sz, corners1, found1)

    if found2:
        cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(gray2, board_sz, corners2, found2)

    if found1 and found2:
        imagePoints1.append(corners1)
        imagePoints2.append(corners2)
        # Not sure where obj comes from..
        object_points.append(obj)
        print("Corners stored")
        success += 1

    if success >= numBoards:
        break

    i += 1

if loaded == False:
    print("Starting Calibration\n")

    cameraMatrix1 = np.array((3, 4, cv2.CV_64FC1), np.uint8)
    cameraMatrix2 = np.array((3, 4, cv2.CV_64FC1), np.uint8)

    distCoeffs1 = np.asarray([( -7.42497976e-03,   3.74099082e+00,  -1.81154814e-03,  3.65969196e-04)])
    distCoeffs2 = np.asarray([(  4.05620881e-03,   3.27334706e+00,  -5.00835868e-04,  1.56068477e-03)])

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F \
        = cv2.stereoCalibrate(object_points, imagePoints1, imagePoints2,
                              cameraMatrix1, distCoeffs1, cameraMatrix2,
                              distCoeffs2, (width, height))
    print("Done Calibration\n")
    print("Starting Rectification\n")
    R1 = np.zeros(shape=(3, 3))
    R2 = np.zeros(shape=(3, 3))
    P1 = np.zeros(shape=(3, 3))
    P2 = np.zeros(shape=(3, 3))
    Q = np.zeros(shape=(4, 4))

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1,
                                                      cameraMatrix2, distCoeffs2,
                                                      (width, height), R, T, alpha=0)

    print("Done Rectification\n")
    print("Applying Undistort\n")

    map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1,
                                               R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2,
                                               R2, P2, (width, height), cv2.CV_32FC1)

    print("Undistort complete\n")

    data = {"cameraMatrixLeft"  : cameraMatrix1.tolist(),
            "cameraMatrixRight" : cameraMatrix2.tolist(),
            "distCoeffsLeft"    : distCoeffs1.tolist(),
            "distCoeffsRight"   : distCoeffs2.tolist(),
            "Q"                 : Q.tolist(),
            "P1"                : P1.tolist(),
            "P2"                : P2.tolist()}

    with open(cameraData, "w") as f:
        json.dump(data, f)


min_disp = 0
num_disp = 144
window_size = 3
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 1,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
)
kernel = np.ones((12,12),np.uint8)

print("Done")
img1 = cv2.imread(imagesFolder + "left" + str(7) + ".jpg", 0)
img2 = cv2.imread(imagesFolder + "right" + str(7) + ".jpg", 0)
disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0
threshold = cv2.threshold(disparity, 0.6, 1.0, cv2.THRESH_BINARY)[1]
morphology = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
# Disparity picture for testing
cv2.imshow('threshold', disparity)
cv2.waitKey(0)

