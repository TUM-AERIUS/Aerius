import cv2
import numpy as np

imagesFolder = "stereo-calibration/calib_imgs/1/"

numBoards = 5  # how many boards would you like to find
board_w = 7
board_h = 6

board_sz = (7, 6)
board_n = board_w*board_h

cv2.waitKey(100)

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


success = 1
k = 0
found1 = False
found2 = False

while success < numBoards:

    img1 = cv2.imread(imagesFolder + "left" + str(success) + ".jpg", 0)
    img2 = cv2.imread(imagesFolder + "right" + str(success) + ".jpg", 0)

    # print(imagesFolder + "left" + str(success) + ".jpg")

    print(img1.shape)

    # cv2.imshow('image1', img1)
    # cv2.imshow('image2', img2)

    print("Corners stored\n")
    success += 1

    if success >= numBoards:
        break
