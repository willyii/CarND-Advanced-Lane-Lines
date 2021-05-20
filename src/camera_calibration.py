"""
This file define the function used in camera callibration
"""
import numpy as np 
import cv2
import matplotlib.image as mpimg
import glob

def calibrate_camera(dir_path):
    """
    This function use the images in dir_path directy to calibrate the camera.
    Then save the result in calibrate_param file.

    :param dir_path string: matching regular experssion can match images 
    :return mtx and dist
    """
    images = glob.glob(dir_path)

    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    for f in images:
        img = mpimg.imread(f);
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret , corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    img_shape = mpimg.imread(images[0]).shape;
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
            img_shape[1::-1], None, None)

    return mtx, dist

