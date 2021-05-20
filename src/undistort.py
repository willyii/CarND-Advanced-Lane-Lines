import cv2

def undistort(image, mtx=None, dist = None ):
    return cv2.undistort(image, mtx, dist, None, mtx)


