from camera_calibration import calibrate_camera
from undistort import undistort
from perspective import unwarp 
from color_grad import color_grad
import numpy as np
import os
from line_finding import fit_polynomial 
import matplotlib.pyplot as plt


def image_pipeline(image):

    if(os.path.exists("./param/calibration_param.npz")):
        data = np.load("./param/calibration_param.npz")
        mxt = data["mxt"]
        dist = data["dist"]
    else:
        # Camera calibration
        mxt, dist = calibrate_camera("../camera_cal/calibration*.jpg") 

    # undistortion
    undist = undistort(image, mxt, dist)
    h,w = undist.shape[:2]

    # perspective
    src = np.float32([(575,464),(707,464), (258,682),(1049,682)])
    dst = np.float32([(450,0), (w-450,0), (450,h), (w-450,h)])
    warped, M = unwarp(undist, src, dst)

    binary = color_grad(warped)
    
    return fit_polynomial(binary)
