from camera_calibration import calibrate_camera
from undistort import undistort
from perspective import transform
from color_grad import color_grad
import numpy as np
import os
from line_finding import *
import matplotlib.pyplot as plt

def helper(fit, y):
    return fit[0] * (y**2) + fit[1] * y + fit[2]

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
    undist, M, Minv = transform(undist, src, dst)

    warped = color_grad(undist)
    
    leftx, lefty, rightx, righty = find_lane_pixels(warped)

    left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)

    left_cur, right_cur = get_curve_real(lefty, righty, left_fit, right_fit)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    pts_left = np.array([np.transpose(np.vstack([helper(left_fit, lefty), lefty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([helper(right_fit,
        righty), righty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    return result
