import cv2
import numpy as np 

s_lower = 100
s_upper = 255
sx_lower = 70
sx_upper = 100

def color_grad(img):
    """
    Use yellow and white mask to select interested color. Then use s_channel to
    mask the image. Finnally, gradient threshold will be used

    """

    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Sobel x
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_lower) & (scaled_sobel <= sx_upper)] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_lower) & (s_channel <= s_upper)] = 1


    # Stack each channel
    color_binary = np.zeros_like(s_channel)
    color_binary[(s_binary == 1)| (sxbinary == 1)] = 1
    # color_binary[(s_binary == 1)] = 1
    return color_binary
