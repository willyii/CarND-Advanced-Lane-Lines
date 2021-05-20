import cv2
import numpy as np 

s_lower = 100
s_upper = 255
sx_lower = 80
sx_upper = 100
lower_yellow = np.array([15,0,0])
upper_yellow = np.array([40,255,255])
lower_white = np.array([0,0,180])
upper_white = np.array([255,255,255])

def color_grad(img):
    """
    Convert image to grayscale and use sobel to filter the gradient. Then
    convert image to hls and use s channel to filter image. Last, mask yellow
    and white in the image.
    """

    # Sobel x
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_lower) & (scaled_sobel <= sx_upper)] = 1
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_lower) & (s_channel <= s_upper)] = 1

    # Yellow mask and White mask
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    # Combine the masks and mask the original image
    combined_mask = cv2.bitwise_or(yellow_mask,white_mask)

    # Apply masks
    color_binary = np.zeros_like(s_channel)
    color_binary[(s_binary == 1)| (sxbinary == 1)] = 1
    color_binary = cv2.bitwise_and(color_binary, color_binary, mask =
            combined_mask)
    return color_binary
