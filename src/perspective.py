import cv2

def unwarp(img, src, dst):
    """
    Return perspective matrix according to image, src points and dist points

    :param img image: original image
    :param src array: array of source points
    :param dst array: array of distenation points
    """
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    return warped, M
