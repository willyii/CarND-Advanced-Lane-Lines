import cv2

def transform(img, src, dst):
    """
    Return perspective matrix according to image, src points and dist points

    :param img image: original image
    :param src array: array of source points
    :param dst array: array of distenation points
    """
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    return warped, M, Minv
