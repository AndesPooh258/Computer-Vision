import numpy as np
import cv2

def line_det(im):
    # convert the image to gray and process with Gaussian blur for robust detection
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_blurred = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # detect edges with Canny operator
    im_canny = cv2.Canny(im_blurred, 100, 200)

    # detect lines with Hough transform
    lines = cv2.HoughLines(im_canny, 1, np.pi / 180, 150)

    # transform results to pixel coordinates and draw the lines
    if lines is not None:
        for line in lines:
            r, theta = line[0]
            pt1 = [int(r * np.cos(theta) - 1000 * np.sin(theta)), int(r * np.sin(theta) + 1000 * np.cos(theta))]
            pt2 = [int(r * np.cos(theta) + 1000 * np.sin(theta)), int(r * np.sin(theta) - 1000 * np.cos(theta))]
            cv2.line(im, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
    
    # return detected result image
    return im

if __name__ == '__main__':
    im = cv2.imread('./misc/road.jpeg')
    
    result = line_det(im)
    cv2.imwrite('./results/line_det.png', result)