import numpy as np
import cv2

def image_t(im, scale=1.0, rot=45, trans=(50,-50)):
    # transform the rotation angle to radian
    rot_rad = np.pi * rot / 180

    # sample three points and compute their point coordinates in the target image
    m = np.float32([[scale * np.cos(rot_rad), scale * np.sin(rot_rad), trans[0]],
                    [-scale * np.sin(rot_rad), scale * np.cos(rot_rad), trans[1]],
                    [0, 0, 1]])
    src_pt1 = np.float32([0, 0, 1])
    src_pt2 = np.float32([0, 1, 1])
    src_pt3 = np.float32([1, 0, 1])
    dst_pt1 = np.dot(m, src_pt1)
    dst_pt2 = np.dot(m, src_pt2)
    dst_pt3 = np.dot(m, src_pt3)
    h, w, _ = im.shape
    center = np.float32([h/2, w/2])
    src_tri = np.float32([center + src_pt1[0:2], center + src_pt2[0:2], center + src_pt3[0:2]])
    dst_tri = np.float32([center + dst_pt1[0:2], center + dst_pt2[0:2], center + dst_pt3[0:2]])

    # calculate transformation matrix
    trans_mat = cv2.getAffineTransform(src_tri, dst_tri)

    # obtain the target image by using transform matrix
    result = cv2.warpAffine(im, trans_mat, (h, w))

    # return transformed result image
    return result

if __name__ == '__main__':
    im = cv2.imread('./misc/pearl.jpeg')
    
    scale  = 0.5
    rot    = 45
    trans  = (50, -50)
    result = image_t(im, scale, rot, trans)
    cv2.imwrite('./results/affine_result.png', result)