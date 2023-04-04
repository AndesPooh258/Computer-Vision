import numpy as np
import cv2

l1 = lambda i, j : abs(i - j)
l2 = lambda i, j : np.sqrt(i ** 2 + j ** 2)
gaussian = lambda x, sigma : 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * (sigma ** 2)))

def bilateral(im, d, sigmaColor, sigmaSpace):
    # return filtered result image
    return cv2.bilateralFilter(im, d, sigmaColor, sigmaSpace)

def joint_bilateral(im, guidance, d, sigmaColor, sigmaSpace):    
    # extending the images by mirroring pixels
    border_size = d // 2
    im_border = cv2.copyMakeBorder(im, border_size, border_size, border_size, border_size, cv2.BORDER_REFLECT)
    gu_border = cv2.copyMakeBorder(guidance, border_size, border_size, border_size, border_size, cv2.BORDER_REFLECT)

    # split the images into three channels
    im_channels = cv2.split(im_border)
    gu_channels = cv2.split(gu_border)
    res_channels = np.zeros([3, im.shape[0], im.shape[1]])

    # calculate spatial weight
    spatial_weight = np.zeros([d, d])
    for i in range(d):
        for j in range(d):
            spatial_diff = l2(-border_size + i, -border_size + j)
            spatial_weight[i, j] = gaussian(spatial_diff, sigmaSpace)

    int_weight = np.zeros([d, d])
    for i in range(3):
        for j in range(im.shape[0]):
            for k in range(im.shape[1]):
                # calculate intensity weight
                for l in range(d):
                    for m in range(d):
                        int_diff = l1(gu_channels[i][j + l, k + m], gu_channels[i][j + border_size, k + border_size])
                        int_weight[l, m] = gaussian(int_diff, sigmaColor)
                # calculate normalized weight
                weight = np.multiply(spatial_weight, int_weight)
                weight /= weight.sum()
                # apply the filter
                res_channels[i][j, k] = np.multiply(weight, im_channels[i][j : j + d, k : k + d]).sum()

    # return filtered result image
    return cv2.merge(res_channels)

if __name__ == '__main__':
    im = cv2.imread('./misc/Original_Bilateral.jpg').astype(np.float32)
    guidance = cv2.imread('./misc/GuidanceImage.jpg').astype(np.float32)
    
    d, sigmaColor, sigmaSpace = 7, 5, 5
    result_bilateral = bilateral(im, d, sigmaColor, sigmaSpace)
    result_joint_bilateral = joint_bilateral(im, guidance, d, sigmaColor, sigmaSpace)
    result_diff = 128 + abs(result_joint_bilateral - result_bilateral)

    cv2.imwrite('./results/Billateral.jpg', result_bilateral)
    cv2.imwrite('./results/JointBillateral.jpg', result_joint_bilateral)
    cv2.imwrite('./results/Bilateral_diff.jpg', result_diff)