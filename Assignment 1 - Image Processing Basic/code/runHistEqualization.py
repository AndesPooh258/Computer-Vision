from matplotlib import pyplot as plt
import numpy as np
import cv2

def save_histogram(im, label="CDF", output="output"):
    num_bins = 256
    hist, _ = np.histogram(im.flatten(), num_bins, density=True)
    cdf = hist.cumsum()
    cdf = (num_bins - 1) * cdf / cdf[-1]
    plt.plot(range(num_bins), cdf, label=label, color="r")
    plt.xlabel('Pixel intensity')
    plt.ylabel('Number of pixels')
    plt.legend()
    plt.savefig("./results/" + output + ".png")
    plt.clf()

# this method mainly follows the histogram_equalization() in Tutorial 3
def histogram_equalization(im):
    # calculate the histogram with normalization
    num_bins = 256
    hist, bin_edges = np.histogram(im.flatten(), num_bins, density=True)

    # compute the targeted projection funtion
    cdf = hist.cumsum()
    cdf = (num_bins - 1) * cdf / cdf[-1]
    
    # equalize the image with linear interpolation
    result = np.interp(im.flatten(), bin_edges[:-1], cdf).reshape(im.shape)

    # return filtered result image
    return result

def local_histogram_equalization(im):
    # extending the image by mirroring pixels
    filter_size = 65
    border_size = filter_size // 2
    im_border = cv2.copyMakeBorder(im, border_size, border_size, border_size, border_size, cv2.BORDER_REFLECT)
    result = np.zeros_like(im)

    # perform histogram equalization based on the histogram of a square surrounding the pixel
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            neighbor = im_border[i: i + filter_size, j: j + filter_size]
            result[i, j] = histogram_equalization(neighbor)[border_size, border_size]

    # return filtered result image
    return result

if __name__ == '__main__':
    im = cv2.imread('./misc/Original_HistEqualization.jpeg')

    result_hist_equalization = histogram_equalization(im)
    result_local_hist_equalization = local_histogram_equalization(im)

    cv2.imwrite('./results/HistoEqualization.jpg', result_hist_equalization)
    cv2.imwrite('./results/LocalHistoEqualization.jpg', result_local_hist_equalization)

    save_histogram(im, label="CDF before Equalization", output="cdf_original")
    save_histogram(result_hist_equalization, label="CDF after Equalization", output="cdf_hist_equalization")
    save_histogram(result_local_hist_equalization, label="CDF after Local Equalization", output="cdf_local_hist_equalization")