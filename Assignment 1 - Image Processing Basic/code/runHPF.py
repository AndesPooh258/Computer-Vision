import numpy as np
import cv2

def hpf_fourier(im, hpf_w=60):
    # conduct Fourier transform to convert the image to the frequency domain
    ft = np.fft.fft2(im)

    # shift zero-frequency component to the center of the spectrum
    ft_shifted = np.fft.fftshift(ft)

    # conduct high-pass filter with the region hpf_w * hpf_w
    filter = 255 - np.zeros_like(im)
    start_x = (filter.shape[0] - hpf_w) // 2
    start_y = (filter.shape[1] - hpf_w) // 2
    end_x = start_x + hpf_w
    end_y = start_y + hpf_w
    cv2.rectangle(filter, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
    ft_hpf = np.multiply(ft_shifted, filter) / 255

    # conduct the inverse zero-frequency component shift to the default setting
    ft_ishifted = np.fft.ifftshift(ft_hpf)

    # conduct inverse Fourier transform to convert back to the time domain
    result = np.abs(np.fft.ifft2(ft_ishifted))

    # return transformed result image
    return result

if __name__ == '__main__':
    im = cv2.imread('./misc/lena_gray.bmp', 0)
    
    hpf_w = 60
    result = hpf_fourier(im, hpf_w)
    cv2.imwrite('./results/hpf_fourier.png', result)