import cv2
import numpy as np

def get_interest_points(image, feature_width):
    """ Returns a set of interest points for the input image
    Args:
        image - can be grayscale or color, your choice.
        feature_width - in pixels, is the local feature width. It might be
            useful in this function in order to (a) suppress boundary interest
            points (where a feature wouldn't fit entirely in the image)
            or (b) scale the image filters being used. Or you can ignore it.
    Returns:
        x and y: n x 1 vectors of x and y coordinates of interest points.
        confidence: an n x 1 vector indicating the strength of the interest
            point. You might use this later or not.
        scale and orientation: are n x 1 vectors indicating the scale and
            orientation of each interest point. These are OPTIONAL. By default you
            do not need to make scale and orientation invariant local features. 
    """
    h, w = image.shape[:2]

    # hyperparameters
    window_size = 9
    neighbor_size = 5
    filter_size = 5
    alpha = 0.06
    threshold = 0.01

    # Gaussian smoothing
    im_blurred = cv2.GaussianBlur(image, (filter_size, filter_size), 0)

    # add padding
    half_size = window_size // 2
    im_padded = cv2.copyMakeBorder(im_blurred, half_size, half_size, half_size, half_size, cv2.BORDER_REFLECT)

    # compute image gradients
    # note that cv2.spatialGradient() only accept input with type CV_8UC1
    # so, we first multiply the input by 255 and then divide the output by 255
    I_x, I_y = cv2.spatialGradient(np.uint8(im_padded * 255))
    I_x, I_y = I_x / 255.0, I_y / 255.0
    I_xx, I_xy, I_yy = I_x * I_x, I_x * I_y, I_y * I_y
    
    res = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            # compute matrix M based on image gradients
            M_11 = np.sum(I_xx[i:i + window_size, j:j + window_size])
            M_21 = np.sum(I_xy[i:i + window_size, j:j + window_size])
            M_22 = np.sum(I_yy[i:i + window_size, j:j + window_size])

            # compute corner response R
            det_M = M_11 * M_22 - (M_21 ** 2)
            tr_M = M_11 + M_22
            R = det_M - alpha * (tr_M ** 2)

            # find points with large corner response
            if R > threshold:
                res[i, j] = R

    for i in range(h - neighbor_size):
        for j in range(w - neighbor_size):
            # only preserve points of local maxima of R
            window = res[i:i + neighbor_size, j:j + neighbor_size]
            local_max = np.max(window)
            window[window != local_max] = 0
            res[i:i + neighbor_size, j:j + neighbor_size] = window

    # output all points with R > 0
    x, y = (res > 0).nonzero()
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)

    return y, x
    
def get_features(image, x, y, feature_width):
    """ Returns a set of feature descriptors for a given set of interest points. 
    Args:
        image - can be grayscale or color, your choice.
        x and y: n x 1 vectors of x and y coordinates of interest points.
            The local features should be centered at x and y.
        feature_width - in pixels, is the local feature width. You can assume
            that feature_width will be a multiple of 4 (i.e. every cell of your
            local SIFT-like feature will have an integer width and height).
        If you want to detect and describe features at multiple scales or
            particular orientations you can add other input arguments.
    Returns:
        features: the array of computed features. It should have the
            following size: [length(x) x feature dimensionality] (e.g. 128 for
            standard SIFT)
    """
    x, y = y, x
    features = np.zeros((x.shape[0], feature_width * 8))

    # Gaussian smoothing
    filter_size = 3
    im_blurred = cv2.GaussianBlur(image, (filter_size, filter_size), 0)

    # add padding
    half_width = feature_width // 2
    im_padded = cv2.copyMakeBorder(im_blurred, half_width, half_width, half_width, half_width, cv2.BORDER_CONSTANT, value=0)
    x, y = x + half_width, y + half_width

    # compute image gradients
    G_x, G_y = cv2.spatialGradient(np.uint8(im_padded * 255))
    G_x, G_y = G_x / 255.0, G_y / 255.0

    # compute magnitude and orientation
    mag = np.sqrt(G_x ** 2 + G_y ** 2)
    orient = np.arctan2(G_y, G_x)
    orient[orient < 0] += 2 * np.pi

    # obtain Gaussian kernel
    Gauss_kernel = cv2.getGaussianKernel(feature_width, half_width)

    quad_width = feature_width // 4
    values = np.zeros((quad_width, quad_width, 8))
    for k in range(x.shape[0]):
        window_start_x, window_start_y = x[k] - half_width, y[k] - half_width
        window_end_x, window_end_y = window_start_x + feature_width, window_start_y + feature_width

        # apply Gaussian kernel on magnitude
        window = cv2.sepFilter2D(mag[window_start_x:window_end_x, window_start_y:window_end_y], 
                                 ddepth = -1, 
                                 kernelX = Gauss_kernel, 
                                 kernelY = Gauss_kernel)
        for i in range(quad_width):
            for j in range(quad_width):
                cell_start_x, cell_start_y = window_start_x + i * quad_width, window_start_y + j * quad_width
                cell_end_x, cell_end_y = cell_start_x + quad_width, cell_start_y + quad_width

                # cast orientations into 8 bins and accumulate Gaussian weighted gradient magnitude
                values[i, j], _ = np.histogram(orient[cell_start_x:cell_end_x, cell_start_y:cell_end_y], 
                                               bins=8, 
                                               range=(0, 2 * np.pi), 
                                               weights=window[cell_start_x - window_start_x:cell_end_x - window_start_x, 
                                                              cell_start_y - window_start_y:cell_end_y - window_start_y])
        
        # concatenate and normalize values
        features[k] = values.flatten() / np.linalg.norm(values)
    return features

def match_features(features1, features2, threshold=0.0):
    """ 
    Args:
        features1 and features2: the n x feature dimensionality features
            from the two images.
        threshold: a threshold value to decide what is a good match. This value 
            needs to be tuned.
        If you want to include geometric verification in this stage, you can add
            the x and y locations of the features as additional inputs.
    Returns:
        matches: a k x 2 matrix, where k is the number of matches. The first
            column is an index in features1, the second column is an index
            in features2. 
        Confidences: a k x 1 matrix with a real valued confidence for every
            match.
        matches' and 'confidences' can be empty, e.g. 0x2 and 0x1.
    """
    num_features = max(features1.shape[0], features2.shape[0])
    matched = np.zeros((num_features, 2))
    confidence = np.random.rand(num_features, 1)

    for i in range(features1.shape[0]):
        # compute feature distances
        dist = np.linalg.norm(features1[i] - features2, axis=1)

        # set matching pair
        matched[i, 0], matched[i, 1] = i, np.argmin(dist)

        # get nearest and second nearest distance
        smallest, second_smallest = np.partition(dist, 1)[0:2]

        # compute confidence = 1 / ratio
        confidence[i] = second_smallest / smallest

    # only preserve matching pairs whose confidence exceeds the threshold
    matched = matched[np.squeeze(confidence > threshold)]
    confidence = confidence[np.squeeze(confidence > threshold)]

    # sort the matching pairs
    order = np.argsort(confidence, axis=0)[::-1, 0]
    confidence = confidence[order, :]
    matched = matched[order, :]

    return matched, confidence
