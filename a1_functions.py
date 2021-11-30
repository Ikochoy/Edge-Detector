import numpy as np


def step_1_get_gaussian_matrix(ksize: int, scale:float):
    """
    Returns a 2D GaussianMatrix for input ksize and scale.
    :param ksize: The kernel size
    :param scale: sigma in the gaussian distribution
    :return: a 2D gaussian matrix for input ksize and scale.
    """
    # create array with [- (ksize-1)/2, ..., 0,... (ksize-1/2)]
    start, end = -(ksize-1)/ 2, (ksize-1)/2
    row_n_col = np.linspace(start, end, ksize)
    if ksize % 2 == 0:
        row_n_col[int(end): int(end)+2] = [0, 0]
    # compute with formula 1/(sqrt(2*pi)*sigma) * e^(-(x-mu)^2/2sigma^2)
    gaussian_exp = np.exp(-np.square(row_n_col)/(2*np.square(scale)))
    gaussian_mtx = (1/(np.sqrt(2*np.pi) *scale) * gaussian_exp)
    # divide the whole gaussian by np.sum such that it is normalized
    gaussian_mtx /= np.sum(gaussian_mtx)
    # Create the 2D matrix
    gaussian = np.outer(gaussian_mtx, gaussian_mtx)
    return gaussian


def pad_image(image, pad_widths):
    """
    Pad the image with 0s according to dimensions in pad_widths
    :param image: 2D matrix representation of image to be padded
    :param pad_widths: tuple wiht r_pad and c_pad which represents the number
    of rows to be added at the top and the bottom and the number of columns to
    be added at both sides respectively.
    :return: padded image 2D matrix representation
    """
    row_image, col_image = image.shape
    r_pad, c_pad = pad_widths
    result = np.zeros((row_image + 2* r_pad, col_image + 2* c_pad))
    for i in range(r_pad, result.shape[0]-r_pad):
        for j in range(c_pad, result.shape[1]-c_pad):
            result[i][j] = image[i-r_pad][j-c_pad]
    return result


def convolve(kernel, image):
    """
    Performs 2D convolution between kernel and image.
    :param kernel: a kernel matrix
    :param image: an image np array
    :return: the convolution result 2D matrix
    """
    # h_flip and v_flip the kernel
    kernel_flipflatten = np.flip(np.flip(kernel, 0), 1).flatten()
    # compute sizes of image matrices and kernel matrices, and value of k
    i_row, i_column = image.shape
    k_row, k_col = kernel.shape
    # initiate return result matrix of image dimensions
    result = np.zeros((i_row, i_column))
    # k_r & k_c should be equal since the kernel should be a square matrix
    k_r, k_c = (k_row-1)//2, (k_col-1)//2
    # pad the image with 0s, in order to allow convolution for corner cells
    image_padded = pad_image(image, (k_r, k_c))
    # compute cross correlation between flipped kernel and image
    for i in range(i_row):
        for j in range(i_column):
            # refer to formula in notes: G(i, j) = np.dot(f, t_ij)
            i_neighbouhood = image_padded[i:i+k_row, j:j+k_col].flatten()
            result[i][j] = np.dot(kernel_flipflatten, i_neighbouhood.T)
    return result


def step_2_compute_gradient_magnitude(gray_image, ksize=3, scale=0.5):
    """
    Computes the gradient magnitude given a grayscale image
    :param gray_image: a gray scale image represented by a numpy array
    :param ksize: the kernel size of the gaussian filter. Note that ksize
    should be odd number
    :param scale: the sigma of the gaussian filter
    :return: a gradient magnitude matrix of gray_image
    """
    # getting gaussian filter
    gauss = step_1_get_gaussian_matrix(ksize, scale)
    # getting sobel operators
    sobel_x = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = sobel_x.T
    # convolving gaussian and sobel operators first for easier computation
    gauss_cv_sobel_x = convolve(sobel_x, gauss)
    gauss_cv_sobel_y = convolve(sobel_y, gauss)

    # convolving with the image to compute the derivatives along x & y direction
    g_x = convolve(gauss_cv_sobel_x, gray_image)
    g_y = convolve(gauss_cv_sobel_y, gray_image)

    # compute the gradient
    gradient = np.sqrt(np.square(g_x) + np.square(g_y))
    return gradient


def step_3_threshold(gradient, epsilon=0.000000001):
    """
    Returns the gradient magnitude matrix after applying threshold such that
    values in the matrix becomes either 0 or 255.
    :param gradient: a gradient magnitude matrix of an image
    :param epsilon: a float that is very close to 0
    :return: a gradient magnitude matrix with values either equal to 0 or 255.
    """
    # step 1: compute tau_0
    grad_row, grad_col = gradient.shape
    tau_i = np.sum(gradient)/(grad_row * grad_col)
    loop = True
    i = 0
    while loop:
        # step 2: set i = 0, find cell value < or > than tau_0
        lower = gradient[gradient < tau_i]
        higher = gradient[gradient >= tau_i]
        # step 3: compute the mean of the lower and higher groups
        ml, mh = np.mean(lower), np.mean(higher)
        # record previous tau
        prev = tau_i
        # compute tau_i
        tau_i = (ml + mh) / 2
        # check whether we should continue looping
        loop = np.abs(tau_i - prev) > epsilon
        i += 1

    grad_copy = gradient.copy()
    grad_copy = (grad_copy >= tau_i).astype(int) * 255
    return grad_copy

