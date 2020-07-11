import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from scipy.ndimage import convolve
from skimage.transform import resize
from cv2 import circle
from skimage.filters import sobel_h, sobel_v
# util
def read_img_gs(fname):
    # this function reads a colored image with file name fname and returns
    # a grayscale image as an numpy ndarray with intensity of [0...1]
    img = image.imread(fname)
    img = img[:,:,0]*0.2989 + img[:,:,1]*0.5870 + img[:,:,2]*0.1140
    if img.max() >= 2:
        img = img/255
    return img
def gen_2_D_Gaussian(size:int, sigma:float):
    # the function will take in the size and width and output a gaussian filter of shape (size, size), with
    # standard deviation sigma
    rtv = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = i-(size-1)/2
            v = j-(size-1)/2
            rtv[i, j] = gaussian(u, v, sigma)
    rtv = rtv/rtv.sum()
    return rtv
def gaussian(u, v, sigma):
    # this function compute the density function of a 2-D gaussian
    result = 1/(2*np.pi*sigma**2)*np.exp(-(u**2 + v**2)/sigma**2)
    return result
def display_blob_on_img(point_list, img, save=False, fname=""):
    # given a list of points at different scale, this function would display the points on the image
    # this function also have the option of saving the image if save=True
    three_channe_img = np.zeros((img.shape[0], img.shape[1], 3))
    # radius to display the blobs at
    radius_dict = {2:4, 4:4, 8:8, 16:16, 1:2}
    # color to display the blobs with
    color_dict = {1:(0, 0, 1), 2:(0, 1, 0), 4:(1, 1, 0), 8:(1, 0, 1), 16:(1, 0, 0)}
    three_channe_img[:, :, 0] = img
    three_channe_img[:, :, 1] = img
    three_channe_img[:, :, 2] = img
    if img.max() >= 2:
        three_channe_img = three_channe_img / 255
    for item in point_list:
        circle(three_channe_img, (item[1], item[0]), radius=radius_dict[item[2]], color=color_dict[item[2]], thickness=-1)
    plt.imshow(three_channe_img)
    if save:
        plt.savefig(fname)
    else:
        plt.show()
def display_points_on_img(edge_mask, img, save=False, fname="", rad = 1, point_list = None):
    # this function displays the mask on the image, with option to save the image as a file
    three_channe_img = np.zeros((img.shape[0], img.shape[1], 3))
    three_channe_img[:, :, 0] = img
    three_channe_img[:, :, 1] = img
    three_channe_img[:, :, 2] = img
    contour = []
    for i in range(0, img.shape[0]):
        for j in range(img.shape[1]):
            if edge_mask[i, j] == 1:
                contour.append([i, j])
    for item in contour:
        # three_channe_img[item[0], item[1], 0] = 1
        circle(three_channe_img, (item[1], item[0]), radius=rad, color=(255, 0, 0), thickness=-1)
    plt.imshow(three_channe_img)
    if save:
        plt.savefig(fname)
    else:
        plt.show()
# Part 2 Harris Corner
def harris_corner(img, alpha = 0.04, thre = 2, sigma = 1, supression_window_size = 3):
    # this function will take in an image, optional paramter alpha, threshold and sigma
    # then output a binary mask with corners labeled by 1s and none corder labeled by 0s.
    # the function first computes the harris response, then perform thresholding and finishes
    # off with a non-maxima supression with a selected windowsize
    # the mask returned will be the same size as the image input
    window = gen_2_D_Gaussian(5, sigma)
    harris_response = np.zeros(img.shape)
    # finding the derivative of the image
    image_dx = sobel_h(img)
    image_dy = sobel_v(img)
    # computer second moment matrix and calculate
    for x in range(2, img.shape[1]-2):
        for y in range(2, img.shape[0]-2):
            M_xy = get_second_moment(image_dx, image_dy, window, y, x)
            s = np.linalg.eigvals(M_xy)
            harris_response[y,x] = s[0]*s[1] - alpha*(s[0]+s[1])**2

    # plt.imshow(harris_response, cmap="gray")
    # plt.show()
    # perform thresholding
    mask = np.where(harris_response >= thre, 1, 0)
    # non-maxima supression
    mask = non_maxima_supression(harris_response, mask, supression_window_size)
    return mask
def get_second_moment(ix, iy, window, y, x):
    # this function returns a 2x2 second moment matrix around value x, y by computed using
    # I_x, I_y, and the 5x5 gaussian window passed in
    M = np.zeros((2,2))
    img_window_x = ix[y-2:y+3, x-2:x+3]
    img_window_y = iy[y-2:y+3, x-2:x+3]
    for i in range(0, 5):
        for j in range(0, 5):
            M[0, 0] = M[0, 0] + window[i, j] * (img_window_x[i, j]**2)
            M[0, 1] = M[0, 1] + window[i, j] * img_window_y[i, j] * img_window_x[i, j]
            M[1, 0] = M[1, 0] + window[i, j] * img_window_y[i, j] * img_window_x[i, j]
            M[1, 1] = M[1, 1] + window[i, j] * (img_window_y[i, j]**2)
    return M
def non_maxima_supression(harris_response, mask, window_size=3):
    # This function performs non-maxima supression on the binary mask passed in. It uses the harris_response matrix
    # to compute response intensity, and select the maximum pixel. The window_size is a parameter that can be changed
    h, w = harris_response.shape
    delta_h = int(np.floor(window_size/2))
    for y in range(delta_h, h - delta_h):
        for x in range(delta_h, w - delta_h):
            if mask[y, x] == 1:
                window = harris_response[y-delta_h:y+delta_h+1, x-delta_h:x+delta_h + 1]
                max_y, max_x = np.unravel_index(window.argmax(), window.shape)
                mask[y-delta_h:y+delta_h+1, x-delta_h:x+delta_h + 1] = np.zeros(window.shape)
                mask[y+max_y-1, x+max_x-1] = 1

    return mask
def run_harris_corner(img, scale=10, thre=0.00002, supression_window_size=3, alpha=0.04, sigma=1, rad = 1):
    img = resize(img, np.round(np.array(img.shape)/scale))
    mask = harris_corner(img, thre=thre, supression_window_size=supression_window_size, alpha=alpha, sigma=sigma)
    display_points_on_img(mask, img, rad = rad)

# part 3 DoG, Gaussian pyramid and blob detection
def gen_gaussian_pyramid(img):
    # returns a 7 level gaussian pyramid with specified sigmas
    # return a list of numpy arrays
    gauss_filters = []
    pyramid = []
    sigmas = [1, 2, 4, 8, 16, 32, 64]
    for i in sigmas:
        gauss_filters.append(gen_2_D_Gaussian(i*2+1, i))
    for i in range(0, 7):
        blurred = convolve(img, gauss_filters[i], mode="nearest")
        downed = subsample(blurred, factor=2**i)
        pyramid.append(downed)
    return pyramid
def subsample(img, factor = 2):
    # subsample the image img with provided factor
    # returns a numpy array with 1/factor the size
    # rounds down if the orignal image size is not divisable by the factor
    h, w = img.shape
    sub_img = img[:,0].reshape((h, 1))
    for x in range(factor, w, factor):
        sub_img = np.concatenate((sub_img, img[:, x][:,None]), axis=1)
    # subsample in the y direction
    output_img = sub_img[0, :].reshape((1, sub_img.shape[1]))
    for y in range(factor, h, factor):
        output_img = np.concatenate((output_img, sub_img[y, :][None,:]), axis=0)
    return output_img
def upsample(img, factor=2, mode="constant"):
    # upsample the image with given factor
    # can choose between linear interpolation or constant
    # returns a numpy array
    if mode == "constant":
        h, w = img.shape
        wider_img = img[:, 0].reshape((h, 1))
        wider_img = np.concatenate((wider_img, img[:, 0][:, None]), axis=1)
        for x in range(1, w):
            wider_img = np.concatenate((wider_img, img[:, x][:, None]), axis=1)
            wider_img = np.concatenate((wider_img, img[:, x][:, None]), axis=1)
        output_img = wider_img[0, :].reshape((1, w*2))
        output_img = np.concatenate((output_img, wider_img[0, :][None, :]), axis=0)
        for y in range(1, h):
            output_img = np.concatenate((output_img, wider_img[y, :][None, :]), axis=0)
            output_img = np.concatenate((output_img, wider_img[y, :][None, :]), axis=0)
        return output_img
    else:
        h, w = img.shape
        output = np.zeros((h*factor, w*factor))
        for y in range(0, h):
            for x in range(0, w):
                output[y*factor, x*factor] = img[y,x]
        for y in range(0, h):
            for k in range(1, factor):
                if y != h-1:
                    output[y*factor+k, :] = output[y*factor, :] + (output[(y+1)*factor, :] - output[y*factor, :])/factor * k
                else:
                    output[y*factor+k, :] = output[-factor, :]
        for x in range(0, h):
            for k in range(0, factor):
                if x != h - 1:
                    output[:, x*factor+k] = output[:, x*factor] + (output[:,(x+1)*factor] - output[:, x*factor])/factor * k
                else:
                    output[:, x * factor + k] = output[:, -factor]
        return output
def gen_dog_pyramid(img):
    # uses the gaussian pyramid function to generate difference of gaussian pyramid
    # upsamples image at a lower scale in order to compute the difference
    # return a list of 6 numpy arrays
    g_pyramid = gen_gaussian_pyramid(img)
    dof_pyramid = []
    for i in range(0, len(g_pyramid)-1):
        dof_pyramid.append(upsample(g_pyramid[i+1], mode="linear") - g_pyramid[i])
    return dof_pyramid
def gen_composite_image(pyramid):
    # given a gaussian/dog pyramid (list of numpy arrays), generate a composite image
    # returns a composite image
    rows, cols = img.shape
    composite_image = np.zeros((rows, cols + cols // 2))
    composite_image[:rows, :cols] = pyramid[0]
    i_row = 0
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows
    return composite_image
def find_extrema(img, window_size=3, thre=0.1, scale=1):
    # given an image, windowsize, threshold, and scale
    # slide the window across the image, find the local maximum/minimum in each window and label it as one.
    # then upscale the point to a coordinate in the original image
    # returns a list of tuples [(x, y, scale)]
    img = np.abs(img)
    h, w = img.shape
    delta_h = int(np.floor(window_size / 2))
    rtv = np.zeros((h, w))
    for y in range(delta_h, h - delta_h):
        for x in range(delta_h, w - delta_h):
            center_val = img[y,x]
            window = img[y - delta_h:y + delta_h + 1, x - delta_h:x + delta_h + 1].copy().flatten()
            window.sort()
            window_second_largest = (window[-2] + window[-3])/2
            if center_val > window_second_largest + thre:
                rtv[y,x] = 1
    rtl = []
    for y in range(0, h):
        for x in range(0, w):
            if rtv[y, x] == 1:
                rtl.append([int(y*scale), int(x*scale), scale])
    return rtl
def blob_detection(img):
    parameter = [(5, 0.05, 1),(3, 0.07, 2), (3, 0.09, 4), (3, 0.075, 8), (5, 0.04, 16)]
    extremas = []
    pyramid = gen_dog_pyramid(img)
    for i in range(0, 5):
        extremas = extremas + find_extrema(pyramid[i], parameter[i][0], parameter[i][1], parameter[i][2])
    display_blob_on_img(extremas, img, save=False)

if __name__ == "__main__":
    img = read_img_gs("sunflower.jpg")
    run_harris_corner(img)


