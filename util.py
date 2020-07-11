import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from scipy.ndimage import convolve
from skimage.transform import resize
from cv2 import circle
from skimage.filters import sobel_h, sobel_v
import os
# utility functions
def save_sift(sift_list, name):
    sift_arr = sift_list[0].reshape((sift_list[0].shape[0], 1))
    for i in sift_list[1:]:
        sift_arr = np.concatenate((sift_arr, i.reshape((sift_list[0].shape[0], 1))), axis=1)
    file_name = "./temp/{}.npy"
    np.save(file_name.format(name), sift_arr)
    print("successfully saved")
def save_pyramid(pyramid, name):
    file_name = "./temp/{}_{}.npy"
    i = 0
    for item in pyramid:
        np.save(file_name.format(name, i), item)
        i = i + 1
def load_pyramid(name=None):
    pyramid = []
    file_name = "./temp/{}_{}.npy"
    dir_list = os.listdir("./temp")
    dir_list.sort()
    for i in range(0, 6):
        pyramid.append(np.load(file_name.format(name, i)))
    if name == "gaussian_pyramid":
        pyramid.append(np.load(file_name.format(name, 6)))
    return pyramid


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
def read_img_gs(fname):
    # this function reads a colored image with file name fname and returns
    # a grayscale image as an numpy ndarray with intensity of [0...1]
    img = image.imread(fname)
    img = img[:,:,0]*0.2989 + img[:,:,1]*0.5870 + img[:,:,2]*0.1140
    if img.max() >= 2:
        img = img/255
    return img
# part 3 DoG, Gaussian pyramid and blob detection
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
def gen_composite_image(pyramid, img):
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
def find_extrema(img, window_size=3, thre=0.1, scale=1, upscale = True):
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
                if upscale:
                    rtl.append([int(y*scale), int(x*scale), scale])
                else:
                    rtl.append([int(y), int(x), scale])
    return rtl
def blob_detection(img, pyramid, upscaling=True):
    parameter = [(5, 0.05, 1),(3, 0.07, 2), (3, 0.09, 4), (3, 0.075, 8), (5, 0.04, 16)]
    # parameter = [(3, 0.05, 1), (3, 0.1, 2), (3, 0.1, 4), (3, 0.1, 8), (5, 0.1, 16)]
    extremas = []
    for i in range(0, 5):
        extremas = extremas + find_extrema(pyramid[i], parameter[i][0], parameter[i][1], parameter[i][2], upscale=upscaling)
    display_blob_on_img(extremas, img, save=False)
    return extremas