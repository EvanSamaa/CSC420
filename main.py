import numpy as np
from util import *
from matplotlib import pyplot as plt
from skimage.filters import prewitt_h, prewitt_v
from skimage.transform import rotate, rescale
from cv2 import line
from scipy.io import loadmat

# ====================================  Part 1  ====================================
def calc_grad(pyramid_dx, pyramid_dy, point, filter):
    scale_index_dict = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}
    index_scale_dict = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16}
    padded_img_level_dx = pyramid_dx[scale_index_dict[point[2]]]
    padded_img_level_dy = pyramid_dy[scale_index_dict[point[2]]]
    window = np.zeros((16, 16, 3))
    for i in range(-7, 9):
        for j in range(-7, 9):
            dx_val = padded_img_level_dx[point[0] + i, point[1] + j]
            dy_val = padded_img_level_dy[point[0] + i, point[1] + j]
            window[7+i, 7+j, 0] = np.sqrt(dx_val**2 + dy_val**2)
            angle = np.arctan(dy_val/dx_val)
            if dx_val== 0 or angle == np.NaN or angle == np.nan or angle == np.NAN:
                if dy_val > 0:
                    angle = np.pi/2
                else:
                    angle = -np.pi/2
            window[7+i, 7-j, 1] = angle
    window[:, :, 2] = np.multiply(filter, window[:, :, 0])
    return window
def get_sift(pyramid_dx, pyramid_dy, point, filter):
    window = calc_grad(pyramid_dx, pyramid_dy, point, filter)
    rtv = np.zeros((39, ))
    sift_rtv = np.zeros((36, ))
    mod_window = np.round(window[:, :, 1] * 180/np.pi/10)
    for x in range(0, 16):
        for y in range(0, 16):
            index = int(mod_window[x, y])
            sift_rtv[index] += window[x, y, 2]
    max_bin = sift_rtv.argmax()
    before_max_sub_list = sift_rtv[0:max_bin]
    after_max_sub_list = sift_rtv[max_bin:]
    sift_rtv = np.concatenate((after_max_sub_list, before_max_sub_list))
    rtv[0:3] = point
    rtv[3:] = sift_rtv
    # plt.hist(sift_rtv, bins=36)
    # plt.show()
    return rtv
def display_quiver(pyramid_dx, pyramid_dy, point, filter_bool=False, filter=None):
    scale_index_dict = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}
    X = np.zeros((16, 16))
    Y = np.zeros((16, 16))
    padded_img_level_dx = pyramid_dx[scale_index_dict[point[2]]]
    padded_img_level_dy = pyramid_dy[scale_index_dict[point[2]]]
    window = np.zeros((16, 16, 2))
    for i in range(-7, 9):
        for j in range(-7, 9):
            dx_val = padded_img_level_dx[point[0] + i, point[1] + j]
            dy_val = padded_img_level_dy[point[0] + i, point[1] + j]
            mod_i = 7+i
            mod_j = 7 + j
            window[mod_i, mod_j, 0] = dx_val
            window[mod_i, mod_j, 1] = dy_val
            X[mod_i, mod_j] = mod_i
            Y[mod_i, mod_j] = mod_j
    if filter_bool:
        window[:, :, 0] = np.multiply(window[:, :, 0], filter)
        window[:, :, 1] = np.multiply(window[:, :, 1], filter)
        plt.quiver(X, Y, window[:, :, 0], window[:, :, 1])
    else:
        plt.quiver(X, Y, window[:,:,0], window[:,:,1])
    plt.show()
def generate_SIFT_Descriptors(img):

    # gaussian_pyramid = load_pyramid("gaussian_pyramid")
    # dog_pyramid = load_pyramid("dog_pyramid")
    gaussian_pyramid = gen_gaussian_pyramid(img)
    dog_pyramid = gen_dog_pyramid(img)
    padded_grad_of_gaussian_pyramid_x = []
    padded_grad_of_gaussian_pyramid_y = []
    for level in gaussian_pyramid:
        padded_grad_of_gaussian_pyramid_y.append(np.pad(prewitt_h(level), 8))
        padded_grad_of_gaussian_pyramid_x.append(np.pad(prewitt_v(level), 8))
    filter = gen_2_D_Gaussian(16, 4)
    extremas = blob_detection(img, dog_pyramid, upscaling=False)
    sift_list = []
    for point in extremas:
        sift_descriptor = get_sift(padded_grad_of_gaussian_pyramid_x, padded_grad_of_gaussian_pyramid_y, point, filter)
        sift_list.append(sift_descriptor)
    return sift_list
def augment_image(img, x0, y0, theta, s):
    # img_after = rotate(img, theta, s, (x0, y0))
    if False:
        # img = rotate(img, theta, (x0, y0))
        img_frame = np.zeros((img.shape[0]*3, img.shape[1]*3))
        img_resized = rescale(img, s)
        # center_before = [int(img_frame.shape[0]/2), int(img_frame.shape[1]/2)]
        center_after = [int(img_resized.shape[0]/2), int(img_resized.shape[1]/2)]
        center_before = [x0, y0]
        diff = [center_before[0] - center_after[0], center_before[1] - center_after[1]]
        # to ensure it always fits
        img_frame[img.shape[0] + diff[0]:img.shape[0] + diff[0]+img_resized.shape[0],
        img.shape[0] + diff[1]:img.shape[1] + diff[1]+img_resized.shape[1]] = img_resized
        final_img = rotate(img_frame, theta, resize=False, center=(img.shape[0] + x0, img.shape[1] + y0))
        final_img = final_img[img.shape[0]: 2*img.shape[0] , img.shape[1]: 2*img.shape[1]]
    else:
        # img = rotate(img, theta, (x0, y0))
        img_frame = np.zeros((img.shape[0]*3, img.shape[1]*3))
        img_resized = rescale(img, s)
        # center_after = [int(img_resized.shape[0] / 2), int(img_resized.shape[1] / 2)]
        center_after = [int(x0*s), int(y0*s)]
        center_before = [int(img_frame.shape[0] / 2), int(img_frame.shape[1] / 2)]
        diff = [center_before[0] - center_after[0], center_before[1] - center_after[1]]
        # to ensure it always fits
        img_frame[diff[0]:diff[0]+img_resized.shape[0],diff[1]:diff[1]+img_resized.shape[1]] = img_resized
        final_img = rotate(img_frame, theta, resize=False, center=center_before)
        final_img = final_img[img.shape[0]: 2*img.shape[0] , img.shape[1]: 2*img.shape[1]]
    return final_img
def compare_sift(img1, img2, siftList1, siftList2, location1, location2, window_size=200, up_scaled = False):
    sift_points_in_bound_box_1 = []
    sift_points_in_bound_box_2 = []

    for i in range(0, siftList1.shape[1]):
        if up_scaled == False:
            siftList1[0, i] = siftList1[0, i] * siftList1[2, i]
            siftList1[1, i] = siftList1[1, i] * siftList1[2, i]
        if siftList1[0, i] >= location1[0] - window_size and siftList1[0, i] < location1[0] + window_size and \
                siftList1[1, i] >= location1[1] - window_size and siftList1[1, i] < location1[1] + window_size:
            sift_points_in_bound_box_1.append(siftList1[:, i])
    for i in range(0, siftList2.shape[1]):
        siftList2[0, i] = siftList2[0, i] * siftList2[2, i]
        siftList2[1, i] = siftList2[1, i] * siftList2[2, i]
        if siftList2[0, i] >= location2[0] - window_size and siftList2[0, i] < location2[0] + window_size and \
                siftList2[1, i] >= location2[1] - window_size and siftList2[1, i] < location2[1] + window_size:
            sift_points_in_bound_box_2.append(siftList2[:, i])
    best_match_1 = []
    best_match_2 = []
    if len(sift_points_in_bound_box_1) >= len(sift_points_in_bound_box_2):
        temp = sift_points_in_bound_box_1
        sift_points_in_bound_box_1 = sift_points_in_bound_box_2
        sift_points_in_bound_box_2 = temp
    for feature_1 in sift_points_in_bound_box_1:
        best = -100
        best_1 = None
        best_2 = None
        for feature_2 in sift_points_in_bound_box_2:
            b_c = compute_Bhattacharyya_coefficient(feature_1, feature_2)
            if b_c >= best:
                best_1 = feature_1
                best_2 = feature_2
                best = b_c
        if best >= 0.9:
            best_match_1.append(best_1)
            best_match_2.append(best_2)
    print(len(best_match_1))
    composed_image = np.zeros((img1.shape[0], img1.shape[1]*2 + 10, 3))
    print(composed_image.shape)
    composed_image[:, 0:img1.shape[1], 0] = img1
    composed_image[:, img1.shape[1]+10:, 0] = img2
    composed_image[:, :, 1] = composed_image[:, :, 0]
    composed_image[:, :, 2] = composed_image[:, :, 0]
    for i in range(0, len(best_match_2)):
        if up_scaled == False:
            pt1 = (int(best_match_1[i][1]), int(best_match_1[i][0]))
            pt2 = (int(best_match_2[i][1])+ img1.shape[1] + 10, int(best_match_2[i][0]))
        composed_image = line(composed_image, pt1, pt2, color=(1, 0, 0), thickness=3)
    plt.imshow(composed_image)
    plt.show()

def compute_Bhattacharyya_coefficient(x1, x2):
    coefficient = np.sqrt(np.multiply(x1[3:]/x1[3:].sum(), x2[3:]/x2[3:].sum())).sum()
    return coefficient
def mat_lab_provided_code():
    sift_features = loadmat('sift_features.mat')
    print(sift_features.keys())
    features_1 = sift_features["features_1"]
    keypoints_1 = sift_features["keypoints_1"]

    features_2 = sift_features["features_2"]
    keypoints_2 = sift_features["keypoints_2"]
    theta = sift_features["theta"]
    img = np.array(sift_features['image'])
    key_points_1 = np.array(keypoints_1) # (4, 962)
    features1 = np.array(features_1) # (128, 962)
    key_points_1_mod = []
    for i in range(0, key_points_1.shape[1]):
        key_points_1_mod.append([key_points_1[0, i], key_points_1[1, i], 4])
    display_blob_on_img(key_points_1_mod, img)
    # plt.imshow(img, cmap="gray")
    # plt.show()
if __name__ == "__main__":
    mat_lab_provided_code()
    A[1]
    img = read_img_gs("UofT.jpg")
    # save_pyramid(gen_gaussian_pyramid(img), name="gaussian_pyramid")
    # save_pyramid(gen_dog_pyramid(img), name="dog_pyramid")
    smol_img = augment_image(img, 200, 500, -15, 0.8)
    large_img = augment_image(img, 200, 500, 15, 1.5)
    # save_sift(generate_SIFT_Descriptors(smol_img), name="smol_img_sift_descriptors")
    # save_sift(generate_SIFT_Descriptors(large_img), name="large_img_sift_descriptors")
    smol_sifts = np.load("temp/smol_img_sift_descriptors.npy")
    large_sifts = np.load("temp/large_img_sift_descriptors.npy")
    original_sift = np.load("temp/original_img_sift_descriptors.npy")
    compare_sift(large_img, img, large_sifts, original_sift, [512, 512], [200, 500], window_size=200)
    # blob_detection(smol_img, gen_dog_pyramid(smol_img))

