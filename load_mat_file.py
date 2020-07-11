from scipy.io import loadmat
sift_features = loadmat('sift_features.mat')

print(sift_features.keys())
features_1 = sift_features["features_1"]
keypoints_1 = sift_features["keypoints_1"]

features_2 = sift_features["features_2"]
keypoints_2 = sift_features["keypoints_2"]

theta = sift_features["theta"]