import numpy as np

# load files
data = np.load('train_sequences_updated.npy')
labels = np.load('train_labels_updated.npy')

# basic info
# print(data.shape)      # (1896, 60, 17, 2)
# # print(data.dtype)      # float32
# print(labels.shape)    # (1896,)

# # catching content
# print(data[0])         # first video
# print(data[0, 0])      # first frame of first video (17 keypoints)
# print(data[0, 0, 0])   # first video, first frame, first keypoint (x, y)

# label count
print(np.unique(labels))  # what classes are there