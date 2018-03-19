import numpy as np
import os

from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import NearestNeighbors

X = []
y = []

file_path_cache = '/home/vlad/audio/dataset_cache/'

for file in os.listdir(file_path_cache):
    ceps = np.load(file_path_cache + file)
    num_ceps = len(ceps)
    y.append(file)
    X.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))

X = np.array(X)
y = np.array(y)
clf = NearestCentroid()
clf.fit(X, y)
NearestCentroid(metric='euclidean', shrink_threshold=None)


ceps = np.load('/home/vlad/audio/test_cache/sergey-babkin_-_de-bi-ya.npy')
num_ceps = len(ceps)
predict = np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0)

print(clf.predict([predict]))