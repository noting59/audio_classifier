import numpy as np
import os
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


def get_test_file(path_to_file):
    ceps = np.load(path_to_file)
    num_ceps = len(ceps)

    return np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0)


neigh = NearestNeighbors(5, 2)
neigh.fit(X)

nbrs = neigh.radius_neighbors([get_test_file('/home/vlad/audio/test_cache/backstreet-boys_-_larger-than-life.npy')], 0.8, return_distance=False)

a = np.asarray(nbrs[0])

for i in a:
    print y[i]