import fnmatch
import tempfile
import os
import pydub
import scipy
import scipy.io.wavfile
import sys
import numpy as np
from scikits.talkbox.features import mfcc


def read_mp3(file_path, as_float = False):
    """
    Read an MP3 File into numpy data.
    :param file_path: String path to a file
    :param as_float: Cast data to float and normalize to [-1, 1]
    :return: Tuple(rate, data), where
        rate is an integer indicating samples/s
        data is an ndarray(n_samples, 2)[int16] if as_float = False
            otherwise ndarray(n_samples, 2)[float] in range [-1, 1]
    """
    mp3 = pydub.AudioSegment.from_mp3(file_path)
    _, path = tempfile.mkstemp()
    mp3.export(path, format="wav")
    rate, data = scipy.io.wavfile.read(path)
    os.remove(path)
    if as_float:
        data = data/(2**15)
    return rate, data


def read_files_recursive(directory_path, pattern = '*.mp3'):
    matches = []

    for root, dirnames, filenames in os.walk(directory_path):
        for filename in fnmatch.filter(filenames, pattern):
            mp3tofmcc(os.path.join(root, filename), filename, file_path_cache)
            matches.append(os.path.join(root, filename))

    return matches


def mp3tofmcc(file_path, filename, file_path_output):
    path, ext = os.path.splitext(filename)
    rate, data = read_mp3(file_path)
    ceps, mspec, spec = mfcc(data)
    np.save(file_path_output + path, ceps)


file_path = sys.argv[1]
file_path_cache = sys.argv[2]

print read_files_recursive(file_path, '*.mp3')