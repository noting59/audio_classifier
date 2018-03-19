import tempfile
import os
import pydub
import scipy
import scipy.io.wavfile
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

file_path = '/home/vlad/audio/dataset/'
file_path_cache = '/home/vlad/audio/dataset_cache/'

for file in os.listdir(file_path):
    path, ext = os.path.splitext(file)
    assert ext == '.mp3'
    rate, data = read_mp3(file_path + file)
    mon = []
    for row in data:
        mon.append(row)
    ceps, mspec, spec = mfcc(mon)
    np.save(file_path_cache + path, ceps)
