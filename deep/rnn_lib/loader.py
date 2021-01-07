from PIL import Image
import numpy as np
import os


def extractData():

    data = open('rnn_lib/data/data.txt', 'r').read()
    data = np.array(list(data))
    unique = np.unique(data, return_counts=False)
    oneHotMap = {key : index for index, key in enumerate(unique)}

    return np.array([oneHotMap[char] for char in data]), oneHotMap
