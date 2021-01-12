from PIL import Image
import numpy as np
import os


def extractData(sequence_length):

    data = open('rnn_lib/data/min.txt', 'r').read()

    remain = sequence_length - (len(data) % sequence_length)
    data = list(data) + [' '] * (remain + 1)
    data = np.array(data)
    unique = np.unique(data, return_counts=False)
    oneHotMap = {key : index for index, key in enumerate(unique)}

    return np.array([oneHotMap[char] for char in data]), oneHotMap


def encodeOneHot(data, classes):

    oneHotEncode = [np.eye(classes)[i].reshape(classes, 1) for i in data]

    return np.array(oneHotEncode)


def loadDataSet(sequence_length):

    data, oneHotmap = extractData(sequence_length)

    index = 0
    data_length = len(data)

    x_list = []
    y_list = []

    while index < data_length:
        length = sequence_length if data_length > (index + sequence_length) else data_length % sequence_length

        x = encodeOneHot(data[index:index + length], len(oneHotmap))
        y = encodeOneHot(data[index + 1:index + length + 1], len(oneHotmap))

        x_list.append(x)
        y_list.append(y)
        index += (length)
        index += 1

    train_x = np.concatenate(x_list).reshape((len(x_list), sequence_length, len(oneHotmap), 1))
    train_y = np.concatenate(y_list).reshape((len(y_list), sequence_length, len(oneHotmap), 1))    

    return train_x, train_y, oneHotmap
