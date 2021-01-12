from PIL import Image
import numpy as np
import random
import os
from array import *
from random import shuffle
import operator

def randomLabels(classes, trainPath):

	labels = []

	for dirname in os.listdir(trainPath):
		labels.append(dirname)

	return random.sample(labels, k=classes)


def loadMNISTFiles(path, lables):

	colorDim = 1

	X = []
	Y = []

	imgSize = 0

	for label in lables:

		subPath = path + '/' + label

		for fileName in os.listdir(subPath):

			if fileName.endswith(".png") == False:
				continue

			filePath = subPath + '/' + fileName

			img = np.array(Image.open(filePath)).astype(np.float32)

			imgSize = img.shape[0]

			list = []

			for i in range(colorDim):
				list = np.append(list, img.copy())

			X.append(list)

			Y.append(label)


	#matrixToImage(np.array(X).reshape(len(X), 1, imgSize * imgSize))

	return np.array(X).reshape(len(X), colorDim, imgSize, imgSize), np.array(Y)


def extractMNIST(classes, trainPath, testPath):

	lables = randomLabels(classes, trainPath)

	train_x, train_y = loadMNISTFiles(trainPath, lables)

	test_x, test_y = loadMNISTFiles(testPath, lables)

	return train_x, train_y, test_x, test_y


def makeOneHotMap(train_y, test_y):

    labels = np.hstack((train_y, test_y))

    unique = np.unique(labels, return_counts=False)

    return {key : index for index, key in enumerate(unique)}


def encodeOneHot(oneHotMap, train_y, test_y):

    labels = np.hstack((train_y, test_y))

    classes = len(oneHotMap)

    labels = [np.eye(classes)[oneHotMap[y]].reshape(classes, 1) for y in labels]
    labels = np.array(labels)

    train = labels[0:len(train_y)]
    test = labels[-len(test_y):]

    return train, test


def loadDataSet(classes):

    train_x, train_y, test_x, test_y = extractMNIST(classes, 'cnn_lib/mnist/train', 'cnn_lib/mnist/test')

    all_x = np.vstack((train_x, test_x))
    all_x -= np.mean(all_x)
    all_x /= np.std(all_x)

    train_x = all_x[0:len(train_x)]
    test_x = all_x[-len(test_x):]

    oneHotMap = makeOneHotMap(train_y, test_y)

    train_y, test_y = encodeOneHot(oneHotMap, train_y, test_y)

    return train_x, train_y, test_x, test_y, oneHotMap
