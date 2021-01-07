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
