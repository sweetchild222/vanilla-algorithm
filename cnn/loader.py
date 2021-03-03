from PIL import Image
import numpy as np
import random
import os
from array import *
from random import shuffle
import operator

def getLabels(trainPath):

	labels = []

	for dirname in os.listdir(trainPath):
		labels.append(dirname)

	return labels

def loadMNISTFiles(path, lables):

	X = []

	imgSize = 0

	for label in lables:

		subPath = path + '/' + label

		for fileName in os.listdir(subPath):

			if fileName.endswith(".png") == False:
				continue

			filePath = subPath + '/' + fileName

			img = np.array(Image.open(filePath)).astype(np.float32)
			imgSize = img.shape[0]

			X.append(img)

	#matrixToImage(np.array(X).reshape(len(X), 1, imgSize * imgSize))

	return np.array(X).reshape(len(X), imgSize, imgSize)


def extractMNIST(trainPath, testPath):

	lables = getLabels(trainPath)

	train_x = loadMNISTFiles(trainPath, lables)

	test_x = loadMNISTFiles(testPath, lables)

	return train_x, test_x


def loadDataSet():

	train_x, test_x = extractMNIST('image/train', 'image/test')

	all_x = np.vstack((train_x, test_x))
	all_x /= np.max(all_x)

	train_x = all_x[0:len(train_x)]
	test_x = all_x[-len(test_x):]

	all_t = np.where(all_x >= 0.5, 1.0, all_x)
	all_t = np.where(all_t < 0.5, 0.0, all_t)

	train_t = all_t[0:len(train_x)]
	test_t = all_t[-len(test_x):]

	return train_x, train_t, test_x, test_t
