from PIL import Image
import numpy as np
import random
import os
from array import *
from random import shuffle
import operator
import sys

def getLabels(trainPath):

	labels = []

	for dirname in os.listdir(trainPath):
		labels.append(dirname)

	return labels

def loadFiles(path, targetPath, lables):

	X = []
	T = []

	imgSize = 0

	for label in lables:

		targetFile = targetPath + '/' + label + '.png'

		if os.path.isfile(targetFile) == False:
			continue

		target = loadTargetFile(targetFile)

		subPath = path + '/' + label

		for fileName in os.listdir(subPath):

			if fileName.endswith(".png") == False:
				continue

			filePath = subPath + '/' + fileName

			img = np.array(Image.open(filePath).convert('L')).astype(np.float32)
			imgSize = img.shape[0]

			X.append(img)
			T.append(target)

	X = np.array(X).reshape(len(X), imgSize, imgSize)
	T = np.array(T).reshape(len(T), imgSize, imgSize)

	return X, T


def loadTargetFile(filePath):

	return np.array(Image.open(filePath).convert('L')).astype(np.float32)


def extractImage(trainPath, targetPath, testPath):

	lables = getLabels(trainPath)

	train_x, train_t = loadFiles(trainPath, targetPath, lables)

	test_x,  test_t = loadFiles(testPath, targetPath, lables)

	return train_x, train_t, test_x, test_t


def loadDataSet(trainPath, targetPath, testPath):

	train_x, train_t, test_x, test_t = extractImage(trainPath, targetPath, testPath)

	all_x = np.vstack((train_x, test_x))
	max = np.max(all_x)
	all_x /= max

	train_x = np.round(all_x[0:len(train_x)], decimals=1)
	test_x = np.round(all_x[-len(test_x):], decimals=1)

	all_t = np.vstack((train_t, test_t))
	all_t /= max

	train_t = np.round(all_t[0:len(train_t)], decimals=1)
	test_t = np.round(all_t[-len(test_t):], decimals=1)

	return train_x, train_t, test_x, test_t
