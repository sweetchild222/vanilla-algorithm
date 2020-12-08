from PIL import Image
import numpy as np
import random
import os
from array import *
from random import shuffle
import operator
from drawer.drawer import *


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


def normalize(x):

	x -= np.mean(x)
	x /= np.std(x)

	return x


def print_table(table, showColumn):

	template = ''

	for key in table:
		template += '{' + key + ':30}'

	if showColumn == True:
		print('')
		print('='*70)
		colmun = {}

		for key in table:
			colmun[key] = key

		print(template.format(**colmun))

		print('-'*70)

	firstKey = list(table.keys())[0]
	length = len(table[firstKey])

	for i in range(length):
		dict = {}
		for key in table:
			dict[key] = str(table[key][i])

		print(template.format(**dict))
