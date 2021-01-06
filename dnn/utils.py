from PIL import Image
import numpy as np
import os


def extractData():

    img = np.array(Image.open('data/train.png')).astype(np.float32)

    (height, width) = img.shape

    x = []
    y = []

    for h in range(height):
        for w in range(width):

            value = img[h][w]

            if value == 0.0:
                continue

            feature1 = float(w)
            feature2 = float(h)

            x.append([feature1, feature2])
            y.append(value)

    return np.array(x), np.array(y), img.shape

showColumnFlag = {}

def print_table(table):

	key_list = sorted(list(table.keys()))
	tableKey = ''.join(key_list)

	showColumn = False
	if tableKey not in showColumnFlag:
	    showColumn = True
	    showColumnFlag[tableKey] = True

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


def normalize(x):

	x -= np.mean(x)
	x /= np.std(x)

	return x
