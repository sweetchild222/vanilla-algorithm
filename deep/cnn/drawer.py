from PIL import Image, ImageDraw, ImageFont
import numpy as np
import datetime
import os

def createFolder(subPath):

	now = datetime.datetime.now()
	folder = 'cnn_output/' + now.strftime("%H%M%S") + '/' + subPath
	os.makedirs(folder, exist_ok=True)

	return folder


def drawWeightBias(weightBiasList):

	folder = createFolder('weightbias')

	for weightBias in weightBiasList:

		layer = weightBias['layer']
		weight = weightBias['weight']
		bias = weightBias['bias']

		path = folder + '/' + layer + '_weight.png'
		matrixToImage(path, weight)

		path = folder + '/' + layer + '_bias.png'
		matrixToImage(path, bias)


def drawOutput(outputList):

	folder = createFolder('output')

	for key in outputList:
		output = outputList[key]
		path = folder + '/' + key + '.png'
		matrixToImage(path, output)


def matrixToImage(path, matrix):

	#print(path)
	numpy = np.array(matrix).reshape(len(matrix), -1)
	image = numpyToImage(numpy)
	image.save(path)


def numpyToImage(matrix):

	divider = np.max(matrix) - np.min(matrix)

	step = 255 / divider if divider != 0 else 1

	(height, width) = matrix.shape

	rectSize = 40

	image = Image.new('RGB', (width * rectSize, height * rectSize))
	drawer = ImageDraw.Draw(image)

	for y in range(height):
		for x in range(width):
			point = (x * rectSize), (y * rectSize)
			row = matrix[y]
			color = int((row[x] - np.min(matrix)) * step)
			drawRect(drawer, point, rectSize, color)
			text = '{: 0.2f}'.format(round(row[x], 2))
			drawText(drawer, point, rectSize, text)

	return image


def drawRect(drawer, point, rectSize, color):

	colorHex = '#%02x%02x%02x' % (color, color, color)

	drawer.rectangle([(point[0], point[1]), (point[0] + rectSize, point[1] + rectSize)], colorHex)


def drawText(drawer, point, rectSize, text):

	fontSize = int(rectSize / 3)
	padding = int((rectSize - fontSize) / 2)
	font = ImageFont.truetype("../util/arial.ttf", fontSize)	
	colorHex = '#%02x%02x%02x' % (255, 0, 255)
	drawer.text((point[0] + padding - 8, point[1] + padding), text, font=font, fill=colorHex)
