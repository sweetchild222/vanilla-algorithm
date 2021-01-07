from PIL import Image
import numpy as np

def extractData():

    img = np.array(Image.open('dnn_lib/data/train.png')).astype(np.float32)

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
