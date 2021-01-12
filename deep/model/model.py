from abc import *
import numpy as np
from model.layer.input import *
from model.layer.convolution import *
from model.layer.max_pooling import *
from model.layer.flatten import *
from model.layer.dense import *
from model.layer.basicRNN import *
import random

class Model:
    def __init__(self, layerList):
        self.layerList = layerList
        self.head = None
        self.tail = None
        self.labelIndexs = None


    def createModel(self, layerList, call_func=None):

        backward_layer = None
        head = None
        tail = None

        for layer in layerList:
            parameter = layer['parameter']
            parameter['backward_layer'] = backward_layer

            layerClass = {'input':Input, 'convolution':Convolution, 'maxPooling':MaxPooling, 'flatten':Flatten, 'dense':Dense, 'basicRNN':BasicRNN}
            type = layer['type']

            backward_layer = layerClass[type](**parameter)

            if call_func is not None:
                call_func(self, backward_layer, parameter)

            if head == None:
                head = backward_layer

        tail = backward_layer

        return head, tail


    def build(self, call_func=None):

        head, tail = self.createModel(self.layerList, call_func)

        self.head = head
        self.tail = tail

        return head, tail

    def shuffle(self, train_x, train_y):

        shuffle_indexs = np.arange(train_x.shape[0])
        np.random.shuffle(shuffle_indexs)

        return train_x[shuffle_indexs], train_y[shuffle_indexs]

    def train(self, train_x, train_y, epochs, batches, shuffle, call_func=None):

        for epoch in range(epochs):

            length = train_x.shape[0]

            if shuffle:
                train_x, train_y = self.shuffle(train_x, train_y)

            loss = 0
            i = 0

            for b in range(0, length, batches):

                batch_x = train_x[b : b + batches]
                batch_y = train_y[b : b + batches]

                loss += self.batchTrain(self.head, self.tail, batch_x, batch_y)
                i += 1

            if call_func is not None:
                call_func(self, epoch + 1, epochs, loss / i)


    def categoricalCrossEntropy(self, predict_y, y):
        return -np.sum(y * np.log2(predict_y))


    def batchTrain(self, head, tail, x, y):

        batches = len(x)

        loss = 0

        self.beginBatch(head)

        for i in range(batches):

            predict_y = self.forward(head, x[i])

            loss += self.categoricalCrossEntropy(predict_y, y[i])

            self.backward(tail, predict_y, y[i])

        self.endBatch(head)

        return loss / batches


    def beginBatch(self, head):

        forward_layer = head

        while True:
            forward_layer.beginBatch()

            next = forward_layer.forwardLayer()

            if next is None:
                return

            forward_layer = next


    def endBatch(self, head):

        forward_layer = head

        while True:
            forward_layer.endBatch()

            next = forward_layer.forwardLayer()

            if next is None:
                return

            forward_layer = next

    def forward(self, head, input):

        forward_layer = head

        while True:
            output = forward_layer.forward(input)

            next = forward_layer.forwardLayer()

            if next is None:
                break

            input = output
            forward_layer = next

        return output


    def backward(self, tail, error, y):

        backward_layer = tail

        while True:
            error = backward_layer.backward(error, y)

            next = backward_layer.backwardLayer()

            if next is None:
                break

            backward_layer = next


    def predict(self, test_x):

        prediction = []

        for x in test_x:
            predict = self.forward(self.head, x)

            prediction.append(predict)

        return prediction

    def test(self, x, y, call_func=None):

        prediction = self.predict(x)

        count = len(prediction)
        correct_count = 0

        np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})

        for i in range(count):

            p_index = np.argmax(prediction[i])
            y_index = np.argmax(y[i])

            correct_count += (1 if p_index == y_index else 0)

            if call_func is not None:
                call_func(self, x[i], prediction[i], y[i])

        accuracy = float(correct_count / count) * 100

        return accuracy


    def captureOutputs(self, test_x):

        captureList = {}

        for x in test_x:

            forward_layer = self.head

            output = x

            layerIndex = 0

            while True:

                layerName = forward_layer.__class__.__name__ + '_' + str(layerIndex)

                if layerName not in captureList:
                    captureList[layerName] = []

                captureList[layerName].append(output.reshape(1, -1))

                output = forward_layer.forward(output)

                next = forward_layer.forwardLayer()

                if next is None:
                    break

                forward_layer = next

                layerIndex += 1

        return captureList


    def captureWeightBias(self):

        captureList = []

        forward_layer = self.head

        layerIndex = 0

        while True:

            layerName = forward_layer.__class__.__name__
            find = False

            if layerName == 'Convolution' or layerName == 'Dense':
                find = True

            if find == True:
                weight = forward_layer.weight
                bias = forward_layer.bias
                captureList.append({'layer': layerName + '_' + str(layerIndex), 'weight': weight, 'bias': bias})

            next = forward_layer.forwardLayer()

            if next is None:
                break

            forward_layer = next

            layerIndex += 1

        return captureList
