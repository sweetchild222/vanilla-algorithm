from abc import *
import numpy as np
from utils import *
from layer.input import *
from layer.dense import *
from layer.flatten import *
import random

class Model:
    def __init__(self, layerList, log='info'):
        self.layerList = layerList
        self.head = None
        self.tail = None
        self.log = log
        self.labelIndexs = None

    def createModel(self, layerList, call_func=None):

        backward_layer = None
        head = None
        tail = None

        for layer in layerList:
            parameter = layer['parameter']
            parameter['backward_layer'] = backward_layer

            layerClass = {'input':Input, 'dense':Dense, 'flatten':Flatten}
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

    def train(self, x, y, epochs, batches, call_func=None):

        for epoch in range(epochs):

            indexs = random.sample(list(range(0, len(x))), batches)

            batch_x = x[indexs]
            batch_y = y[indexs]

            loss = self.batchTrain(self.head, self.tail, batch_x, batch_y)

            if call_func is not None:
                call_func(self, epoch + 1, loss)

    def categoricalCrossEntropy(self, predict_y, y):
        return -np.sum(y * np.log2(predict_y))

    def batchTrain(self, head, tail, x, y):

        batches = len(x)

        loss = 0

        for i in range(batches):

            predict_y = self.forward(head, x[i])

            loss += self.categoricalCrossEntropy(predict_y, y[i])

            self.backward(tail, predict_y, y[i])

        self.updateGradient(head)

        return loss / batches


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


    def updateGradient(self, head):

        forward_layer = head

        while True:
            forward_layer.updateGradient()

            next = forward_layer.forwardLayer()

            if next is None:
                break

            forward_layer = next


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
                call_func(self, prediction[i], y[i])

        accuracy = float(correct_count / count) * 100

        return accuracy
