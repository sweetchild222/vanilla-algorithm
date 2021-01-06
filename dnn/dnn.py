import argparse
from drawer.drawer import *
import datetime as dt
import os
from utils import *
from model import *
from model_templates import *
from functools import partial


def encodeOneHot(y):

    unique = np.unique(y, return_counts=False)
    oneHotMap = {key : index for index, key in enumerate(unique)}
    classes = len(oneHotMap)
    oneHotEncode = [np.eye(classes)[oneHotMap[i]].reshape(classes, 1) for i in y]

    return np.array(oneHotEncode)


def loadTestDataSet(feature_max):

    test_x = []

    height = feature_max[0]
    width = feature_max[1]

    for h in range(height):
        for w in range(width):
            feature1 = float(w)
            feature2 = float(h)
            test_x.append([feature1, feature2])

    return np.array(test_x)


def loadDataSet():

    train_x, train_y, feature_max = extractData()

    train_y = encodeOneHot(train_y)

    test_x = loadTestDataSet(feature_max)

    all_x = np.vstack((train_x, test_x))
    all_x = normalize(all_x)

    train_x = all_x[0:len(train_x)]
    test_x = all_x[-len(test_x):]

    return train_x, train_y, test_x, feature_max


def drawPredicts(predicts, feature_max):

    path = 'result/' + datetime.datetime.now().strftime("%m%d_%H%M")
    os.makedirs(path, exist_ok=True)

    showColumn = True

    for value in reversed(predicts):

        epoch = value['epoch']
        predict = value['predict']

        index = 0

        height = feature_max[0]
        width = feature_max[1]
        matrix = np.zeros((height, width))

        for h in range(height):
            for w in range(width):
                matrix[h, w] = np.argmax(predict[index])
                index += 1

        filePath = path + '/epoch_' + str(epoch) + '.png'

        print_table({'Write Path':[filePath]}, showColumn)
        showColumn = False

        matrixToImage(filePath, matrix)


def print_arg(model, activation, weight, weightRandom, gradient, epochs, batches, train_dataset_count):

    reduced = batches > train_dataset_count

    batches = train_dataset_count if reduced else batches

    batch_str = str(batches) + (' (reduced)' if reduced else '')

    arg = ['model', 'activation', 'weight', 'weightRandom', 'gradient', 'epochs', 'train dataset count', 'batches']
    values = [model, activation, weight, weightRandom, gradient, epochs, train_dataset_count, batch_str]
    table = {'Argument':arg, 'Values':values}
    print_table(table, True)

def print_performance(span):

    performance = ['train minute span']

    min_span = '{:.2f}'.format(span.total_seconds() / 60)
    values = [min_span]
    table = {'Performance':performance, 'Values':values}
    print_table(table, True)


def predict(test_x, feature_max, predicts, max_epoch, draw_epoch_term, model, epoch):

    remain = max_epoch % draw_epoch_term

    if ((epoch % draw_epoch_term) - remain) != 0:
        return

    predict = model.predict(test_x)
    predicts.append({'epoch': epoch, 'predict': predict})

def test(train_x, train_y, test_x, modelTemplate, epochs, batches, train_hook_func):

    model = Model(modelTemplate, log='info')
    model.build()

    start_time = dt.datetime.now()

    model.train(train_x, train_y, epochs, batches, train_hook_func)

    train_span = (dt.datetime.now() - start_time)

    return train_span


def adjust_batches(batches, train_dataset_len):

    return batches if train_dataset_len > batches else train_dataset_len


def parse_arg():

    parser = argparse.ArgumentParser(prog='DNN')
    parser.add_argument('-m', dest='modelType', type=str, default='light', choices=['light', 'complex'], help='sample model type (default:light)')
    parser.add_argument('-g', dest='gradientType', type=str, default='rmsProp', choices=['adam', 'sgd', 'rmsProp'], help='sample gradient type (default: rmsProp)')
    parser.add_argument('-a', dest='activationType', type=str, default='elu', choices=['linear', 'relu', 'elu', 'leakyRelu', 'sigmoid', 'tanh'], help='sample activation type (default: relu)')
    parser.add_argument('-w', dest='weightType', type=str, default='he', choices=['lecun', 'glorot', 'he'], help='initial weight type (default: he)')
    parser.add_argument('-r', dest='weightRandomType', type=str, default='normal', choices=['uniform', 'normal'], help='initial weight random type (default: normal)')
    parser.add_argument('-e', dest='epochs', type=int, default=1000, help='epochs (default: 1000)')
    parser.add_argument('-b', dest='batches', type=int, default=100, help='batches (default: 100)')
    parser.add_argument('-d', dest='draw_epoch_term', type=int, default=200, help='draw epoch term (default: 200)')

    args = parser.parse_args()

    if args.epochs < 1:
        print('DNN: error: argument -e: invalid value: ', str(args.epochs), ' (value must be over 0')
        return None

    if args.batches < 1:
        print('DNN: error: argument -b: invalid value: ', str(args.batches), ' (value must be over 0')
        return None

    return args


def main(modelType, activationType, weightType, weightRandomType, gradientType, epochs, batches, draw_epoch_term):

    train_x, train_y, test_x, feature_max = loadDataSet()

    print_arg(modelType, activationType, weightType, weightRandomType, gradientType, epochs, batches, len(train_x))

    batches = adjust_batches(batches, len(train_x))

    modelTemplate = createModelTemplate(modelType, activationType, weightType, weightRandomType, gradientType, train_x.shape[1:], train_y.shape[1])

    predicts = []

    train_hook_func = partial(predict, test_x, feature_max, predicts, epochs, draw_epoch_term)

    train_span = test(train_x, train_y, test_x, modelTemplate, epochs, batches, train_hook_func)

    print_performance(train_span)

    drawPredicts(predicts, feature_max)


if __name__ == "__main__":

    args = parse_arg()

    if args != None:
        main(args.modelType, args.activationType, args.weightType, args.weightRandomType, args.gradientType, args.epochs, args.batches, args.draw_epoch_term)
