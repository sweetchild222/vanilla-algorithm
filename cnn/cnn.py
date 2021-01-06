from utils import *
from model import *
from model_templates import *
import datetime as dt
import argparse
from drawer.drawer import *

def makeOneHotMap(train_y, test_y):

    labels = np.hstack((train_y, test_y))

    unique = np.unique(labels, return_counts=False)

    return {key : index for index, key in enumerate(unique)}


def encodeOneHot(oneHotMap, train_y, test_y):

    labels = np.hstack((train_y, test_y))

    classes = len(oneHotMap)

    labels = [np.eye(classes)[oneHotMap[y]].reshape(classes, 1) for y in labels]
    labels = np.array(labels)

    train = labels[0:len(train_y)]
    test = labels[-len(test_y):]

    return train, test


def loadDataSet(classes):

    train_x, train_y, test_x, test_y = extractMNIST(classes, './mnist/train', './mnist/test')

    all_x = np.vstack((train_x, test_x))
    all_x = normalize(all_x)

    train_x = all_x[0:len(train_x)]
    test_x = all_x[-len(test_x):]

    return train_x, train_y, test_x, test_y

def print_oneHotMap(oneHotMap):

    oneHotList = []
    labelList = []

    classes = len(oneHotMap)

    for mapKey in oneHotMap:
        map = np.eye(classes)[oneHotMap[mapKey]].reshape(classes, 1)
        oneHotList.append(map.reshape(-1))
        labelList.append(mapKey)

    print_table({'Label':labelList, 'OneHot':oneHotList}, True)


def print_shapes(train_x, train_y, test_x, test_y):

    data = ['train_x', 'train_y', 'test_x', 'test_y']
    shape = [train_x.shape, train_y.shape, test_x.shape, test_y.shape]
    table = {'Data':data, 'Shape':shape}
    print_table(table, True)


def print_performance(accuracy, span):

    performance = ['accuracy', 'train minute span']

    min_span = '{:.2f}'.format(span.total_seconds() / 60)
    values = [str(accuracy) + ' %', min_span]
    table = {'Performance':performance, 'Values':values}
    print_table(table, True)


def print_arg(model, activation, weight, weightRandom, gradient, classes, epochs, batches, train_dataset_count):

    reduced = batches > train_dataset_count

    batches = train_dataset_count if reduced else batches

    batch_str = str(batches) + (' (reduced)' if reduced else '')

    arg = ['classes', 'model', 'activation', 'weight', 'weightRandom', 'gradient', 'epochs', 'train dataset count', 'batches']
    values = [classes, model, activation, weight, weightRandom, gradient, epochs, train_dataset_count, batch_str]
    table = {'Argument':arg, 'Values':values}
    print_table(table, True)


def test(train_x, train_y, test_x, test_y, modelTemplate, epochs, batches, draw):

    model = Model(modelTemplate, log='info')
    model.build()

    start_time = dt.datetime.now()
    model.train(train_x, train_y, epochs, batches)

    train_span = (dt.datetime.now() - start_time)

    accuracy = model.test(test_x, test_y)

    if draw == True:
        outputList = model.captureOutputs(test_x)
        drawOutput(outputList)
        weightBiasList = model.captureWeightBias()
        drawWeightBias(weightBiasList)

    return accuracy, train_span


def adjust_batches(batches, train_dataset_len):

    return batches if train_dataset_len > batches else train_dataset_len


def main(modelType, activationType, weightType, weightRandomType, gradientType, classes, epochs, batches, draw):

    train_x, train_y, test_x, test_y = loadDataSet(classes)

    print_arg(modelType, activationType, weightType, weightRandomType, gradientType, classes, epochs, batches, len(train_x))

    batches = adjust_batches(batches, len(train_x))

    print_shapes(train_x, train_y, test_x, test_y)

    oneHotMap = makeOneHotMap(train_y, test_y)

    print_oneHotMap(oneHotMap)

    train_y, test_y = encodeOneHot(oneHotMap, train_y, test_y)

    modelTemplate = createModelTemplate(modelType, activationType, weightType, weightRandomType, gradientType, train_x.shape[1:], train_y.shape[1])

    accuracy, train_span = test(train_x, train_y, test_x, test_y, modelTemplate, epochs, batches, draw)

    print_performance(accuracy, train_span)


def parse_arg():

    parser = argparse.ArgumentParser(prog='CNN')
    parser.add_argument('-c', dest='classes', type=int, default='3', metavar="[1-10]", help='classes (default: 3)')
    parser.add_argument('-m', dest='modelType', type=str, default='light', choices=['light', 'complex'], help='sample model type (default:light)')
    parser.add_argument('-g', dest='gradientType', type=str, default='adam', choices=['adam', 'sgd', 'rmsProp'], help='sample gradient type (default: rmsProp)')
    parser.add_argument('-a', dest='activationType', type=str, default='elu', choices=['linear', 'relu', 'elu', 'leakyRelu', 'sigmoid', 'tanh'], help='sample activation type (default: relu)')
    parser.add_argument('-w', dest='weightType', type=str, default='he', choices=['lecun', 'glorot', 'he'], help='initial weight type (default: he)')
    parser.add_argument('-r', dest='weightRandomType', type=str, default='normal', choices=['uniform', 'normal'], help='initial weight random type (default: normal)')
    parser.add_argument('-e', dest='epochs', type=int, default=60, help='epochs (default: 60)')
    parser.add_argument('-b', dest='batches', type=int, help='batches (default: classes x 3)')
    parser.add_argument('-d', dest='draw', type=bool, default=False, help='draw result (default: False)')

    args = parser.parse_args()

    if args.classes < 1 or args.classes > 10:
        print('CNN: error: argument -c: invalid value: ', str(args.classes), ' (value must be 1 from 10')
        return None

    if args.batches == None:
        args.batches = args.classes * 3

    if args.batches < 1:
        print('CNN: error: argument -b: invalid value: ', str(args.batches), ' (value must be over 0')
        return None

    return args

if __name__ == "__main__":

    args = parse_arg()

    if args != None:
        main(args.modelType, args.activationType, args.weightType, args.weightRandomType, args.gradientType, args.classes, args.epochs, args.batches, args.draw)
