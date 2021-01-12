from cnn_lib.loader import *
from cnn_lib.drawer import *
from cnn_lib.model_templates import *
from model.model import *
from util.util import *
import argparse
import datetime as dt
from functools import partial


def print_oneHotMap(oneHotMap):

    oneHotList = []
    labelList = []

    classes = len(oneHotMap)

    for mapKey in oneHotMap:
        map = np.eye(classes)[oneHotMap[mapKey]].reshape(classes, 1)
        oneHotList.append(map.reshape(-1))
        labelList.append(mapKey)

    print_table({'Label':labelList, 'OneHot':oneHotList})


def print_performance(accuracy, span):

    performance = ['accuracy', 'train minute span']

    min_span = '{:.2f}'.format(span.total_seconds() / 60)
    values = [str(accuracy) + ' %', min_span]
    table = {'Performance':performance, 'Values':values}
    print_table(table)


def print_arg(model, activation, weight, weightRandom, gradient, classes, epochs, batches, shuffle, train_dataset_count):

    reduced = batches > train_dataset_count

    batches = train_dataset_count if reduced else batches

    batch_str = str(batches) + (' (reduced)' if reduced else '')

    arg = ['classes', 'model', 'activation', 'weight', 'weightRandom', 'gradient', 'epochs', 'train dataset count', 'batches', 'shuffle']
    values = [classes, model, activation, weight, weightRandom, gradient, epochs, train_dataset_count, batch_str, shuffle]
    table = {'Argument':arg, 'Values':values}
    print_table(table)


def train_hook(model, epoch, epochs, loss):

    table = {'Epochs':[str(epoch) +'/' + str(epochs)], 'Loss':[loss]}
    print_table(table)


def build_hook(model, layer, parameter):

    layerName = layer.__class__.__name__

    if 'activation' in parameter:
        layerName += (' (' + parameter['activation']['type'] + ')')

    table = {'Layer':[layerName], 'Output Shape':[layer.outputShape()]}

    print_table(table)


def test_hook(model, x, prediction, y):

    p_index = np.argmax(prediction)
    y_index = np.argmax(y)

    correct = 'O' if p_index == y_index else 'X'
    y_label = y.reshape(-1).round(decimals=2)
    y_predict = prediction.reshape(-1).round(decimals=2)
    table = {'Predict':[y_predict], 'Label':[y_label], 'Correct':[correct]}
    print_table(table)


def test(train_x, train_y, test_x, test_y, modelTemplate, epochs, batches, shuffle, build_hook_func, train_hook_func, test_hook_func, draw):

    model = Model(modelTemplate)
    model.build(build_hook_func)

    start_time = dt.datetime.now()

    model.train(train_x, train_y, epochs, batches, shuffle, train_hook_func)

    train_span = (dt.datetime.now() - start_time)

    accuracy = model.test(test_x, test_y, test_hook_func)

    if draw:
        outputList = model.captureOutputs(test_x)
        drawOutput(outputList)
        weightBiasList = model.captureWeightBias()
        drawWeightBias(weightBiasList)

    return accuracy, train_span


def adjust_batches(batches, train_dataset_len):

    return batches if train_dataset_len > batches else train_dataset_len


def main(modelType, activationType, weightType, weightRandomType, gradientType, classes, epochs, batches, shuffle, draw):

    train_x, train_y, test_x, test_y, oneHotMap = loadDataSet(classes)

    print_arg(modelType, activationType, weightType, weightRandomType, gradientType, classes, epochs, batches, shuffle, len(train_x))

    batches = adjust_batches(batches, len(train_x))

    print_oneHotMap(oneHotMap)

    modelTemplate = createModelTemplate(modelType, activationType, weightType, weightRandomType, gradientType, train_x.shape[1:], len(oneHotMap))

    build_hook_func = partial(build_hook)

    train_hook_func = partial(train_hook)

    test_hook_func = partial(test_hook)

    accuracy, train_span = test(train_x, train_y, test_x, test_y, modelTemplate, epochs, batches, shuffle, build_hook_func, train_hook_func, test_hook_func, draw)

    print_performance(accuracy, train_span)


def parse_arg():

    parser = argparse.ArgumentParser(prog='CNN')
    parser.add_argument('-c', dest='classes', type=int, default='3', metavar="[1-10]", help='classes (default: 3)')
    parser.add_argument('-m', dest='modelType', type=str, default='light', choices=['light', 'complex'], help='sample model type (default:light)')
    parser.add_argument('-g', dest='gradientType', type=str, default='adam', choices=['adam', 'sgd', 'rmsProp'], help='sample gradient type (default: rmsProp)')
    parser.add_argument('-a', dest='activationType', type=str, default='elu', choices=['linear', 'relu', 'elu', 'leakyRelu', 'sigmoid', 'tanh'], help='sample activation type (default: relu)')
    parser.add_argument('-w', dest='weightType', type=str, default='he', choices=['lecun', 'glorot', 'he'], help='initial weight type (default: he)')
    parser.add_argument('-r', dest='weightRandomType', type=str, default='normal', choices=['uniform', 'normal'], help='initial weight random type (default: normal)')
    parser.add_argument('-e', dest='epochs', type=int, default=5, help='epochs (default: 5)')
    parser.add_argument('-b', dest='batches', type=int, help='batches (default: classes x 3)')
    parser.add_argument('--suffle-off', dest='shuffle', action='store_false', help='shuffle (default: True)')
    parser.add_argument('--d', dest='draw', action='store_true', help='draw result (default: False)')
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
        main(args.modelType, args.activationType, args.weightType, args.weightRandomType, args.gradientType, args.classes, args.epochs, args.batches, args.shuffle, args.draw)
