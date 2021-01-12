from dnn_lib.drawer import *
from dnn_lib.loader import *
from dnn_lib.model_templates import *
from model.model import *
from util.util import *
import argparse
import datetime as dt
from functools import partial



def drawPredicts(predicts, feature_max):

    path = 'dnn_output/' + datetime.datetime.now().strftime("%m%d_%H%M")

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

        fileName = 'epoch_' + str(epoch) + '.png'

        print_table({'Write Path':[path + '/' + fileName]})

        matrixToImage(path, fileName, matrix)


def print_arg(model, activation, weight, weightRandom, gradient, epochs, batches, shuffle, train_dataset_count):

    reduced = batches > train_dataset_count

    batches = train_dataset_count if reduced else batches

    batch_str = str(batches) + (' (reduced)' if reduced else '')

    arg = ['model', 'activation', 'weight', 'weightRandom', 'gradient', 'epochs', 'train dataset count', 'batches', 'shuffle']
    values = [model, activation, weight, weightRandom, gradient, epochs, train_dataset_count, batch_str, shuffle]
    table = {'Argument':arg, 'Values':values}
    print_table(table)


def print_performance(span):

    performance = ['train minute span']

    min_span = '{:.2f}'.format(span.total_seconds() / 60)
    values = [min_span]
    table = {'Performance':performance, 'Values':values}
    print_table(table)


def train_hook(test_x, feature_max, predicts, draw_epoch_term, model, epoch, epochs, loss):

    table = {'Epochs':[str(epoch) +'/' + str(epochs)], 'Loss':[loss]}
    print_table(table)

    remain = epochs % draw_epoch_term

    if ((epoch % draw_epoch_term) - remain) != 0:
        return

    predict = model.predict(test_x)
    predicts.append({'epoch': epoch, 'predict': predict})


def build_hook(model, layer, parameter):

    layerName = layer.__class__.__name__

    if 'activation' in parameter:
        layerName += (' (' + parameter['activation']['type'] + ')')

    table = {'Layer':[layerName], 'Output Shape':[layer.outputShape()]}

    print_table(table)


def test(train_x, train_y, test_x, modelTemplate, epochs, batches, shuffle, build_hook_func, train_hook_func):

    model = Model(modelTemplate)
    model.build(build_hook_func)

    start_time = dt.datetime.now()

    model.train(train_x, train_y, epochs, batches, shuffle, train_hook_func)

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
    parser.add_argument('-e', dest='epochs', type=int, default=50, help='epochs (default: 50)')
    parser.add_argument('-b', dest='batches', type=int, default=100, help='batches (default: 100)')
    parser.add_argument('-d', dest='draw_epoch_term', type=int, default=10, help='draw epoch term (default: 10)')
    parser.add_argument('--suffle-off', dest='shuffle', action='store_false', help='shuffle (default: True)')

    args = parser.parse_args()

    if args.epochs < 1:
        print('DNN: error: argument -e: invalid value: ', str(args.epochs), ' (value must be over 0')
        return None

    if args.batches < 1:
        print('DNN: error: argument -b: invalid value: ', str(args.batches), ' (value must be over 0')
        return None

    return args


def main(modelType, activationType, weightType, weightRandomType, gradientType, epochs, batches, shuffle, draw_epoch_term):

    train_x, train_y, test_x, feature_max = loadDataSet()

    print_arg(modelType, activationType, weightType, weightRandomType, gradientType, epochs, batches, shuffle, len(train_x))

    batches = adjust_batches(batches, len(train_x))

    modelTemplate = createModelTemplate(modelType, activationType, weightType, weightRandomType, gradientType, train_x.shape[1:], train_y.shape[1])

    predicts = []

    train_hook_func = partial(train_hook, test_x, feature_max, predicts, draw_epoch_term)

    build_hook_func = partial(build_hook)

    train_span = test(train_x, train_y, test_x, modelTemplate, epochs, batches, shuffle, build_hook_func, train_hook_func)

    print_performance(train_span)

    drawPredicts(predicts, feature_max)


if __name__ == "__main__":

    args = parse_arg()

    if args != None:
        main(args.modelType, args.activationType, args.weightType, args.weightRandomType, args.gradientType, args.epochs, args.batches, args.shuffle, args.draw_epoch_term)
