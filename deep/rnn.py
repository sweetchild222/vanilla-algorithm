
from rnn_lib.loader import *
from rnn_lib.model_templates import *
from model.model import *
from util.util import *
import argparse
import datetime as dt
from functools import partial
import datetime as dt
import os


def print_arg(model, activation, weight, weightRandom, gradient, epochs, data_length):

    arg = ['model', 'activation', 'weight', 'weightRandom', 'gradient', 'epochs', 'data length']
    values = [model, activation, weight, weightRandom, gradient, epochs, data_length]
    table = {'Argument':arg, 'Values':values}
    print_table(table)

def print_performance(span):

    performance = ['train minute span']

    min_span = '{:.2f}'.format(span.total_seconds() / 60)
    values = [min_span]
    table = {'Performance':performance, 'Values':values}
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



def test(train_x, train_y, oneHotmap, modelTemplate, epochs, build_hook_func, train_hook_func):

    model = Model(modelTemplate)
    model.build(build_hook_func)

    start_time = dt.datetime.now()

    #model.train(train_x, train_y, epochs, word_length, False, train_hook_func)

    train_span = (dt.datetime.now() - start_time)

    return train_span

def parse_arg():

    parser = argparse.ArgumentParser(prog='RNN')
    parser.add_argument('-m', dest='modelType', type=str, default='light', choices=['light', 'complex'], help='sample model type (default:light)')
    parser.add_argument('-g', dest='gradientType', type=str, default='rmsProp', choices=['adam', 'sgd', 'rmsProp'], help='sample gradient type (default: rmsProp)')
    parser.add_argument('-a', dest='activationType', type=str, default='elu', choices=['linear', 'relu', 'elu', 'leakyRelu', 'sigmoid', 'tanh'], help='sample activation type (default: relu)')
    parser.add_argument('-w', dest='weightType', type=str, default='he', choices=['lecun', 'glorot', 'he'], help='initial weight type (default: he)')
    parser.add_argument('-r', dest='weightRandomType', type=str, default='normal', choices=['uniform', 'normal'], help='initial weight random type (default: normal)')
    parser.add_argument('-e', dest='epochs', type=int, default=200, help='epochs (default: 200)')

    args = parser.parse_args()

    if args.epochs < 1:
        print('RNN: error: argument -e: invalid value: ', str(args.epochs), ' (value must be over 0')
        return None

    return args


def main(modelType, activationType, weightType, weightRandomType, gradientType, epochs):

    train_x, train_y, oneHotmap = loadDataSet(25)

    print_arg(modelType, activationType, weightType, weightRandomType, gradientType, epochs, train_x.shape[0])

    modelTemplate = createModelTemplate(modelType, activationType, weightType, weightRandomType, gradientType, train_x.shape[1:], len(oneHotmap))

    build_hook_func = partial(build_hook)

    train_hook_func = partial(train_hook)

    train_span = test(train_x, train_y, oneHotmap, modelTemplate, epochs, build_hook_func, train_hook_func)

    print_performance(train_span)


if __name__ == "__main__":

    args = parse_arg()

    if args != None:
        main(args.modelType, args.activationType, args.weightType, args.weightRandomType, args.gradientType, args.epochs)
