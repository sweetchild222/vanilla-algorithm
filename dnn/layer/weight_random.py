import numpy as np


def glorotUniform(fab_in, fab_out, size):
    limit = np.sqrt(6/(fab_in + fab_out))
    return np.random.uniform(-limit, limit, size=size)

def glorotNormal(fab_in, fab_out, size):
    stddev = np.sqrt(2/(fab_in + fab_out))
    return np.random.normal(0, stddev, size=size)

def heUniform(fab_in, fab_out, size):
    limit = np.sqrt(6/(fab_in))
    return np.random.uniform(-limit, limit, size=size)

def heNormal(fab_in, fab_out, size):
    stddev = np.sqrt(2/(fab_in))
    return np.random.normal(0, stddev, size=size)

def lecunUniform(fab_in, fab_out, size):
    limit = np.sqrt(3/(fab_in))
    return np.random.uniform(-limit, limit, size=size)

def lecunNormal(fab_in, fab_out, size):
    stddev = np.sqrt(1/(fab_in))
    return np.random.normal(0, stddev, size=size)

def createWeightRandom(weight_random, fab_in, fab_out, size):

    typeClass = {'glorot_uniform':glorotUniform, 'glorot_normal':glorotNormal, 'he_uniform':heUniform, 'he_normal':heNormal, 'lecun_uniform':lecunUniform, 'lecun_normal':lecunNormal}

    return typeClass[weight_random['type'] + '_' + weight_random['random']](fab_in, fab_out, size)
