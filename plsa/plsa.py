import numpy as np

def EStep(R, K, pZW, pZD):

    dCount = R.shape[0];
    wCount = R.shape[1];
    kCount = K;

    pZDW = np.zeros((dCount, wCount, kCount))

    for d in range(dCount):
        for w in range(wCount):
            for k in range(kCount):
                pZDW[d, w, k] += pZW[k, w] * pZD[d, k]

            sum = pZDW[d, w,:].sum();

            if sum > 0:
                pZDW[d, w, :] /= sum

    return pZDW


def MStep(R, K, pZDW):

    dCount = R.shape[0];
    wCount = R.shape[1];
    kCount = K;

    pWZ = np.zeros((kCount, wCount))
    pZD = np.zeros((dCount, kCount))

    for d in range(dCount):
        for w in range(wCount):
            for k in range(kCount):
                value = pZDW[d, w, k] * R[d, w]
                pWZ[k, w] += value
                pZD[d, k] += value
                '''print('[d' + str(d) + ', w' + str(w) + ', z' + str(k) + ']  ' +  str(pZDW[d, w, k]) + ' * ' + str(R[d, w]))'''

    pWZ /= pWZ.sum(axis=1)[:,None]
    pZD /= pZD.sum(axis=1)[:,None]

    return pWZ, pZD


def init_pZW_pZD(R, K):

    dCount = R.shape[0];
    wCount = R.shape[1];
    kCount = K;

    pZW = np.empty([kCount, wCount])
    pZD = np.empty([dCount, kCount])

    for k in range(kCount):
        pZW[k] = np.random.dirichlet(np.ones(wCount), size=1)

    for d in range(dCount):
        pZD[d] = np.random.dirichlet(np.ones(kCount), size=1)

    return pZW, pZD


K = 2   # cluster count
S = 2   # EM step count
#R = np.array([[10, 0, 10, 0], [0, 10, 30, 10], [20, 0, 20, 0], [0, 20, 0, 20]])
R = np.array([[1, 0],[0, 1]])
pZD = np.array([[0.6, 0.4],[0.3, 0.7]])
pZW = np.array([[0.6, 0.4], [0.3, 0.7]])
#pZD = np.array([[0.5, 0.5], [0.5, 0.5]])
#pZW = np.array([[0.5, 0.5], [0.5, 0.5]])

pZW, pZD = init_pZW_pZD(R, K)


print('=========================================')
print('ZD : ', pZD)
print('ZW : ', pZW)
print('-----------------------------------------')

for s in range(S):

    pZDW = EStep(R, K, pZW, pZD)
    pZW, pZD = MStep(R, K, pZDW)

print('s : ', s)
print('ZDW : ', pZDW)
print('ZD : ', pZD)
print('ZW : ', pZW)
