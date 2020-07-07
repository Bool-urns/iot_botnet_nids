# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
from __future__ import print_function
import sys
import os
import numpy as np
import tensorflow as tf

from edgeml.trainer.protoNNTrainer import ProtoNNTrainer
from edgeml.graph.protoNN import ProtoNN
import edgeml.utils as utils
import helpermethods as helper

def to_onehot(y, numClasses, minlabel = None):
    '''
    If the y labelling does not contain the minimum label info, use min-label to
    provide this value.
    '''
    lab = y.astype('uint8')
    if minlabel is None:
        minlabel = np.min(lab)
    minlabel = int(minlabel)
    lab = np.array(lab) - minlabel
    lab_ = np.zeros((y.shape[0], numClasses))
    lab_[np.arange(y.shape[0]), lab] = 1
    return lab_

# Load data
#DATA_DIR = './usps10'
#DATA_DIR = "bot_iot_dataset/" #BotIoT dataset
DATA_DIR = "./botiot"
OUT_DIR = "./model"
#OUT_DIR  = "bot_iot_dataset/ProtoNN_output"
#DATA_DIR = "sampled_dataset/" #N_BaIoT dataset

#train, test = np.load(DATA_DIR + '/train.npy'), np.load(DATA_DIR + '/test.npy') #three class version
#train, test = np.load(DATA_DIR + '/train_all_classes.npy'), np.load(DATA_DIR + '/test_all_classes.npy')
#train, test = np.load(DATA_DIR + '//train_15w_11c.npy'), np.load(DATA_DIR + '//test_15w_11c.npy')
#train, test = np.load(DATA_DIR + '//bot_iot_train_10.npy'), np.load(DATA_DIR + '//bot_iot_test_10.npy')
train, test = np.load(DATA_DIR + '//train.npy'), np.load(DATA_DIR + '//test.npy')

x_train, y_train = train[:, 1:], train[:, 0]
x_test, y_test = test[:, 1:], test[:, 0]

numClasses = max(y_train) - min(y_train) + 1
numClasses = max(numClasses, max(y_test) - min(y_test) + 1)
numClasses = int(numClasses)

#y_train = helper.to_onehot(y_train, numClasses)
#y_test = helper.to_onehot(y_test, numClasses)

y_train = to_onehot(y_train, numClasses)
y_test = to_onehot(y_test, numClasses)

dataDimension = x_train.shape[1]
numClasses = y_train.shape[1]

def preprocessData(train, test):
    '''
    Loads data from the dataDir and does some initial preprocessing
    steps. Data is assumed to be contained in two files,
    train.npy and test.npy. Each containing a 2D numpy array of dimension
    [numberOfExamples, numberOfFeatures + 1]. The first column of each
    matrix is assumed to contain label information.
    For an N-Class problem, we assume the labels are integers from 0 through
    N-1.
    '''
    dataDimension = int(train.shape[1]) - 1
    x_train = train[:, 1:dataDimension + 1]
    y_train_ = train[:, 0]
    x_test = test[:, 1:dataDimension + 1]
    y_test_ = test[:, 0]

    numClasses = max(y_train_) - min(y_train_) + 1
    numClasses = max(numClasses, max(y_test_) - min(y_test_) + 1)
    numClasses = int(numClasses)

    # mean-var
    mean = np.mean(x_train, 0)
    std = np.std(x_train, 0)
    std[std[:] < 0.000001] = 1
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # one hot y-train
    lab = y_train_.astype('uint8')
    lab = np.array(lab) - min(lab)
    lab_ = np.zeros((x_train.shape[0], numClasses))
    lab_[np.arange(x_train.shape[0]), lab] = 1
    y_train = lab_

    # one hot y-test
    lab = y_test_.astype('uint8')
    lab = np.array(lab) - min(lab)
    lab_ = np.zeros((x_test.shape[0], numClasses))
    lab_[np.arange(x_test.shape[0]), lab] = 1
    y_test = lab_

    return dataDimension, numClasses, x_train, y_train, x_test, y_test

def getGamma(gammaInit, projectionDim, dataDim, numPrototypes, x_train):
    if gammaInit is None:
        print("Using median heuristic to estimate gamma.")
        gamma, W, B = utils.medianHeuristic(x_train, projectionDim,
                                            numPrototypes)
        print("Gamma estimate is: %f" % gamma)
        return W, B, gamma
    return None, None, gammaInit

def getModelSize(matrixList, sparcityList, expected=True, bytesPerVar=4):
    '''
    expected: Expected size according to the parameters set. The number of
        zeros could actually be more than that is required to satisfy the
        sparsity constraint.
    '''
    nnzList, sizeList, isSparseList = [], [], []
    hasSparse = False
    for i in range(len(matrixList)):
        A, s = matrixList[i], sparcityList[i]
        assert A.ndim == 2
        assert s >= 0
        assert s <= 1
        nnz, size, sparse = utils.countnnZ(A, s, bytesPerVar=bytesPerVar)
        nnzList.append(nnz)
        sizeList.append(size)
        hasSparse = (hasSparse or sparse)

    totalnnZ = np.sum(nnzList)
    totalSize = np.sum(sizeList)
    if expected:
        return totalnnZ, totalSize, hasSparse
    numNonZero = 0
    totalSize = 0
    hasSparse = False
    for i in range(len(matrixList)):
        A, s = matrixList[i], sparcityList[i]
        numNonZero_ = np.count_nonzero(A)
        numNonZero += numNonZero_
        hasSparse = (hasSparse or (s < 0.5))
        if s <= 0.5:
            totalSize += numNonZero_ * 2 * bytesPerVar
        else:
            totalSize += A.size * bytesPerVar
    return numNonZero, totalSize, hasSparse

_, _, x_train, y_train, x_test, y_test = preprocessData(train, test)

y_ = np.expand_dims(np.argmax(y_train, axis=1), axis=1)
train = np.concatenate([y_, x_train], axis=1)

y_ = np.expand_dims(np.argmax(y_test, axis=1), axis=1)
test = np.concatenate([y_, x_test], axis=1)


PROJECTION_DIM = 60
NUM_PROTOTYPES = 60
REG_W = 0.000005
REG_B = 0.0
REG_Z = 0.00005
SPAR_W = 0.8
SPAR_B = 1.0
SPAR_Z = 1.0
LEARNING_RATE = 0.05
#NUM_EPOCHS = 200
NUM_EPOCHS = 20
#BATCH_SIZE = 32
BATCH_SIZE = 100
GAMMA = 0.0015

#W, B, gamma = helper.getGamma(GAMMA, PROJECTION_DIM, dataDimension, NUM_PROTOTYPES, x_train)

W, B, gamma = getGamma(GAMMA, PROJECTION_DIM, dataDimension,
                       NUM_PROTOTYPES, x_train)

# Setup input and train protoNN
X = tf.placeholder(tf.float32, [None, dataDimension], name='X')
Y = tf.placeholder(tf.float32, [None, numClasses], name='Y')
protoNN = ProtoNN(dataDimension, PROJECTION_DIM,
                  NUM_PROTOTYPES, numClasses,
                  gamma, W=W, B=B)
trainer = ProtoNNTrainer(protoNN, REG_W, REG_B, REG_Z,
                         SPAR_W, SPAR_B, SPAR_Z,
                         LEARNING_RATE, X, Y, lossType='xentropy')
sess = tf.Session()
trainer.train(BATCH_SIZE, NUM_EPOCHS, sess, x_train, x_test, y_train, y_test,
              printStep=600, valStep=10)

acc = sess.run(protoNN.accuracy, feed_dict={X: x_test, Y: y_test})
# W, B, Z are tensorflow graph nodes
W, B, Z, _ = protoNN.getModelMatrices()
matrixList = sess.run([W, B, Z])
sparcityList = [SPAR_W, SPAR_B, SPAR_Z]
#nnz, size, sparse = helper.getModelSize(matrixList, sparcityList)
nnz, size, sparse = getModelSize(matrixList, sparcityList)
print("Final test accuracy", acc)
print("Model size constraint (Bytes): ", size)
print("Number of non-zeros: ", nnz)
#nnz, size, sparse = helper.getModelSize(matrixList, sparcityList, expected=False)
nnz, size, sparse = getModelSize(matrixList, sparcityList, expected=False)
print("Actual model size: ", size)
print("Actual non-zeros: ", nnz)

print("Saving model matrices to: ", OUT_DIR)
np.save(OUT_DIR + '/W.npy', matrixList[0])
np.save(OUT_DIR + '/B.npy', matrixList[1])
np.save(OUT_DIR + '/Z.npy', matrixList[2])
np.save(OUT_DIR + '/gamma.npy', gamma)
