# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import helpermethods
import tensorflow as tf
import numpy as np
import sys
import os
import datetime

#Provide the GPU number to be used
#os.environ['CUDA_VISIBLE_DEVICES'] =''

#Bonsai imports
from edgeml.trainer.bonsaiTrainer import BonsaiTrainer
from edgeml.graph.bonsai import Bonsai

# Fixing seeds for reproducibility
tf.set_random_seed(42)
np.random.seed(42)

def preProcessData(dataDir, isRegression=False):
    '''
    Function to pre-process input data
    Expects a .npy file of form [lbl feats] for each datapoint
    Outputs a train and test set datapoints appended with 1 for Bias induction
    dataDimension, numClasses are inferred directly
    '''
    
    '''
    #eleven class version
    train = np.load(dataDir + '/train_all_classes.npy')
    test = np.load(dataDir + '/test_all_classes.npy')
    '''
    '''
    train = np.load(dataDir + '/train_100w_11c.npy')
    test = np.load(dataDir + '/test_100w_11c.npy')
    '''
    '''
    train = np.load(dataDir + '/train.npy')
    test = np.load(dataDir + '/test.npy')
    '''
    train = np.load(dataDir + '/train.npy')
    test = np.load(dataDir + '/test.npy')
    
    dataDimension = int(train.shape[1]) - 1

    Xtrain = train[:, 1:dataDimension + 1]
    Ytrain_ = train[:, 0]

    Xtest = test[:, 1:dataDimension + 1]
    Ytest_ = test[:, 0]

    # Mean Var Normalisation
    mean = np.mean(Xtrain, 0)
    std = np.std(Xtrain, 0)
    std[std[:] < 0.000001] = 1
    Xtrain = (Xtrain - mean) / std
    Xtest = (Xtest - mean) / std
    # End Mean Var normalisation

    # Classification.
    if (isRegression == False):
        numClasses = max(Ytrain_) - min(Ytrain_) + 1
        numClasses = int(max(numClasses, max(Ytest_) - min(Ytest_) + 1))

        lab = Ytrain_.astype('uint8')
        lab = np.array(lab) - min(lab)

        lab_ = np.zeros((Xtrain.shape[0], numClasses))
        lab_[np.arange(Xtrain.shape[0]), lab] = 1
        if (numClasses == 2):
            Ytrain = np.reshape(lab, [-1, 1])
        else:
            Ytrain = lab_

        lab = Ytest_.astype('uint8')
        lab = np.array(lab) - min(lab)

        lab_ = np.zeros((Xtest.shape[0], numClasses))
        lab_[np.arange(Xtest.shape[0]), lab] = 1
        if (numClasses == 2):
            Ytest = np.reshape(lab, [-1, 1])
        else:
            Ytest = lab_

    elif (isRegression == True):
        # The number of classes is always 1, for regression.
        numClasses = 1
        Ytrain = Ytrain_
        Ytest = Ytest_

    trainBias = np.ones([Xtrain.shape[0], 1])
    Xtrain = np.append(Xtrain, trainBias, axis=1)
    testBias = np.ones([Xtest.shape[0], 1])
    Xtest = np.append(Xtest, testBias, axis=1)

    mean = np.append(mean, np.array([0]))
    std = np.append(std, np.array([1]))

    if (isRegression == False):
        return dataDimension + 1, numClasses, Xtrain, Ytrain, Xtest, Ytest, mean, std
    elif (isRegression == True):
        return dataDimension + 1, numClasses, Xtrain, Ytrain.reshape((-1, 1)), Xtest, Ytest.reshape((-1, 1)), mean, std

#Loading and Pre-processing dataset for Bonsai

#dataDir = "sampled_dataset/"
dataDir = "botiot/"
(dataDimension, numClasses, Xtrain, Ytrain, Xtest, Ytest, mean, std) = preProcessData(dataDir, isRegression=False)
print("Feature Dimension: ", dataDimension)
print("Num classes: ", numClasses)

sigma = 1.0 #Sigmoid parameter for tanh
depth = 3 #Depth of Bonsai Tree
projectionDimension = 28 #Lower Dimensional space for Bonsai to work on

#Regularizers for Bonsai Parameters
regZ = 0.0001
regW = 0.001
regV = 0.001
regT = 0.001

#totalEpochs = 100
totalEpochs = 20

learningRate = 0.01

outFile = None

#Sparsity for Bonsai Parameters. x => 100*x % are non-zeros
sparZ = 0.2
sparW = 0.3
sparV = 0.3
sparT = 0.62

batchSize = np.maximum(100, int(np.ceil(np.sqrt(Ytrain.shape[0]))))

useMCHLoss = True #only for Multiclass cases True: Multiclass-Hing Loss, False: Cross Entropy. 

#Bonsai uses one classier for Binary, thus this condition
if numClasses == 2:
    numClasses = 1

X = tf.placeholder("float32", [None, dataDimension])
Y = tf.placeholder("float32", [None, numClasses])

def createTimeStampDir(dataDir):
    '''
    Creates a Directory with timestamp as it's name
    '''
    if os.path.isdir(dataDir + '/TFBonsaiResults') is False:
        try:
            os.mkdir(dataDir + '/TFBonsaiResults')
        except OSError:
            print("Creation of the directory %s failed" %
                  dataDir + '/TFBonsaiResults')

    currDir = 'TFBonsaiResults/' + datetime.datetime.now().strftime("%H_%M_%S_%d_%m_%y")
    if os.path.isdir(dataDir + '/' + currDir) is False:
        try:
            os.mkdir(dataDir + '/' + currDir)
        except OSError:
            print("Creation of the directory %s failed" %
                  dataDir + '/' + currDir)
        else:
            return (dataDir + '/' + currDir)
    return None


def dumpCommand(list, currDir):
    '''
    Dumps the current command to a file for further use
    '''
    commandFile = open(currDir + '/command.txt', 'w')
    command = "python"

    command = command + " " + ' '.join(list)
    commandFile.write(command)

    commandFile.flush()
    commandFile.close()

currDir = createTimeStampDir(dataDir)
dumpCommand(sys.argv, currDir)

bonsaiObj = Bonsai(numClasses, dataDimension, projectionDimension, depth, sigma)

bonsaiTrainer = BonsaiTrainer(bonsaiObj, regW, regT, regV, regZ, sparW, sparT, sparV, sparZ,
                              learningRate, X, Y, useMCHLoss, outFile)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

bonsaiTrainer.train(batchSize, totalEpochs, sess,
                    Xtrain, Xtest, Ytrain, Ytest, dataDir, currDir)
