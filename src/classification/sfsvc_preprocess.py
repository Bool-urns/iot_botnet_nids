import pandas as pd
from sklearn.model_selection import train_test_split
import numpy
import scipy.io as sio

from sfsvc import sfsvc_classify
from sfsvc import test_classifier

def train_preprocess(df):
    #SFSVC needs training and testing input in the form samples and labels
    df = df.drop(['Unnamed: 0'], axis=1)
    train, test = train_test_split(df, test_size=0.3)

    train_labels_df = train['subcategory_number'].copy()
    train_labels = train_labels_df.values
    train_labels = train_labels.reshape(1, train_labels.shape[0])
    train_labels = train_labels.astype(numpy.double)

    train_samples_df = train.drop(['subcategory_number'], axis=1)
    train_samples = train_samples_df.values
    train_samples = numpy.swapaxes(train_samples, 0, 1)

    #added for testing purposes
    #sio.savemat('botiot_train.mat', {'Samples': train_samples, 'Labels': train_labels})

    test_labels_df = test['subcategory_number'].copy()
    test_labels = test_labels_df.values
    test_labels = test_labels.reshape(1, test_labels.shape[0])
    test_labels = test_labels.astype(numpy.double)

    test_samples_df = test.drop(['subcategory_number'], axis=1)
    test_samples = test_samples_df.values
    test_samples = numpy.swapaxes(test_samples, 0, 1)

    #added for testing purposes
    #sio.savemat('botiot_test.mat', {'Samples':test_samples, 'Labels': test_labels})

    return train_labels, train_samples, test_labels, test_samples

def test_preprocess(df):
    test_samples = df.values
    test_samples = numpy.swapaxes(test_samples, 0, 1)
    return test_samples

'''
csv = "botiot_12c_8f.csv"
df = pd.read_csv(csv)
train_labels, train_samples, test_labels, test_samples = train_preprocess(df)

sfsvc_classify(train_labels, train_samples, test_labels, test_samples)


csv = "test.csv"
df = pd.read_csv(csv)
test_samples = test_preprocess(df)
pred = test_classifier(test_samples)
print(pred)
'''
