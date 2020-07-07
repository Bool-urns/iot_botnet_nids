import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import pandas as pd
import pickle


def train_rf_classifier():
  start = time.time()
  csv = "botiot_8_features.csv"
  df = pd.read_csv(csv)
  df = df.drop(['Unnamed: 0'], axis=1)

  X = df.drop(['subcategory_number'], axis=1).values

  Y = df['subcategory_number'].values

  kf = KFold(n_splits=10, random_state=42)

  kf.get_n_splits(X)

  for train_index, test_index in kf.split(X):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

  rf = RandomForestClassifier(n_estimators=100)

  rf.fit(x_train, y_train)

  rf_acc = rf.score(x_test, y_test)

  dumped = pickle.dumps(rf)
  size = str(len(dumped))


  print("model accuracy on training data: %s%%" % rf_acc*100)
  print("model size: %sBytes" % size)
  print("training time %s seconds" % (time.time() - start))


#def test_rf_classifier():

train_rf_classifier()
