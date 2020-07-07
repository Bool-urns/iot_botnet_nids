import pandas as pd
import time
start = time.time()
import xgboost as xgb
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib

import pickle

def train_xgb_classifier():
  #csv = "botiot_12c_8f.csv"
  csv = "botiot_8_features.csv"
  df = pd.read_csv(csv)
  df = df.drop(['Unnamed: 0'], axis=1)

  X = df.drop(['subcategory_number'],axis=1).values

  Y = df['subcategory_number'].values

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

  xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
  xgb_model.fit(X, Y)

  xgb_acc = xgb_model.score(X_test, Y_test)

  dumped = pickle.dumps(xgb_model)
  size = str(len(dumped))
  print("model accuracy on training data: %s%%" % xgb_acc*100)
  print("model size: %sBytes" % size)
  print("training time %s seconds" % (time.time() - start))

  #saving model so that training doesn't have to be done every time
  joblib.dump(xgb_model, 'xgb_botiot_model.pkl')
  #return xgb_model

def test_xgb_classifier(df):
  test_data = df.values

  #xgb_model = train_classifier()
  xgb_botiot_model = open('xgb_botiot_model.pkl','rb')
  xgb_model = joblib.load(xgb_botiot_model)

  pred = xgb_model.predict(test_data)
  return pred

train_xgb_classifier()

