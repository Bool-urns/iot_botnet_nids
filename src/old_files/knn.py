import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib

def train_classifier():
  csv = "botiot_12c_8f.csv"
  df = pd.read_csv(csv)
  df = df.drop(['Unnamed: 0'], axis=1)
  x = df.drop(['subcategory_number'], axis=1).values
  y = df['subcategory_number'].values

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
  knn = KNeighborsClassifier()

  knn = knn.fit(x_train, y_train)

  y_pred = knn.predict(x_test)

  knn_acc = knn.score(x_test, y_test)
  print("model accuracy on training data:", knn_acc)

  joblib.dump(knn, 'knn_model.pkl') #saving model so that training doesn't have to be done every time
  #return knn


def test_classifier(df):
  test_data = df.values
  
  #knn_model = train_classifier()
  knn_botiot_model = open('knn_model.pkl', 'rb')
  knn_model = joblib.load(knn_botiot_model)

  pred = knn_model.predict(test_data)
  return pred
