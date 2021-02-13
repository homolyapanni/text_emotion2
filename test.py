from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

def test(train_set,val_set):
  train_data = read_data(train_set)
  val_data = read_data(dev_set)

  featurizer = Featurizer()
  train_features, train_labels = featurizer.featurize(
        train_data, allow_new_features=True)
   
  val_features, val_labels = featurizer.featurize(
     val_data, allow_new_features=False)

  model = LogisticRegression()
  model.fit(train_features, train_labels)

  predictions = model.predict(val_features)
  print(classification_report(val_labels, predictions))
