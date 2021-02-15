from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

def test(path):
    dataset = read_data(path)
    
    clead_data = pre_processing(dataset)
    
    train_data, val_data, test_data = split_data(clean_data)

    featurizer = Featurizer()
    train_features, train_labels = featurizer.featurize(train_data, allow_new_features=True)
   
    val_features, val_labels = featurizer.featurize(val_data, allow_new_features=False)

    model = LogisticRegression()
    model.fit(train_features, train_labels)

    predictions = model.predict(val_features)
    print(classification_report(val_labels, predictions))
