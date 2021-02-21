from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

def test(path):
    print("Read data")
    dataset = read_data(path)
    
    print("Cleaning data")
    clead_data = pre_processing(dataset)
    
    print("Splitting data")
    train_data, val_data, test_data = split_data(clean_data)

    featurizer = Featurizer()
    print("Featurizer train data")
    train_features, train_labels = featurizer.featurize(train_data, allow_new_features=True)
   
    print("Featurizer validation data")
    val_features, val_labels = featurizer.featurize(val_data, allow_new_features=False)
    
    print("Logistic Regression")
    model = LogisticRegression()

    print("Fitting")
    model.fit(train_features, train_labels)

    print("Prediction")
    predictions = model.predict(val_features)

    targets = []
    for label in train_labels:
        f=featurizer.labels_by_id[label]
        if f not in targets:
            targets.append(f)

    print(classification_report(val_labels, predictions, target_names=targets))
