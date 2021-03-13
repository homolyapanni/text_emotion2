from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import joblib

def test_saved(path,filename):
    print("Read data")
    dataset = read_saved_file(path,filename)
    
    print("Splitting data")
    train_data, val_data  = split_data(dataset)
    
    featurizer = Featurizer()
    print("Featurizer train data")
    train_features, train_labels = featurizer.featurize(train_data, allow_new_features=True)
   
    print("Featurizer validation data")
    val_features, val_labels = featurizer.featurize(val_data, allow_new_features=False)
    
    print("Logistic Regression")
    model = LogisticRegression()

    print("Fitting")
    model.fit(train_features, train_labels)
    
    print("Save the model")
    save = 'model_' + filename + '.sav'
    joblib.dump(model, save)

    print("Prediction")
    predictions = model.predict(val_features)

    predicted_labels = [
        featurizer.labels_by_id[label] for label in predictions]

    dev_labels = [
       featurizer.labels_by_id[label] for label in val_labels]

    target=label_names(predicted_labels, dev_labels)
    
    print(classification_report(val_labels, predictions,target_names=target)) 
    
    print("Wrong classes")
    wrong_class(val_features,val_labels,predictions)
    print("End")
