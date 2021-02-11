import numpy as np
import scipy

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class Featurizer():
    @staticmethod
    def bag_of_words(text):
        for word in word_tokenize(text):
            yield word
    
    @staticmethod
    def count_vectorizer(text):
        count_v = CountVectorizer() 
        count_v.fit(texts) 
        yield count_v.transform(texts)
    
    @staticmethod
    def tfidf_vectorizer(text):
        tfidf_v = CountVectorizer() 
        tfidf_v.fit(texts) 
        yield tfidf_v.transform(texts)
        
    @staticmethod
    def number_of_words(text):
        

    feature_functions = [
        'bag_of_words']

    def __init__(self):
        self.labels = {}
        self.labels_by_id = {}
        self.features = {}
        self.features_by_id = {}
        self.next_feature_id = 0
        self.next_label_id = 0

    def to_sparse(self, events):
        """convert sets of ints to a scipy.sparse.csr_matrix"""
        data, row_ind, col_ind = [], [], []
        for event_index, event in enumerate(events):
            for feature in event:
                data.append(1)
                row_ind.append(event_index)
                col_ind.append(feature)
                
        n_features = self.next_feature_id
        n_events = len(events)
        matrix = scipy.sparse.csr_matrix(
            (data, (row_ind, col_ind)), shape=(n_events, n_features))

        return matrix

    def featurize(self, dataset, allow_new_features=False):
        events, labels = [], []
        n_events = len(dataset)
        for c, (text, label) in enumerate(dataset):
            if c % 2000 == 0:
                print("{0:.0%}...".format(c/n_events), end='')
            if label not in self.labels:
                self.labels[label] = self.next_label_id
                self.labels_by_id[self.next_label_id] = label
                self.next_label_id += 1
            labels.append(self.labels[label])
            events.append(set())
            for function_name in Featurizer.feature_functions:
                function = getattr(Featurizer, function_name)
                for feature in function(text):
                    if feature not in self.features:
                        if not allow_new_features:
                            continue
                        self.features[feature] = self.next_feature_id
                        self.features_by_id[self.next_feature_id] = feature
                        self.next_feature_id += 1
                    feat_id = self.features[feature]
                    events[-1].add(feat_id)
        
        print('done, sparsifying...', end='')
        events_sparse = self.to_sparse(events)
        labels_array = np.array(labels)
        print('done!')

        return events_sparse, labels_array
