import numpy as np
import scipy

import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
nltk.download('punkt')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm

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
        for num,sentence in enumerate(text):
            s=nltk.word_tokenize(sentence)
            count=len(s)
            l=[(num,count)]
            yield l
            
    @staticmethod
    def POS_t(text):
        for sentence in text:
           tokens = nltk.word_tokenize(sentence)            
            txt=nltk.Text(tokens)                      
            tags = nltk.pos_tag(txt)  
            yield tags
            
    @staticmethod      
    def bigrams(text):
        for i in text:
            token=nltk.word_tokenize(i)
            bigrams=ngrams(token,2)
            yield bigrams
            
    @staticmethod      
    def trigrams(text):
        for i in text:
            token=nltk.word_tokenize(i)
            trigrams=ngrams(token,3)
            yield trigrams

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
        events, labels, already_sparse_m, = [], [], []
        n_events = len(dataset)
        for c, (text, label) in tqdm(enumerate(dataset)):
            if label not in self.labels:
                self.labels[label] = self.next_label_id
                self.labels_by_id[self.next_label_id] = label
                self.next_label_id += 1
            labels.append(self.labels[label])
            events.append(set())
            
            for function_name in Featurizer.feature_functions:
                function = getattr(Featurizer, function_name)
                if type(function(text)) == scipy.sparse.csr.csr_matrix:
                    already_sparse_m.append(function(text))
                    continue
                    
                if  type(function(text)[0]) == tuple:
                    for (num,count) in function(text):
                        if num not in self.features:
                            self.features[num] = count
                        feat_id = self.features[num]
                        events[-1].add(feat_id)
           
                for feature in function(text):
                    if feature not in self.features:
                        if not allow_new_features:
                            continue
                        self.features[feature] = self.next_feature_id
                        self.features_by_id[self.next_feature_id] = feature
                        self.next_feature_id += 1
                    feat_id = self.features[feature]
                    events[-1].add(feat_id)
        
        events_sparse = self.to_sparse(events)
        labels_array = np.array(labels)
        
        if len(already_sparse_m) == 0:
            return events_sparse, labels_array
        
        else:
            for i in already_sparse_m:
                new_events=scipy.sparse.hstack((i,events_sparse))
                events_sparse=new_events
            return events_sparse, labels_array
