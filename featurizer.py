import numpy as np
import scipy

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from tqdm import tqdm

class Featurizer():
    @staticmethod
    def bag_of_words(text):
        for word in word_tokenize(text):
            yield word
    
       
    @staticmethod
    def number_of_words(text):
       t=text.split()
       l=len(t)
       yield l
            
    @staticmethod
    def POS_tag_noun(text):
        for word in word_tokenize(text):                              
            tags = nltk.pos_tag(word)
            noun = sum(1 for w, tag in tags if tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNP')
            yield ('NOUN', noun)
    
    @staticmethod
    def POS_tag_verb(text):
        for word in word_tokenize(text):                              
            tags = nltk.pos_tag(word)
            verb = sum(1 for w, tag in tags if tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN')
            yield ('VERB',verb)
    
    @staticmethod
    def POS_tag_adj(text):
        for word in word_tokenize(text):                              
            tags = nltk.pos_tag(word)
            adj = sum(1 for w, tag in tags if tag == 'JJ' or tag == 'JJR' or tag == 'JJS')
            yield ('ADJ',adj)
    
    @staticmethod
    def POS_tag_adv(text):
      for word in word_tokenize(text):                              
          tags = nltk.pos_tag(word)
          adv = sum(1 for w, tag in tags if tag == 'RB' or tag == 'RBR' or tag == 'RBS')
          yield ('ADV',adv)

    @staticmethod      
    def bigrams(text):
        for word in word_tokenize(text):
            bigrams=ngrams(word,2)
            yield bigrams
            
    @staticmethod      
    def trigrams(text):
        for word in word_tokenize(text):
            trigrams=ngrams(word,3)
            yield trigrams

    #feature_functions = [
      #  'bag_of_words','number_of_words','POS_tag_noun','POS_tag_verb','POS_tag_adj','POS_tag_adv','bigrams','trigrams']
    feature_functions = ['number_of_words','POS_tag_noun','POS_tag_verb','POS_tag_adj','POS_tag_adv','bigrams','trigrams']

    def __init__(self):
        self.labels = {}
        self.labels_by_id = {}
        self.features = {}
        self.next_feature_id = 0
        self.next_label_id = 0

    def to_sparse(self, events,values):
        """convert sets of ints to a scipy.sparse.csr_matrix"""
        data, row_ind, col_ind = [], [], []
        n = 0
        for event_index, event in enumerate(events):
            for feature in event:
                data.append(values[n])
                n += 1
                row_ind.append(event_index)
                col_ind.append(feature)
                
        n_features = self.next_feature_id
        n_events = len(events)
        matrix = scipy.sparse.csr_matrix(
            (data, (row_ind, col_ind)), shape=(n_events, n_features))

        return matrix

    def featurize(self, dataset, allow_new_features=False):
        events, values, labels = [], [], []
        for c, (text, label) in tqdm(enumerate(dataset)):
            if label not in self.labels:
                self.labels[label] = self.next_label_id
                self.labels_by_id[self.next_label_id] = label
                self.next_label_id += 1
            labels.append(self.labels[label])
            events.append(set())
            
            for function_name in Featurizer.feature_functions:
                function = getattr(Featurizer, function_name)
                
                if  function_name == 'number_of_words' :
                    for length in function(text):
                        f= 'A'+str(c)
                        if f not in self.features:
                          if not allow_new_features:
                                continue
                          self.features[f] = self.next_feature_id
                          self.next_feature_id += 1
                        feat_id = self.features[f]
                        events[-1].add(feat_id)
                        values.append(length)

                if  function_name in ['POS_tag_noun','POS_tag_verb','POS_tag_adj','POS_tag_adv']:
                    for (POS,s) in function(text):
                        if POS not in self.features:
                          if not allow_new_features:
                                continue
                          self.features[POS] = self.next_feature_id
                          self.next_feature_id += 1
                        feat_id = self.features[POS]
                        events[-1].add(feat_id)
                        values.append(s)

                else:
                    for feature in function(text):
                        if feature not in self.features:
                            if not allow_new_features:
                                continue
                            self.features[feature] = self.next_feature_id
                            self.next_feature_id += 1
                        feat_id = self.features[feature]
                        events[-1].add(feat_id)
                        values.append(1)
        
        events_sparse = self.to_sparse(events,values)
        labels_array = np.array(labels)

        return events_sparse, labels_array
