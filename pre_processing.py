import preprocessor as p
import re
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def pre_processing(dataset):
    data=[]
    
    for (texts,label) in dataset:
        # remove hastag, url,emoji ect 
        texts=p.clean(texts)
        
        texts = texts.lower()
    
        # remove duplicate
        pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
        texts = texts.replace(str(pattern), r"\1")

        # remove contraction
        texts = texts.replace(r"(can't|cannot)", 'can not')
        texts = texts.replace(r"n't", ' not')
        texts = texts.replace(r"i'm","i am")
        texts = texts.replace(r"im","i am")
        texts = texts.replace(r"you're","you are")

        # stopwords
        stopwords = set(nltk.corpus.stopwords.words('english'))  
        stopwords.remove('not')
        stopwords.remove('nor')
        stopwords.remove('no')
        
        word_tokens =nltk.word_tokenize(texts)
    
        sentence = [w for w in word_tokens if not w in stopwords] 
        string=" "
        text=string.join(sentence)
        data.append((text,label))   
    return data
