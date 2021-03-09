import preprocessor as p
import re
import nltk

def pre_processing(dataset,N):
    data=[]
    
    for (texts,label) in tqdm(dataset):
        # remove hastag, url,emoji ect 
        texts=p.clean(texts)
        
        texts = texts.lower()

        # remove contraction
        texts = texts.replace(r"(can't|cannot)", 'can not')
        texts = texts.replace(r"n't", ' not')
        texts = texts.replace(r"i'm","i am")
        texts = texts.replace(r"im","i am")
        texts = texts.replace(r"you're","you are")
        texts = texts.replace(r"it's","it is")
        texts = texts.replace(r"she's","she is")
        texts = texts.replace(r"he's","he is")
        texts = texts.replace(r"we're","we are")
        texts = texts.replace(r"they're","they are")
        
        #remove punctuation
        texts = re.sub(r'[^\w\s]', '', texts)

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
        
    with open('pre_proc_'+str(N)+'.csv', 'w',encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        for item in data:
            csv_writer.writerow((item[0],item[1]))
        
    return data
