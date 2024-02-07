# pip install gensim==4.3.2

import re
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

embed_lookup = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary= True)

def process(text):
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r"\d+", "", text)
    word_tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]  
    features = np.zeros((len(filtered_sentence), 300))
    for j, word in enumerate(filtered_sentence):
        if word in embed_lookup.index_to_key:
            features[j] = embed_lookup[word]
        else:
            pass 
    return np.mean(features, axis=0)   

def w2v_embedding(text1, text2): 
    q1 = process(text1)
    q2 = process(text2)
    return q1-q2

x = w2v_embedding("How are you finish?", "Whats up?")
print(x.shape)
