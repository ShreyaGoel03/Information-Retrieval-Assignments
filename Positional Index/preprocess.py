import glob
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english')) #extracting stop words from nltk repo

def delete_spec_chars(input): #function to delete special characters
    regex = r'[^a-zA-Z0-9\s]'
    output = re.sub(regex,' ',input)    
    return output

def find_unique(words): #function to find unique words along with its frequency in the doc
    unique_words = list(set(words))
    word_freq = {}
    for word in unique_words:
        word_freq[word] = words.count(word)
    return word_freq

def process_query(query):
    query = delete_spec_chars(query)
    tokens = word_tokenize(query)
    tokens_final = [word.lower() for word in tokens if word not in stop_words and len(word) > 1] #Removing stopwords                                   
    return ' '.join(tokens_final)


   