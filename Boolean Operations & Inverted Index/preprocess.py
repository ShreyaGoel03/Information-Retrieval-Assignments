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
    output = re.sub(regex,'',input)
    return output

def find_unique(words): #function to find unique words along with its frequency in the doc
    unique_words = list(set(words))
    word_freq = {}
    for word in unique_words:
        word_freq[word] = words.count(word)
    return word_freq

def lematize(words): #lematization
    lemmatizer = WordNetLemmatizer() 
    words_lematized = [lemmatizer.lemmatize(word) for word in words]
    return words_lematized

def process(path):
    unique_words_dict = {}   
    file_info = {} 
    doc_id = 1
    for file1 in glob.glob(path): 
        fpath = file1
        fname = file1.split("/")[1]        
        fname = fname.split(".")[0]                                           
        if os.path.isdir(file1):                        
            if fname == "SRE":                
                for file1 in glob.glob(file1+'/*'):                                  
                    fpath1 = file1
                    fname1 = file1.split("/")[2]        
                    fname1 = fname1.split(".")[0]                    
                    if fname1 == "" or fname1=="index":
                        continue
                    else:
                        print(doc_id,fname1)                        
                        file1 = open(file1,"r",encoding='unicode_escape')        
                        doc = file1.read() #reading contents of doc        
                        doc = delete_spec_chars(str(doc)) #deleting special characters
                        doc = re.sub(r'\d+','',doc) #deleting numbers
                        tokens = word_tokenize(doc) #extracting tokens                      
                        tokens_lower = [word.lower() for word in tokens] #Removing stopwords                                   
                        tokens_lematized = lematize(tokens_lower)
                        tokens_final = [word for word in tokens_lematized if word not in stop_words and len(word) > 1]
                        file_info[doc_id] = os.path.basename(fpath1)
                        doc_id += 1
                        uq_dict = find_unique(tokens_final)
                        unique_words_dict.update(uq_dict)
            else:
                continue   
        else:                                          
            if fname == "index":
                continue
            print(doc_id,fname)
            file1 = open(file1,"r",encoding='unicode_escape')        
            doc = file1.read() #reading contents of doc        
            doc = delete_spec_chars(str(doc)) #deleting special characters
            doc = re.sub(r'\d+','',doc) #deleting numbers
            tokens = word_tokenize(doc) #extracting tokens
            tokens_lower = [word.lower() for word in tokens] #Removing stopwords                                   
            tokens_lematized = lematize(tokens_lower)
            tokens_final = [word for word in tokens_lematized if word not in stop_words and len(word) > 1]
            file_info[doc_id] = os.path.basename(fpath)
            doc_id += 1
            uq_dict = find_unique(tokens_final)
            unique_words_dict.update(uq_dict) #unique words along with number of times it appears in doc
    unique_words = list(set(unique_words_dict)) #list of unique words in all docs combined
    return unique_words,unique_words_dict,file_info

def process_query(query):
    tokens = word_tokenize(query)
    tokens_lower = [word.lower() for word in tokens]
    tokens_lematized = lematize(tokens_lower)
    tokens_final = [word for word in tokens_lematized if word not in stop_words and len(word) > 1]                               
    return ' '.join(tokens_final)


   