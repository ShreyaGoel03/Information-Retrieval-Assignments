import nltk
nltk.download('stopwords')

import nltk
nltk.download('punkt')

"""IMPORT STATEMENTS

"""

import glob
import numpy as np
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import operator
stop_words = set(stopwords.words('english')) 
#extracting stop words from nltk repo

"""DATA PREPROCESSING

"""

def delete_spec_chars(input): 
    #function to delete special characters

    regex = r'[^a-zA-Z0-9\s]'
    output = re.sub(regex,'',input)
    return output

def find_unique(words): 
    #function to find unique words along with its frequency in the doc

    unique_words = list(set(words))
    word_freq = {}
    for word in unique_words:
        word_freq[word] = words.count(word)
    return word_freq

def process(path):
    #function to find unique words along with its frequency in the doc
    unique_words_dict = {}   
    file_info = {} 
    file_data=[]
    doc_id = 1
    
    for file in glob.glob(path): 
        fpath = file
        # print(file)
        fname = file.split("/")[-1]        
        fname = fname.split(".")[0]                                           
        if os.path.isdir(file):     
            # print(fname)                   
            if fname == "SRE":                
                for file1 in glob.glob(file+'/*'):                                  
                    fpath1 = file1
                    fname1 = file1.split("/")[-1]        
                    fname1 = fname1.split(".")[0]                    
                    if fname1 == "" or fname1=="index":
                        continue
                    else:
                        print(doc_id,fname1)                        
                        file = open(file1,"r",encoding='unicode_escape')        
                        doc = file.read() #reading contents of doc        
                        doc = delete_spec_chars(str(doc)) #deleting special characters
                        # doc = re.sub(r'\d+','',doc) #deleting numbers
                        tokens = word_tokenize(doc) #extracting tokens                      
                        tokens_lower = [word.lower() for word in tokens] 
                        # tokens_lematized = lematize(tokens_lower)
                        tokens_final = [word for word in tokens_lower if word not in stop_words and len(word) > 1] #Removing stopwords                                   
                        file_info[doc_id] = os.path.basename(fpath1)#Storing documents with it's id
                        file_data.append(tokens_final)#storing data of each document in list of list format 
                        doc_id += 1
                        uq_dict = find_unique(tokens_final)
                        unique_words_dict.update(uq_dict)
            else:
                continue   
        else:                                          
            if fname == "index":
                continue
            print(doc_id,fname)
            file = open(file,"r",encoding='unicode_escape')        
            doc = file.read() #reading contents of doc        
            doc = delete_spec_chars(str(doc)) #deleting special characters
            doc = re.sub(r'\d+','',doc) #deleting numbers
            tokens = word_tokenize(doc) #extracting tokens
            tokens_lower = [word.lower() for word in tokens] #Removing stopwords                                   
            # tokens_lematized = lematize(tokens_lower)
            tokens_final = [word for word in tokens_lower if word not in stop_words and len(word) > 1]
            file_info[doc_id] = os.path.basename(fpath)#Storing documents with it's id
            file_data.append(tokens_final)#storing data of each document in list of list format 
            doc_id += 1
            uq_dict = find_unique(tokens_final)
            unique_words_dict.update(uq_dict) #unique words along with number of times it appears in doc
    unique_words = list(set(unique_words_dict)) #list of unique words in all docs combined
    return unique_words_dict,file_info,file_data

    # return file_info,file_data

path="/content/drive/MyDrive/Colab Notebooks/IR-ASSIGNMENT2/stories/*"
#preprocessing stories dataset
unique_words_dict,file_info,file_data=process(path)

#printing file info
print(file_info)

#printing file data
print(file_data[1])

#total number of documents
print(len(file_data))

"""TF-IDF CALCULATION"""

def idf():
    #Function to find inverse document frequency
   
    #Firstly finding Document Frequency: Number of Documents that contain term t.

    #doc_freq : Dictionary which contain count of number of document for each term in key value format.
    doc_freq = {}

    for item in file_data:
        #traversing through file_data to get count of document for each term.   

        terms  =  list(set(item))#finding all unique ters.
      
        for word in terms:
            #storing count of each term in doc_freq dicitonary.
            if doc_freq.get(word) is not None:
                doc_freq[word] =doc_freq.get(word)+1
            else:
                doc_freq[word] = 1

    #Secondly Finding inverse document frequency: IDF(word)=log(total no. of documents/document frequency(word)+1)

    #Finding total number of documents 
    total_docs=len(file_data)

    #inverse_doc_freq: dictionary to store IDF value.
    inverse_doc_freq = {}

    for word in doc_freq.keys():
        #Calculating IDF value for each term using given formula.
        
        inverse_val=np.log((total_docs)/doc_freq[word]+1)
        inverse_doc_freq[word] = inverse_val

    return doc_freq,inverse_doc_freq

#Calculating inverse document frequency
doc_freq,inverse_doc_freq=idf()

print(doc_freq)

print(inverse_doc_freq)

def term_frequency():
    #Function to Caclulate Term Frequency: Number of particular term t in Document d.

    #list of list to store term frequency where sub list is created for each document.
    final_tf=[]

    for li in file_data:
        #Traversing through file data to get each document.

        #tf: dictionary which contain key value pair in the form of -> "term": count 
        tf={}
        for term in li:  
            #Traversing through each document to get count of terms in that document.
            if tf.get(term) is not None:
                tf[term] =tf.get(term)+1
            else:
                tf[term] = 1

        final_tf.append(tf)
    return final_tf

#Calculating term frequency
term_freq=term_frequency()
print(term_freq[1])

def tf_idf(freq_val):
    #Function to Calculate TF-IDF VALUE= tf*idf

    #list to store calculated tf*idf value for each term in each document.
    final_tf_idf=[]
    
    for term in freq_val:
        #Traversing through each term to calculate it's tf-idf value.
        tf_idf={}
        for word in term.keys():
            tf_idf[word]=term[word]*inverse_doc_freq[word]
        final_tf_idf.append(tf_idf)

    return final_tf_idf

"""Binary Term Frequency"""

#Binary term Frequency

#always be 1 for all values because no word with 0 freq is added in the list,words with list freq 1 are added in the list.
def binary_term_freq(term_freq):
     #Function to Caclulate Binary Term Frequency

     #list of list to store binary term frequency where sub list is created for each document. 
    binary_tf= []

    for term in term_freq:
         #Traversing through term_freq 
        tf = {}
    
        for word in term.keys():
            #Fingng binary term frequency.
            if term.get(word) < 0:
                tf[word] = 0
            else:
                tf[word] = 1
            
        binary_tf.append(tf)
    return binary_tf

#Calculating binary tf
binary_tf=binary_term_freq(term_freq)
print(binary_tf[1])

print("\n")

#calculating tfidf value for binary weighting scheme
binary_tfidf=tf_idf(binary_tf)
print(binary_tfidf[1])

"""RAW COUNT TERM FREQUENCY"""

#Raw count term frequency
def raw_count_tf(term_freq):
    #Function to calculate raw count term frequency.

    raw_count_tf= term_freq
    return raw_count_tf

#Calculating raw count tf
raw_count_tf=raw_count_tf(term_freq)
print(raw_count_tf[1])

print("\n")

#calculating tfidf value for RAW COUNT weighting scheme
raw_count_tfidf=tf_idf(raw_count_tf)
print(raw_count_tfidf[1])

"""Term Frequency"""

#Term Frequency

def term_Freq(term_freq):
     #Function to calculate Term Frequency.

     #list of list to store binary Term Frequency where sub list is created for each document. 
    Term_Frequency = []

    
    for term in term_freq:
        #Traversing through term_freq
        tf = {}
        values = term.values() #Return values of a dictionary.
        total = sum(values) #Compute sum of the values.
        # print(total)

        for word in term.keys():
            #Calculating Term Frequency for each word.
 
            tf[word]=term.get(word)/total

        Term_Frequency.append(tf)
    return Term_Frequency

#Calculating term frequency
Term_Frequency=term_Freq(term_freq)
print(Term_Frequency[1])

print("\n")
Term_Frequency_tfidf=tf_idf(Term_Frequency)
print(Term_Frequency_tfidf[1])

"""Log tf-idf Weighting """

#Log term frequency
def Log_term_freq(term_freq):
     #Function to calculate log normalized Term Frequency.

    #list of list to store log normalized Term Frequency where sub list is created for each document. 
    log_tf = []

    for term in term_freq:
        tf = {}

        for word in term.keys():
            #Calculating log tf value for each term

            tf[word]=np.log(1+term.get(word))

        log_tf.append(tf)
    return log_tf

#Calculating log tf
log_tf=Log_term_freq(term_freq)

print(log_tf[1])
print("\n")
log_tfidf=tf_idf(log_tf)

print(log_tfidf[1])

"""Double log normaliztion tf-idf weighting"""

#Double Log Normalization

def double_Log_term_freq(term_freq):
     #Function to calculate double log normalized Term Frequency.

    #list of list to store double log normalized Term Frequency where sub list is created for each document. 
    double_log_tf = []

    for term in term_freq:
        tf = {}
        max_tf=max(term.values())
        for word in term.keys():

            tf[word]=0.5 + 0.5*(term.get(word)/max_tf)

        double_log_tf.append(tf)
    return double_log_tf

#Calculating double log normalized term frequency
double_log_tf=double_Log_term_freq(term_freq)

print(double_log_tf[1])
print("\n")

double_log_tfidf=tf_idf(double_log_tf)
print(double_log_tfidf[1])

#Creating list of all available term in document.
total_terms=list(doc_freq.keys())
print(total_terms)
len(total_terms)

"""Calculate TOP 5 Documents"""

def calculate_top(tfidf_val,query):


    dict_tfidf_val={}

    for i in range(len(file_data)+1):
        dict_tfidf_val[i] = 0.0

    for word in query:
        #Traversing through query 
        doc_id=0
        for val in tfidf_val:
            #Traversing through each document to check if query term is present in that document, if present then storing tfidf val with document id.
            doc_id=doc_id+1
            if word in val:
                
                dict_tfidf_val[doc_id]=dict_tfidf_val[doc_id]+val[word]

    sorted_d=dict(sorted(dict_tfidf_val.items(),key=operator.itemgetter(1),reverse=True)) 
    print(sorted_d)
    count_num=0
    tfidf_file=[]

    for item in sorted_d.keys():

        if (count_num != 5):
            tfidf_file.append(item)
        else: 
            break
        count_num=count_num+1
    
    return tfidf_file

def process_query():

    query_input=input("Enter the Input text for Query ")
    query_preprocessed=delete_spec_chars(query_input)
    tokens = word_tokenize(query_preprocessed )
    query_final = [word.lower() for word in tokens if word not in stop_words and len(word) > 1] #Removing stopwords 
    print(query_final)
    
    binary=calculate_top(binary_tfidf,query_final)
    raw_count=calculate_top(raw_count_tfidf,query_final)
    termfreq=calculate_top(Term_Frequency_tfidf,query_final)
    log=calculate_top(log_tfidf,query_final)
    doublelog=calculate_top(double_log_tfidf,query_final)
    return binary,raw_count,termfreq,log,doublelog

binary,raw_count,termfreq,log,doublelog=process_query()

print("top 5 documents for binary tfidf are:")
for i in binary:
    print(file_info[i])

print("top 5 documents for raw_count tfidf are:")
for i in raw_count:
    print(file_info[i])

print("top 5 documents for termfreq tfidf are:")
for i in termfreq:
    print(file_info[i])

print("top 5 documents for log tfidf are:")
for i in log:
    print(file_info[i])

print("top 5 documents for doublelog tfidf are:")
for i in doublelog:
    print(file_info[i])
