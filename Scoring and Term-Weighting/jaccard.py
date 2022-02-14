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
            print(fname)                   
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

from google.colab import drive
drive.mount('/content/drive')

def union(query,doc):
    #Calculating Union
    result = list(set(query) | set(doc))
    return result

def intersection(query,doc):
    #Calculating intersection
    result = list(set(query) & set(doc))
    return result

def calculate_jaccardCoef(query):
    #Function to Calculate Jaccard Coeffecient
    doc_id=1
    #Dictionary to store jaccard coeff with respective doc-id in key:value format.
    dict_jaccard_val={}
    for li in file_data:
        # print(val)
        union_result=union(query,li)
        intersection_result=intersection(query,li)
        print(len(union_result))
        print(len(intersection_result))
        #Finding jaccard val for query with each document
        dict_jaccard_val[doc_id] = len(intersection_result)/len(union_result)
        
        doc_id = doc_id + 1
      
    return dict_jaccard_val

def calculate_top(jaccard_list):
    #Function to calculate top 5 documents according to the value of jaccard coeffecient.
    count=0
    jaccard_file=[]

    #Sorting the value of jaccard coeff in decreasing order
    sorted_d = dict( sorted(jaccard_list.items(), key=operator.itemgetter(1),reverse=True))
    print(sorted_d)
    for item in sorted_d.keys():
        #Finding top5 documents

        if (count != 5):
            jaccard_file.append(item)
        else: 
            break
        count=count+1
    
    return jaccard_file

def process_query():
    #Function for Query preprocessing and calculation of jaccard coeff.

    query_input=input("Enter the Input text for Query ")
    query_preprocessed=delete_spec_chars(query_input)
    tokens = word_tokenize(query_preprocessed )
    query_final = [word.lower() for word in tokens if word not in stop_words and len(word) > 1] #Removing stopwords 
    print(query_final)
    jaccard_val=calculate_jaccardCoef(query_final)
    print(jaccard_val)
    top5=calculate_top(jaccard_val)
    print(top5)
    return top5

final_result=process_query()

print("top 5 documents are:")
for i in final_result:
    print(file_info[i])



