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

def calculate_top(cosine_sim):
    #Function to find top5 documents on the basis of cosine similarity: according to higher cosine similarity.

    #Sorting obatined similairty values of term in decreasing order.
    sorted_d=dict(sorted(cosine_sim.items(),key=operator.itemgetter(1),reverse=True))
    print(sorted_d)

    count_num=0
    cosine_file=[]

    for item in sorted_d.keys():
        #finding document id of top 5 documents and storing them in cosine_file
        if (count_num != 5):
           cosine_file.append(item)
        else: 
            break
        count_num=count_num+1
    
    return cosine_file

"""QUERY INPUT AND PREPROCESSING"""

def process_query():
    #Function for Query input and preprocessing

    #Taking input
    query_input=input("Enter the Input text for Query ")
    #Deleting special characters
    query_preprocessed=delete_spec_chars(query_input)
    #extracting tokens
    tokens = word_tokenize(query_preprocessed )
    #Removing stopwords 
    query_final = [word.lower() for word in tokens if word not in stop_words and len(word) > 1] 
    # print(query_final)

    return query_final

#preprocessing query
query_final=process_query()
print(query_final)



"""Query Normalization"""

def query_normalize(tfidf_query):
    # Function to find normalized query vector

    #-------------------query vector-------------------------#

    #query_tf_vector: list to store all tfidf value in form of vector representation.
    query_tf_vector=[]
    for term in total_terms:
        if term not in tfidf_query  :
            query_tf_vector.append(0.0)     
        else:
            query_tf_vector.append(tfidf_query [term])

    #--------------------Normalize Query Vector-----------------------#

    #normalizing query vector
    #l2 normalization
    query_vector_normalize = []

    norm = 0.0
    for term in query_tf_vector:
        norm  = norm+pow(term,2)
    
    # norm  = 1/np.sqrt(norm )
    #normalizign vector to calcullate cosine similairty
    for term in query_tf_vector:
    #    query_vector_normalize.append(term* val )
     query_vector_normalize.append(term/np.sqrt(norm )  )

    return query_vector_normalize

"""Document Normalization"""

def doc_normalize(doc_vector):
     #------------------bianry doc normalized vector---------------------#

    #list with sub list contain normalized vector of document.
    doc__norm_vector = []

    for value in doc_vector:
        norm = 0.0
        for count in value:
            norm += pow(count,2)
        # norm = 1 / np.sqrt(norm)
    
        sub_list = []
    
        for count in value:
            # k = freq * norm
            sub_list.append(count/np.sqrt(norm))
        
        doc__norm_vector.append(sub_list)
    
    return doc__norm_vector

"""COSINE SIMILARITY USING BINARY WEIGHTING SCHEME"""

def binary_query_tfidf(query):
    # Function to find tfidf of query using binary weighting scheme

    #------------------term frequency----------------------------#

    #query_tf:Dictionary to store term frequency i.e. count of term for query
    query_tf = {}
    for term in query:
        #Traversing through query and calculating count of each term and storing it in dictionary in key:value format.
        if term not in query_tf:
            query_tf[term] = 1     
        else:
            query_tf[term] = query_tf[term]+1


    #--------------binary query term frequency------------------------#

    #query_binary_tf:Dictionary to store binary term frequency i.e. if available then 1 otherwise 0.
    #But binary term frequency will always 1 for all the values.
    query_binary_tf= {}
    for term in query_tf.keys():
        #Calculating binary term frequency in query.
        if query_tf[term] < 0:
            query_binary_tf[term] = 0
        else:
            query_binary_tf[term] = 1


    #-------------------tfidf value-------------------------------------#

    #Calculating tfidf value for each term of query.
    #tfidf_query: dictionary to store tf-idf value for each term of query.
    tfidf_query = {}

    for item in query_binary_tf.keys():
        #Calculating tfidf value: tf*idf
        if item not in inverse_doc_freq  :
            tfidf_query[item] = 0.0   
        else:
            tfidf_query[item] = query_binary_tf[item]*inverse_doc_freq[item]

    return tfidf_query

def binary_doc_vector():
    #Function to calculate normalized vector for binary_tfidf

    #--------------Binary doc vector----------------------#
    #Creating list to convert binary_tfidf into vector which contain tfidf value of term..
    binary_doc_vector = []

    for term in binary_tfidf:
        #traversing through binary_tfidf for each document
        sub_list= []
        for val in total_terms:
            #to create vector takeing each word and if it is present in document then adding tfidf value of that otherwise 0 
            if val not in term :
                 sub_list.append(0.0)            
            else:          
                sub_list.append(term[val])
        binary_doc_vector += [sub_list]

    return binary_doc_vector

def process_binary(query):
    #Function to find Cosine similarity between query and document.
    tfidf_query =binary_query_tfidf(query)
    query_vector_normalize= query_normalize(tfidf_query)
    doc_vector=binary_doc_vector()
    binary_doc__norm_vector=doc_normalize(doc_vector)

    #cosine_value: dictionary to store all the cosine values with respect to document id for query
    cosine_value = {}

    doc_id = 0
    for item in binary_doc__norm_vector:
        #calculating cosine similarity
        doc_id += 1
        document = np.array(item)
        query= np.array(query_vector_normalize)
        cosine_value [doc_id]= sum(list(document * query))
    
    final_result=calculate_top( cosine_value )

    return final_result

cosine_binary=process_binary(query_final)
print("top 5 documents for binary cosine are:")
for i in cosine_binary:
    print(file_info[i])

"""COSINE SIMILARITY USING RAW COUNT WEIGHTING SCHEME"""

def rawcount_query_tfidf(query):
    # Function to find tfidf of query using rawcount weighting scheme

    #------------------term frequency----------------------------#

    #query_tf:Dictionary to store term frequency i.e. count of term for query
    query_tf = {}
    for term in query:
        #Traversing through query and calculating count of each term and storing it in dictionary in key:value format.
        if term not in query_tf:
            query_tf[term] = 1     
        else:
            query_tf[term] = query_tf[term]+1


    #-------------rawcount query term frequency------------------------#

    #query_rawcount_tf:Dictionary to store rawcount term frequency i.e. if available then 1 otherwise 0.
   
    query_rawcount_tf=query_tf
 
    #-------------------tfidf value-------------------------------------#

    #Calculating tfidf value for each term of query.
    #tfidf_query: dictionary to store tf-idf value for each term of query.
    tfidf_query = {}

    for item in query_rawcount_tf.keys():
        #Calculating tfidf value: tf*idf
        if item not in inverse_doc_freq  :
            tfidf_query[item] = 0.0   
        else:
            tfidf_query[item] = query_rawcount_tf[item]*inverse_doc_freq[item]

    return tfidf_query

def rawcount_doc_vector():
    #Function to calculate normalized vector for rawcount_tfidf

    #--------------rawcount doc vector----------------------#
    #Creating list to convert rawcount_tfidf into vector which contain tfidf value of term..
    doc_vector = []

    for term in raw_count_tfidf:
        #traversing through rawcount_tfidf for each document
        sub_list= []
        for val in total_terms:
            
            if val not in term :
                 sub_list.append(0.0)            
            else:          
                sub_list.append(term[val])
        doc_vector += [sub_list]

    
    return doc_vector

def process_raw_count(query):
    #Function to find Cosine similarity between query and document.
    tfidf_query =rawcount_query_tfidf(query)
    query_vector_normalize= query_normalize(tfidf_query)
    doc_vector=rawcount_doc_vector()
    rawcount_doc__norm_vector=doc_normalize(doc_vector)

    #cosine_value: dictionary to store all the cosine values with respect to document id for query
    cosine_value = {}

    doc_id = 0
    for item in rawcount_doc__norm_vector:
        #calculating cosine similarity
        doc_id += 1
        document = np.array(item)
        query= np.array(query_vector_normalize)
        cosine_value [doc_id]= sum(list(document * query))
    
    final_result=calculate_top( cosine_value )

    return final_result

cosine_raw_count=process_raw_count(query_final)

print("top 5 documents for cosine_raw_count cosine are:")
for i in cosine_raw_count:
    print(file_info[i])

"""COSINE SIMILARITY USING Term Frequency WEIGHTING SCHEME"""

def TermFrequency_query_tfidf(query):
    # Function to find tfidf of query using Term Frequency weighting scheme

    #------------------term frequency----------------------------#

    #query_tf:Dictionary to store term frequency i.e. count of term for query
    query_tf = {}
    for term in query:
        #Traversing through query and calculating count of each term and storing it in dictionary in key:value format.
        if term not in query_tf:
            query_tf[term] = 1     
        else:
            query_tf[term] = query_tf[term]+1


    #-------------rawcount query term frequency------------------------#

    #query_Term_Freq_tf:Dictionary to store  term frequency 
   
    query_Term_Freq_tf= {}
   
    values = query_tf.values() #Return values of a dictionary.
    total = sum(values) #Compute sum of the values.
    for term in query_tf.keys():
         query_Term_Freq_tf[term]=query_tf[term]/total
 
    #-------------------tfidf value-------------------------------------#

    #Calculating tfidf value for each term of query.
    #tfidf_query: dictionary to store tf-idf value for each term of query.
    tfidf_query = {}

    for item in query_Term_Freq_tf.keys():
        #Calculating tfidf value: tf*idf
        if item not in inverse_doc_freq  :
            tfidf_query[item] = 0.0   
        else:
            tfidf_query[item] = query_Term_Freq_tf[item]*inverse_doc_freq[item]

    return tfidf_query

def TermFrequency_doc_vector():
    #Function to calculate normalized vector for rawcount_tfidf

    #--------------rawcount doc vector----------------------#
    #Creating list to convert rawcount_tfidf into vector which contain tfidf value of term..
    doc_vector = []

    for term in Term_Frequency_tfidf:
        #traversing through rawcount_tfidf for each document
        sub_list= []
        for val in total_terms:
            
            if val not in term :
                 sub_list.append(0.0)            
            else:          
                sub_list.append(term[val])
        doc_vector += [sub_list]

    
    return doc_vector

def process_termfreq(query):
    #Function to find Cosine similarity between query and document.
    tfidf_query =TermFrequency_query_tfidf(query)
    query_vector_normalize= query_normalize(tfidf_query)
    doc_vector=TermFrequency_doc_vector()
    TermFrequency_doc__norm_vector=doc_normalize(doc_vector)

    #cosine_value: dictionary to store all the cosine values with respect to document id for query
    cosine_value = {}

    doc_id = 0
    for item in TermFrequency_doc__norm_vector:
        #calculating cosine similarity
        doc_id += 1
        document = np.array(item)
        query= np.array(query_vector_normalize)
        cosine_value [doc_id]= sum(list(document * query))
    
    final_result=calculate_top( cosine_value )

    return final_result

cosine_termfreq=process_termfreq(query_final)

print("top 5 documents for cosine_termfreq are:")
for i in cosine_termfreq:
    print(file_info[i])

"""COSINE SIMILARITY USING Log Normalization WEIGHTING SCHEME"""

def log_query_tfidf(query):
    # Function to find tfidf of query using log weighting scheme

    #------------------term frequency----------------------------#

    #query_tf:Dictionary to store log term frequency i.e. count of term for query
    query_tf = {}
    for term in query:
        #Traversing through query and calculating count of each term and storing it in dictionary in key:value format.
        if term not in query_tf:
            query_tf[term] = 1     
        else:
            query_tf[term] = query_tf[term]+1


    #-------------log query term frequency------------------------#

    #query_Term_Freq_tf:Dictionary to store  term frequency 
   
    query_log_tf= {}
   
    for term in query_tf.keys():
         query_log_tf[term]=np.log(1+query_tf[term])
 
    #-------------------tfidf value-------------------------------------#

    #Calculating tfidf value for each term of query.
    #tfidf_query: dictionary to store tf-idf value for each term of query.
    tfidf_query = {}

    for item in query_log_tf.keys():
        #Calculating tfidf value: tf*idf
        if item not in inverse_doc_freq  :
            tfidf_query[item] = 0.0   
        else:
            tfidf_query[item] = query_log_tf[item]*inverse_doc_freq[item]

    return tfidf_query

def log_doc_vector():
    #Function to calculate normalized vector for rawcount_tfidf

    #--------------rawcount doc vector----------------------#
    #Creating list to convert rawcount_tfidf into vector which contain tfidf value of term..
    doc_vector = []

    for term in log_tfidf:
        #traversing through rawcount_tfidf for each document
        sub_list= []
        for val in total_terms:
            
            if val not in term :
                 sub_list.append(0.0)            
            else:          
                sub_list.append(term[val])
        doc_vector += [sub_list]

    
    return doc_vector

def process_log(query):
     #Function to find Cosine similarity between query and document.
    tfidf_query =log_query_tfidf(query)
    query_vector_normalize= query_normalize(tfidf_query)
    doc_vector=log_doc_vector()
    log_doc__norm_vector=doc_normalize(doc_vector)

    #cosine_value: dictionary to store all the cosine values with respect to document id for query
    cosine_value = {}

    doc_id = 0
    for item in log_doc__norm_vector:
        #calculating cosine similarity
        doc_id += 1
        document = np.array(item)
        query= np.array(query_vector_normalize)
        cosine_value [doc_id]= sum(list(document * query))
    
    final_result=calculate_top( cosine_value )

    return final_result


cosine_log=process_log(query_final)

print("top 5 documents for cosine_log are:")
for i in cosine_log:
    print(file_info[i])

"""COSINE SIMILARITY USING DOUBLE LOG NORMALIZATION WEIGHTING SCHEME"""

def doublelog_query_tfidf(query):
    # Function to find tfidf of query using doublelog weighting scheme

    #------------------term frequency----------------------------#

    #query_tf:Dictionary to store doublelog term frequency i.e. count of term for query
    query_tf = {}
    for term in query:
        #Traversing through query and calculating count of each term and storing it in dictionary in key:value format.
        if term not in query_tf:
            query_tf[term] = 1     
        else:
            query_tf[term] = query_tf[term]+1


    #-------------doublelog query term frequency------------------------#

    #query_Term_Freq_tf:Dictionary to store  term frequency 
   
    query_doublelog_tf= {}

    
    values = query_tf.values() #Return values of a dictionary.
    total_max = max(values) #Compute max of the values.
    for term in query_tf.keys():
         query_doublelog_tf[term]=0.5 + 0.5*(query_tf[term]/total_max)
 
    #-------------------tfidf value-------------------------------------#

    #Calculating tfidf value for each term of query.
    #tfidf_query: dictionary to store tf-idf value for each term of query.
    tfidf_query = {}

    for item in query_doublelog_tf.keys():
        #Calculating tfidf value: tf*idf
        if item not in inverse_doc_freq  :
            tfidf_query[item] = 0.0   
        else:
            tfidf_query[item] = query_doublelog_tf[item]*inverse_doc_freq[item]

    return tfidf_query

def doublelog_doc_vector():
    #Function to calculate normalized vector for rawcount_tfidf

    #--------------rawcount doc vector----------------------#
    #Creating list to convert rawcount_tfidf into vector which contain tfidf value of term..
    doc_vector = []

    for term in double_log_tfidf:
        #traversing through rawcount_tfidf for each document
        sub_list= []
        for val in total_terms:
            
            if val not in term :
                 sub_list.append(0.0)            
            else:          
                sub_list.append(term[val])
        doc_vector += [sub_list]

    
    return doc_vector

def process_doublelog(query):
 #Function to find Cosine similarity between query and document.
    tfidf_query =doublelog_query_tfidf(query)
    query_vector_normalize= query_normalize(tfidf_query)
    doc_vector=doublelog_doc_vector()
    doublelog__doc__norm_vector=doc_normalize(doc_vector)

    #cosine_value: dictionary to store all the cosine values with respect to document id for query
    cosine_value = {}

    doc_id = 0
    for item in doublelog__doc__norm_vector:
        #calculating cosine similarity
        doc_id += 1
        document = np.array(item)
        query= np.array(query_vector_normalize)
        cosine_value [doc_id]= sum(list(document * query))
    
    final_result=calculate_top( cosine_value )

    return final_result

cosine_doublelog=process_doublelog(query_final)

print("top 5 documents for cosine_doublelog are:")
for i in cosine_doublelog:
    print(file_info[i])

