from preprocess import *
import pickle

#Inverted Index Node
class node:
    def __init__(self,doc_id,freq=0):
        self.doc_id = doc_id
        self.freq = freq
        self.next = None

#Inverted Index List
class linked_list:
    def __init__(self, head=None, tail=None):
        self.head = head
        self.tail = tail
        self.len = 0

    def append(self,id,freq):
        new_node = node(id,freq)
        if self.head is None:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.len += 1

    def display(self,file_info):
        temp = self.head
        while(temp):
            print("Doc Id ", temp.doc_id, "Doc Name ", file_info[temp.doc_id])
            temp = temp.next



def buildIndex(path,inverted_index, universal_list):
    doc_id = 1        
    for file in glob.glob(path): 
        fpath = file
        fname = file.split("/")[1]        
        fname = fname.split(".")[0]                                           
        if os.path.isdir(file):                        
            if fname == "SRE":                
                for file1 in glob.glob(file+'/*'):                                                      
                    fname1 = file1.split("/")[2]        
                    fname1 = fname1.split(".")[0]                    
                    if fname1 == "" or fname1=="index":
                        continue
                    else:
                        universal_list.append(doc_id,0)
                        print(doc_id,fname1)                        
                        file = open(file1,"r",encoding='unicode_escape')        
                        doc = file.read() #reading contents of doc        
                        doc = delete_spec_chars(str(doc)) #deleting special characters
                        doc = re.sub(r'\d+','',doc) #deleting numbers
                        tokens = word_tokenize(doc) #extracting tokens
                        tokens_lower = [word.lower() for word in tokens] #Removing stopwords                                   
                        tokens_lematized = lematize(tokens_lower)
                        tokens_final = [word for word in tokens_lematized if word not in stop_words and len(word) > 1]              
                        uq_dict = find_unique(tokens_final)
                        for word in uq_dict.keys(): #Adding the document details for word as node in Inverted Index List
                            inverted_index[word].append(doc_id,uq_dict[word]) 
                        doc_id += 1                        
            else:
                continue   
        else:                                          
            if fname == "index":
                continue
            universal_list.append(doc_id,0)
            print(doc_id,fname)
            file = open(file,"r",encoding='unicode_escape')        
            doc = file.read() #reading contents of doc        
            doc = delete_spec_chars(str(doc)) #deleting special characters
            doc = re.sub(r'\d+','',doc) #deleting numbers
            tokens = word_tokenize(doc) #extracting tokens
            tokens_lower = [word.lower() for word in tokens] #Removing stopwords                                   
            tokens_lematized = lematize(tokens_lower)
            tokens_final = [word for word in tokens_lematized if word not in stop_words and len(word) > 1]                       
            uq_dict = find_unique(tokens_final)
            for word in uq_dict.keys():
                inverted_index[word].append(doc_id,uq_dict[word]) #Adding the document details for word as node in Inverted Index List
            doc_id += 1
    return inverted_index, universal_list

if __name__ == "__main__":
    universal_list = linked_list()    
    unique_words,unique_words_dict,file_info = process('stories/*') 
    f = open('tokens.pkl','wb')
    pickle.dump(unique_words,f)
    f.close()
    f1 = open('token_freq.pkl','wb')
    pickle.dump(unique_words_dict,f1)
    f1.close()   
    f2 = open('file_info.pkl','wb')
    pickle.dump(file_info,f2)
    f2.close()      
    # file1 = open('tokens.pkl','rb')
    # tokens = pickle.load(file1)
    # file2 = open('token_freq.pkl','rb')
    # tokens_freq = pickle.load(file2)        
    # file3 = open('file_info.pkl','rb')
    # file_info = pickle.load(file3)       
    # inverted_index = {}     
    # for word in tokens:
    #     inverted_index[word] = linked_list()  
    # inverted_index, universal_list = buildIndex('stories/*',inverted_index,universal_list)
    # f = open('inverted_index.pkl','wb')
    # pickle.dump(inverted_index,f)
    # f.close()    
    # f3 = open('overall_postinglist.pkl','wb')     
    # pickle.dump(universal_list,f3)
    # f3.close()  
    # file4 = open('inverted_index.pkl','rb')
    # inverted_index = pickle.load(file4)   
    # file5 = open('overall_postinglist.pkl','rb')
    # universal_list = pickle.load(file5)   



