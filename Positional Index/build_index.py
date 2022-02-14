from preprocess import *
import pickle

class node:
    def __init__(self,doc_id,pos=[],freq=0):
        self.doc_id = doc_id
        self.freq = freq
        self.pos = pos
        self.next = None

class linked_list:
    def __init__(self, head=None, tail=None):
        self.head = head
        self.tail = tail
        self.len = 0        

    def append(self,id,freq,pos):
        new_node = node(id,pos,freq)        
        if self.head is None:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.len += 1

    def display(self,file_info):
        temp = self.head
        print("No of Docs: ",self.len)
        while(temp):
            print("Doc ID: {} Term Freq: {} Positions: {} File Name: {}".format(temp.doc_id,temp.freq,temp.pos,file_info[temp.doc_id]))
            temp = temp.next

def buildIndex(path):
    doc_id = 1     
    inverted_index = {}   
    file_info = {}
    for file in glob.glob(path): 
        fpath = file
        fname = file.split("\\")[1]        
        fname = fname.split(".")[0]                                           
        if os.path.isdir(file):                        
            if fname == "SRE":                
                for file1 in glob.glob(file+'/*'):  
                    fpath1 = file1                                                    
                    fname1 = file1.split("\\")[2]        
                    fname1 = fname1.split(".")[0]                    
                    if fname1 == "" or fname1=="index":
                        continue
                    else:
                        # universal_list.append(doc_id,0) 
                        print(doc_id,fname1)                        
                        file = open(file1,"r",encoding='unicode_escape')        
                        doc = file.read() #reading contents of doc        
                        doc = delete_spec_chars(str(doc)) #deleting special characters
                        # doc = re.sub(r'\d+','',doc) #deleting numbers
                        tokens = word_tokenize(doc) #extracting tokens
                        tokens_lower = [word.lower() for word in tokens] #Removing stopwords                                                           
                        tokens_final = [word for word in tokens_lower if word not in stop_words and len(word) > 1]                                      
                        file_info[doc_id] = os.path.basename(fpath1)
                        for pos,word in enumerate(tokens_final):                            
                            if word in inverted_index:                                 
                                found = 0                             
                                temp = inverted_index[word].head
                                while(temp):
                                    if temp.doc_id == doc_id:
                                        temp.pos.append(pos)
                                        temp.freq = temp.freq + 1
                                        found = 1
                                        break
                                    temp = temp.next
                                if found == 0:
                                    pos_list = []             
                                    pos_list.append(pos)   
                                    inverted_index[word].append(doc_id,1,pos_list)
                            else:                                
                                inverted_index[word] = linked_list()    
                                pos_list = []             
                                pos_list.append(pos)               
                                inverted_index[word].append(doc_id,1,pos_list)                                                 
                        doc_id += 1                        
            else:
                continue   
        else:                                          
            if fname == "index":
                continue
            # universal_list.append(doc_id,0)                        
            print(doc_id,fname)
            file = open(file,"r",encoding='unicode_escape')        
            doc = file.read() #reading contents of doc        
            doc = delete_spec_chars(str(doc)) #deleting special characters
            # doc = re.sub(r'\d+','',doc) #deleting numbers
            tokens = word_tokenize(doc) #extracting tokens
            tokens_lower = [word.lower() for word in tokens] #Removing stopwords                                               
            tokens_final = [word for word in tokens_lower if word not in stop_words and len(word) > 1] 
            file_info[doc_id] = os.path.basename(fpath) 
            for pos,word in enumerate(tokens_final):                            
                if word in inverted_index:                       
                    found = 0                             
                    temp = inverted_index[word].head
                    while(temp):
                        if temp.doc_id == doc_id:                            
                            temp.pos.append(pos)
                            temp.freq = temp.freq + 1
                            found = 1
                            break
                        temp = temp.next
                    if found == 0:
                        pos_list = []             
                        pos_list.append(pos)   
                        inverted_index[word].append(doc_id,1,pos_list)
                else:                      
                    inverted_index[word] = linked_list()    
                    pos_list = []             
                    pos_list.append(pos)               
                    inverted_index[word].append(doc_id,1,pos_list)                          
            doc_id += 1
    return inverted_index, file_info

# if __name__ == "__main__":     
#     inverted_index, file_info = buildIndex('stories/*')
    # f = open('inverted_index.pkl','wb')
    # pickle.dump(inverted_index,f)
    # f.close()    
    # file4 = open('inverted_index.pkl','rb')
    # inverted_index = pickle.load(file4)   
    # i = 1
    # for word in sorted(inverted_index.keys()):
    #     if word in stop_words:
    #         print(word)
    #         inverted_index[word].display(file_info)
        # if i==5:
        #     break
        # i += 1
   
       


