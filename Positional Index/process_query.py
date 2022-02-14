from preprocess import *
from build_index import *
import pickle

if __name__ == "__main__":
    i=1
    inverted_index, file_info = buildIndex('../stories/*')
    for word in inverted_index.keys():
        print(word)
        inverted_index[word].display(file_info)
        if i==1:
            break
        i += 1
    n = int(input("Enter number of queries "))
    while(n):
        query = input("Enter input query ")
        processed_query = process_query(query)
        process_word = processed_query.split()
        print(process_word)
        not_found = 0
        result = linked_list()
        if len(process_word) > 1:
            for i in range(len(process_word)-1):        
                resultant = linked_list()
                word1 = process_word[i]
                word2 = process_word[i+1]     
                if i == 0:
                    temp1 = inverted_index[word1].head
                else:
                    temp1 = result.head        
                temp2 = inverted_index[word2].head 
                flag = 0       
                while temp1 and temp2:
                    if temp1.doc_id == temp2.doc_id:                
                        pos_list = []
                        for pos1 in temp1.pos:
                            for pos2 in temp2.pos:
                                if pos1 == pos2 - 1:
                                    pos_list.append(pos2)
                                elif pos2 > pos1:
                                    break
                        if len(pos_list) > 0:   
                            flag = 1
                            resultant.append(temp1.doc_id,len(pos_list),pos_list)  
                            result = resultant                     
                        elif flag==0:
                            result = linked_list()
                        temp1 = temp1.next
                        temp2 = temp2.next
                    elif temp1.doc_id < temp2.doc_id:
                        temp1 = temp1.next
                    else:
                        temp2 = temp2.next
        else:
            if process_word[0] in inverted_index:
                result = inverted_index[process_word[0]]
            else:
                not_found = 1
        if not_found == 0:
            result.display(file_info)
        else:
            print("Phrase not found")
        n -= 1
    
    
