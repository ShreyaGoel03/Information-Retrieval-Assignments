from preprocess import *
from build_index import *
import pickle

#Boolean AND - Finds the intersecting/common documents of the two posting lists
def query_and(list1, list2):
    p1 = list1.head
    p2 = list2.head
    resultant = linked_list() #Resultant Linked List formed by AND operation
    count = 0

    while p1 is not None and p2 is not None:
        if p1.doc_id == p2.doc_id: 
            count += 1
            resultant.append(p1.doc_id, p1.freq)
            p1 = p1.next
            p2 = p2.next
        
        elif p1.doc_id > p2.doc_id:
            count+= 1
            p2 = p2.next
        
        else:
            count+= 1
            p1 = p1.next
        
    return resultant, count

#Boolean OR - Finds all documents from the two posting lists
def query_or(list1, list2):
    p1 = list1.head
    p2 = list2.head
    resultant = linked_list()
    count = 0
    while p1 is not None and p2 is not None:
        if p1.doc_id == p2.doc_id:
            count += 1
            resultant.append(p1.doc_id, p1.freq)
            p1 = p1.next
            p2 = p2.next
        
        elif p1.doc_id > p2.doc_id:
            resultant.append(p2.doc_id, p2.freq)
            count+= 1
            p2 = p2.next
        
        else:
            resultant.append(p1.doc_id, p1.freq)
            count+= 1
            p1 = p1.next
        
    while p1 is not None:
        resultant.append(p1.doc_id, p1.freq)
        p1 = p1.next

    while p2 is not None:
        resultant.append(p2.doc_id, p2.freq)
        p2 = p2.next

    return resultant, count

#Subtract the documents of second posting list from the first posting list
def subtract(list1, list2):
    p1 = list1.head
    p2 = list2.head
    resultant = linked_list()
    count = 0
    while p1 is not None and p2 is not None:
        if p1.doc_id == p2.doc_id:
            count += 1
            p1 = p1.next
            p2 = p2.next
        
        elif p1.doc_id > p2.doc_id:
            resultant.append(p2.doc_id, p2.freq)
            count+= 1
            p2 = p2.next
        
        elif p2.doc_id > p1.doc_id:
            resultant.append(p1.doc_id, p1.freq)
            count+= 1
            p1 = p1.next

    while p1 is not None:
        resultant.append(p1.doc_id, p1.freq)
        p1 = p1.next
    
    return resultant, count
        
#Boolean ANDNOT - Return the documents which are present in first posting list and not second posting list
def query_andnot(list1,list2):
    resultant, count1 = subtract(list1, list2)
    return resultant, count1

#Boolean ORNOT - Return the documents which are not present in second posting list only
def query_ornot(list1, list2):
    list3, count1 = subtract(list2, list1)
    resultant, count2 = subtract(universal_list, list3)
    return resultant, count1+count2


if __name__ == "__main__":
    file1 = open('tokens.pkl','rb')
    tokens = pickle.load(file1)
    file2 = open('token_freq.pkl','rb')
    tokens_freq = pickle.load(file2)        
    file3 = open('file_info.pkl','rb')
    file_info = pickle.load(file3)     
    universal_list = linked_list()    
    inverted_index = {}  
    #Building the Inverted Index   
    for word in tokens:
        inverted_index[word] = linked_list() 
    inverted_index, universal_list = buildIndex('stories/*',inverted_index,universal_list)
    num = int(input('Enter the Number of Queries '))
    while(num):
        query = input('Enter the Query ')
        processed_query = process_query(query) #Input Query
        process_word = processed_query.split()
        query_op = input('Enter the Operations ') #Input Operations
        query_op = query_op.replace('[','')
        query_op = query_op.replace(']','')
        query_op = query_op.split(',')
        s = ""
        for i in range(len(query_op)):
            s += process_word[i]
            s += " "
            s += query_op[i]
            s += " "
        s += process_word[len(process_word)-1]
        print("Processed Query ", s)
        total = [] #Storing the Posting list for each word in Query
        i=0
        for word in process_word:
            if word in tokens:
                total.append(inverted_index[word])

        final_count = 0
        for i in range(len(process_word)-1):
            word1 = total[0]
            word2 = total[1]
            if query_op[i] == 'OR':
                word3,count1 = query_or(word1,word2)
            elif query_op[i] == 'AND':
                word3,count1 = query_and(word1,word2)
            elif query_op[i] == 'OR NOT':
                word3,count1 = query_ornot(word1,word2)
            elif query_op[i] == 'AND NOT':
                word3,count1 = query_andnot(word1,word2)
            final_count += count1
            total.remove(word1)
            total.remove(word2)
            total.insert(0,word3)
        resultant = total[0]
        print("Number of Documents Matched ", total[0].len)
        print("Number of Comparisons Required ", final_count)
        print("Documents Retrieved")
        resultant.display(file_info)
        num -= 1