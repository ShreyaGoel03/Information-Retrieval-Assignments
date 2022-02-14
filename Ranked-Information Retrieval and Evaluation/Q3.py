#Libraries Used
import math
from matplotlib import pyplot as plt

#Loading the Dataset
path = "IR-assignment-2-data.txt"
file = open(path, 'r',encoding='ISO-8859-1')
data_doc = file.readlines()
data=[]
#Getting data only of qid:4
for x in data_doc:
  if(x.split()[1]=='qid:4'):
      data.append(x)
print("Total Qid:4 Data ", len(data))
print("Data")
print(data)


#Function to find DCG
def discounted_cumulative_gain(data, count):

  dcg = 0 #Initialising the DCG
  i = 1 #Rank

  #Iterating over data
  for x in data:
    if i <= count and i != 1: #Iterate over till the position specified
      #Calculating the DCG
      relevance = int(x.split()[0])
      numerator = relevance
      denominator = math.log2(i)
      dcg = dcg + (numerator/denominator)
    elif i == 1:
      relevance = int(x.split()[0])
      dcg = relevance
    else:
      break
    i += 1
  return dcg

#Q3_a
def Q3_a(data):
  file_dcg = open("Q3_a.text", "w")
  labels = set()

  for query in data:
    #Relevance score labels 
    labels.add(query.split()[0])

  sorted_dcg = data.copy()
  sorted_dcg.sort(reverse = True)

  labels = sorted(labels)

  #Initialising the number of URL pairs for each relevance score
  count_relevance = {}
  for i in labels:
    count_relevance[i] = 0
  print(count_relevance)

  #Adding data in the file sorted according to max DCG 
  print("Data Stored")
  for x in sorted_dcg:
    print(x)
    file_dcg.write(x)
    count_relevance[x[0]] += 1

  #Max DCG
  max_dcg = discounted_cumulative_gain(sorted_dcg, len(sorted_dcg))
  print("Obtained max DCG", max_dcg)

  #Counting the number of files that can be made
  number_of_files = 1
  for count in count_relevance.values():
    number_of_files *= math.factorial(count)
  
  file_dcg.close()

  print("Number of files that can be made ", number_of_files)
  print("Count of relevance score ", count_relevance)

  return count_relevance

count_relevance = Q3_a(data)

#Q3_b
def Q3_b(data, position):

  #Finding the DCG at the position
  dcg = discounted_cumulative_gain(data, position)

  #Sorting the data  in decreasing order according to relevance score which will the ideal rankings  
  sorted_data = data.copy()
  sorted_data = sorted_data[0:position]
  sorted_data.sort(reverse = True)

  #Finding the Ideal DCG at that position
  idcg = discounted_cumulative_gain(sorted_data, position)

  #Calculating the Normalised DCG
  ndcg = dcg/idcg

  if position == len(data):
    print("\nWhole Dataset ")
  else:
    print("\nPosition at ", position)
  
  print("DCG: ", dcg)
  print("IDCG: ", idcg)
  print("NDCG: ", ndcg)

Q3_b(data, 50)
Q3_b(data, len(data))

#Q3_c
def Q3_c(data):

  #Extracting the feature 75 values
  data_feature_75 = {}
  for query in data:
    value = query.split()[76]
    value = value[3:]
    data_feature_75[query] = float(value)
  
  #Sorting the data according to the 75 value
  sorted_values = sorted(data_feature_75.items(), key=lambda item: item[1], reverse = True)
  sorted_dcg_dict = {k: v for k, v in sorted_values}

  #Total relevant docs exclude the one with relevance score as 0
  total_relevant_docs = len(data) - count_relevance['0']
  total_retrieved_docs = 0

  precision = []
  recall = []

  #Iterating over data
  current_relevant_docs = 0
  for query in sorted_dcg_dict:
    #If relevant score is 0 then not a relevant document
    if query.split()[0] != '0':
      current_relevant_docs += 1

    total_retrieved_docs += 1

    #Finding the precision and recall
    precision_value = current_relevant_docs/total_retrieved_docs
    recall_value = current_relevant_docs/total_relevant_docs
    precision.append(precision_value)
    recall.append(recall_value)

  print("Precision Recall Curve")
  #Plotting Precision vs Recall
  plt.plot(recall, precision)
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('A Precision-Recall Curve')
  plt.show()


Q3_c(data)

