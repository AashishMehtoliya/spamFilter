# -*- coding: utf-8 -*-
"""
Created on Wed May 29 19:15:47 2019

@author: Aashish Mehtoliya
"""

import os
import pickle 
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def make_dictionary(root_dir):
    all_words = []
    emails = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
    for mail in emails:
        #print(mail)
        with open(mail) as m:
            for line in m:
                words = line.split()
                all_words += words
    dictionary = Counter(all_words)
    #print(dictionary)
    list_to_remove = dictionary.keys()
    
    for item in list(list_to_remove):
        if item.isalpha() == False:
            del dictionary[item]
            
        elif len(item) == 1:
            del dictionary[item]
    
    dictionary = dictionary.most_common(3000)
#    for i,d in enumerate(dictionary):
#        print(i)
    
    return dictionary
        
def extract_feature(mail_dir):
    files = [os.path.join(mail_dir,f) for f in os.listdir(mail_dir)]
    feature_matrix = np.zeros((len(files),3000))
    train_labels = np.zeros(len(files))
    count = 0;
    docID = 0;
    
    for fil in files:
        #print(fil)
        with open(fil) as fi:
            for i,line in enumerate(fi):
                #print(i,line)
                #print('#########################')
                if i==2:
                    words = line.split()
                    #print(words)
                    for word in words:
                        wordID = 0
                        for i,d in enumerate(dictionary):
                            if d[0]== word:
                                wordID = i
                                feature_matrix[docID,wordID] = words.count(word)
            train_labels[docID] = 0;
            filepathToken = fil.split('\\')
#            print(filepathToken)
            lastToken = filepathToken[len(filepathToken)-1]
            
            #print(len(filepathToken))
            if lastToken.startswith("spmsg"):
                train_labels[docID] = 1
                count = count+1
            docID = docID+1
            #print(feature_matrix)
            
    return feature_matrix,train_labels

train_dir = 'D:/python prog/train-mails'
test_dir = 'D:/python prog/test-mails'


#dictionary = make_dictionary(train_dir)
#pickle_out = open('dict.pickle','wb')
#pickle.dump(dictionary,pickle_out)
#pickle_out.close()

pickle_in = open('dict.pickle','rb')
dictionary = pickle.load(pickle_in)

feature_matrix,labels = extract_feature(train_dir)
test_feature_matrix, test_labels = extract_feature(test_dir)

model = MultinomialNB()

model.fit(feature_matrix,labels)

predicted_labels = model.predict(test_feature_matrix)

print(accuracy_score(test_labels,predicted_labels))

                
                                
    