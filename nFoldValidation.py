#prepare data for n-fold cross-validation

import random
import configparser
import pickle
import copy

#the number of pieces

N = 5

#read config file to get path to the dataset
#read config file
config = configparser.RawConfigParser()
config.read('config.txt')
AllDataFile = config.get('data paths', 'all_data_file')
ClassesLabelsDataFile = config.get('data paths', 'classes_labels_data_file')

#read data
data = pickle.load(open(AllDataFile,"rb"))
random.shuffle(data)

numTotal = len(data)
numPart = int(numTotal/N)

test = []
train = []

for i in range(N):
    
    train = copy.copy(data)
    test = []
    i1 = i*numPart
    i2 = (i+1)*numPart
    
    print(" -----> N: %d, range: %d - %d" % (i, i1 , i2))
    
       
    for j in range (i1, i2):
        test.append(train[j])
    
    print(" -------> test len: ", len(test))
    
    for j in range(len(test)):
        train.remove(test[j])
    
    print(" -------> train len: ", len(train))
    
    pathTest = "test%d.p" % (i+1)
    pickle.dump(test, open(pathTest, "wb"))
    
    pathTrain = "train%d.p" % (i+1)
    pickle.dump(train, open(pathTrain, "wb"))
