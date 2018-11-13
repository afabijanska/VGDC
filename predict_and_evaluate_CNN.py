import configparser
import pickle
import numpy as np
import operator

from keras.models import model_from_json
from helpers import genome2tabInt, plot_confusion_matrix 

#read config file
config = configparser.RawConfigParser()
config.read('config.txt')
TestDataFile = config.get('data paths', 'test_data_file')
PredDataFile = config.get('data paths', 'predictions_data_file')
ClassesLabelsDataFile = config.get('data paths', 'classes_labels_data_file')
modelJsonFile = config.get('network params', 'model_json_file')
modelVisFile = config.get('network params', 'model_vis_file')
bestWeightsFile = config.get('network params', 'best_weights')
lastWeightsFile = config.get('network params', 'last_weights')
batchSize = int(config.get('network params', 'batch_size'))

#read class-labels correspondences
diLabels = pickle.load(open(ClassesLabelsDataFile, "rb"))
numClasses = len(diLabels)

#load model and the corresponding weights
model = model_from_json(open(modelJsonFile).read())
model.load_weights(bestWeightsFile)

#read and prepare testing data
test = pickle.load(open(TestDataFile,"rb"))
n_test = len(test)
max_ = len(test[0][1])

test_genomes = []
test_labels = []

for i in range(len(test)):
    test_genomes.append(genome2tabInt(test[i][1], max_))
    labels = np.zeros(numClasses,dtype='float16')
    labels[diLabels[test[i][0]]-1] = 1
    test_labels.append(labels)
    
print('Shape of data tensor:', np.asarray(test_genomes).shape)
print('Shape of label tensor:', np.asarray(test_labels).shape)

#feed CNN into the CNN
x_test = np.reshape(np.asarray(test_genomes), (n_test, max_, 1)).astype('float16')
y_test = np.reshape(np.asarray(test_labels), (n_test, numClasses))

y_pred = model.predict(x_test, batch_size=batchSize, verbose=2)
print ("predicted images size :")
print (y_pred.shape)
y_pred = y_pred.astype('float16')

#evaluate predictions
#global evaluation
y_pred2 = np.argmax(y_pred, axis=1)
y_test2 = np.argmax(y_test, axis=1)

total = 0
okays = 0

for i in range(y_pred2.shape[0]):
    total += 1
    if (y_pred2[i] == y_test2[i]):
        okays += 1
        
print("total acc: ", 100*okays/total)
print("correct classifications: ", okays)
print("errors: ", total - okays)

#evaluation by class
diClasses = pickle.load(open(ClassesLabelsDataFile, "rb"))

confusionMatrix = np.zeros((numClasses, numClasses), dtype = 'int')

for i in range(y_pred2.shape[0]):
    confusionMatrix[y_test2[i], y_pred2[i]] += 1


sorted_by_labels = sorted(diClasses.items(), key=operator.itemgetter(1))
p, q = zip(*sorted_by_labels)

plot_confusion_matrix(cm=confusionMatrix, target_names = p, normalize=False)
plot_confusion_matrix(cm=confusionMatrix, target_names = p, normalize=True)