from keras.layers import Conv1D, MaxPooling1D, Dropout, Input
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.models import Model
from keras.layers.core import Dense, Flatten
from keras.layers.normalization import BatchNormalization

import pickle
import configparser
import numpy as np

from helpers import getStats, plotDict, genome2tabInt 

#define CNN architecture
def getNetwork(maxLen, numClasses, maskSize):                  
    
    inputs = Input(shape=(maxLen,1))
    
    conv1 = Conv1D(filters=8, kernel_size=(maskSize), strides=(2), padding='valid', activation='relu')(inputs)
    pool1 = MaxPooling1D(pool_size=(2), strides=(2), padding='valid')(conv1)
    norm1 = BatchNormalization()(pool1)

    conv2 = Conv1D(filters=16, kernel_size=(maskSize), strides=(2), padding='valid', activation='relu')(norm1)
    pool2 = MaxPooling1D(pool_size=(2), strides=(2), padding='valid')(conv2)
    norm2 = BatchNormalization()(pool2)

    conv3 = Conv1D(filters=32, kernel_size=(maskSize), strides=(2), padding='valid', activation='relu')(norm2)
    pool3 = MaxPooling1D(pool_size=(2), strides=(2), padding='valid')(conv3)
    norm3 = BatchNormalization()(pool3)

    conv4 = Conv1D(filters=64, kernel_size=(maskSize), strides=(2), padding='valid', activation='relu')(norm3)
    pool4 = MaxPooling1D(pool_size=(2), strides=(2), padding='valid')(conv4)
    norm4 = BatchNormalization()(pool4)

    conv5 = Conv1D(filters=128, kernel_size=(maskSize), strides=(2), padding='valid', activation='relu')(norm4)
    pool5 = MaxPooling1D(pool_size=(2), strides=(2), padding='valid')(conv5)
    norm5 = BatchNormalization()(pool5)

    flat6 = Flatten()(norm5)
    dens6 = Dense(256, activation='relu')(flat6)
    drop6 = Dropout(0.4)(dens6)
    norm6 = BatchNormalization()(drop6)
    
    dens7 = Dense(128, activation='relu')(norm6)
    drop7 = Dropout(0.4)(dens7)
    norm7 = BatchNormalization()(drop7)

    dens8 = Dense(64, activation='relu')(norm7)
    drop8 = Dropout(0.4)(dens8)
    norm8 = BatchNormalization()(drop8)

    dens9 = Dense(numClasses, activation='softmax')(norm8)
    
    model = Model(inputs=inputs, outputs=dens9)
    
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['mse'])

    return model

#read config file
config = configparser.RawConfigParser()
config.read('config.txt')
TrainDataFile = config.get('data paths', 'train_data_file')
ClassesLabelsDataFile = config.get('data paths', 'classes_labels_data_file')

#load training and testing data
train = pickle.load(open(TrainDataFile, "rb"))
max_ = len(train[0][1])
 
#get some stats about training and testing dataset
diTrain = getStats(train)
plotDict(diTrain, 'train.png')

#create labels for classess
diLabels = {}
classId = 0;
numClasses = len(diTrain)

for item in diTrain:
    classId += 1
    diLabels[item] = classId;
    
print(diLabels)

#save label-class correspondences to file
pickle.dump(diLabels, open(ClassesLabelsDataFile, "wb"))
 
#prepare training data for feeding it to CNN
n_train = len(train)

train_genomes = []
train_labels = []

for i in range(len(train)):
    train_genomes.append(genome2tabInt(train[i][1], max_))
    labels = np.zeros(numClasses,dtype='float16')
    labels[diLabels[train[i][0]]-1] = 1
    train_labels.append(labels)

print('Shape of data tensor:', np.asarray(train_genomes).shape)
print('Shape of label tensor:', np.asarray(train_labels).shape)

x = np.reshape(np.asarray(train_genomes), (n_train, max_, 1)).astype('float16')
y = np.reshape(np.asarray(train_labels), (n_train, numClasses))

#read config file to get training params
maskSize = int(config.get('network params', 'filters_size'))
poolStrides = int(config.get('network params', 'pool_strides'))
batchSize = int(config.get('network params', 'batch_size'))
numEpochs = int(config.get('network params', 'num_epochs'))
valSplit = float(config.get('network params', 'validation_split'))
modelJsonFile = config.get('network params', 'model_json_file')
modelVisFile = config.get('network params', 'model_vis_file')
bestWeightsFile = config.get('network params', 'best_weights')
lastWeightsFile = config.get('network params', 'last_weights')

model = getNetwork(max_, numClasses, maskSize)

plot_model(model, to_file=modelVisFile, show_shapes='True')   #check how the model looks like
json_string = model.to_json()
open(modelJsonFile, 'w').write(json_string)
checkpointer = ModelCheckpoint(bestWeightsFile, verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased

model.fit(x, y, epochs = numEpochs, batch_size = batchSize, verbose=2, shuffle=True, validation_split=valSplit, callbacks=[checkpointer])
model.save_weights(lastWeightsFile, overwrite=True)