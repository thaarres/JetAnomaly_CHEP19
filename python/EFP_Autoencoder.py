#!/usr/bin/env python
import setGPU
import numpy as np
import h5py, sys, glob
import keras
import random
from sklearn.utils import shuffle


# Dataset preparation
fileIN = shuffle(glob.glob("../data/training/qcd*SIDEBAND*.h5"), random_state=1111)
i_train = int(0.5*len(fileIN))
i_test = int(0.75*len(fileIN))
X_train = fileIN[:i_train]
X_test = fileIN[i_train:i_test]
print(len(fileIN), len(X_train), len(X_test))
minN = 4096

# Model Definition
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.layers import BatchNormalization
from keras.utils import plot_model
from keras import regularizers
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.regularizers import l1

inputShape = 102

inputArray = Input(shape=(inputShape,))
x = BatchNormalization()(inputArray)
x = Dense(50, activation="relu", kernel_initializer='lecun_uniform', name='enc_0')(x)
x = Dense(25, activation="relu", kernel_initializer='lecun_uniform', name='enc_1')(x)
enc = Dense(10, activation="linear", kernel_initializer='lecun_uniform', name='enc_2')(x)

x = Dense(25, activation="relu", kernel_initializer='lecun_uniform', name='dec_0')(enc)
x = Dense(50, activation="relu", kernel_initializer='lecun_uniform', name='dec_1')(x)
output = Dense(inputShape, activation="linear", kernel_initializer='lecun_uniform', name='dec_2')(x)

model = Model(inputs=inputArray, outputs=output)
encoder = Model(inputs=inputArray, outputs=enc)

model.compile(optimizer='adam', loss='mse')
encoder.compile(optimizer='adam', loss='mse')
#model.summary()

# Data Generator
class DataGenerator(keras.utils.Sequence):
    def __init__(self, label, fileList, batch_size, batch_per_file, verbose =0):
        self.verbose = verbose
        self.label = label
        self.fileList = fileList
        self.batch_size = batch_size
        self.batch_per_file = batch_per_file
        # open first file
        self.f =  h5py.File(fileList[0],"r")
        self.X =  np.array(self.f.get('EFP'))
        self.X =  np.concatenate((self.X[:,:,0], self.X[:,:,1]))
        self.X = shuffle(self.X)
        self.y = self.X
        self.nBatch = 0
        self.iFile = 0
        #self.on_epoch_end()

    #def on_epoch_end(self):
    #    print("%s boh" %self.label)

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        if self.verbose: print("%s LEN = %i" %(self.label, self.batch_per_file*len(self.fileList)))
        return self.batch_per_file*len(self.fileList)

    def __getitem__(self, index): 
        if index == 0:
            # reshuffle data
            if self.verbose: print("%s new epoch" %self.label)
            random.shuffle(self.fileList)
            self.iFile = 0
            self.nBatch = 0
            if self.verbose: print("%s new file" %self.label)
            if self.f != None: self.f.close()
            self.f = h5py.File(self.fileList[self.iFile], "r")
            self.X =  np.array(self.f.get('EFP'))
            self.X =  np.concatenate((self.X[:,:,0], self.X[:,:,1]))
            self.X = shuffle(self.X)
            self.y = self.X
        if self.verbose: print("%s: %i" %(self.label,index))

        #'Generate one batch of data'
        iStart = index*self.batch_size
        iStop = min(9999, (index+1)*self.batch_size)
        if iStop == 9999: iStart = iStop-self.batch_size
        myx = self.X[iStart:iStop,:]
        myy = self.y[iStart:iStop,:]
        if self.nBatch == self.batch_per_file-1:
            self.iFile+=1
            if self.iFile >= len(self.fileList):
                if self.verbose: print("%s Already went through all files" %self.label)
            else:
                if self.verbose: print("%s new file" %self.label)
                self.f.close()
                self.f = h5py.File(self.fileList[self.iFile], "r")
                self.X =  np.array(self.f.get('EFP'))
                self.X =  np.concatenate((self.X[:,:,0], self.X[:,:,1]))
                self.X = shuffle(self.X)
                self.y = self.X
            self.nBatch = 0
        else:
            self.nBatch += 1
        return myx, myy 


# Training
batch_size = 128
file_length = minN

my_batch_per_file = int(file_length/batch_size)
myTrainGen = DataGenerator("TRAINING", X_train, batch_size, my_batch_per_file)
myTestGen = DataGenerator("TEST", X_test, batch_size, my_batch_per_file)

n_epochs = 5000
verbosity = 2

history = model.fit_generator(generator=myTrainGen, epochs=n_epochs,
                    steps_per_epoch= my_batch_per_file*len(X_train), validation_data = myTestGen,
                    validation_steps =  my_batch_per_file*len(X_test), verbose=verbosity,
                    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=verbosity),
                                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=verbosity),
                                 TerminateOnNaN()])


# # Save Model
nameModel = 'AE_EFP_LongTrain'
# store history                                                                                                         
f = h5py.File("../models/%s_history.h5" %nameModel, "w")
f.create_dataset("training_loss", data=np.array(history.history['loss']),compression='gzip')
f.create_dataset("validation_loss", data=np.array(history.history['val_loss']),compression='gzip')
f.close()

# store model                                                                                                           
model_json = model.to_json()
with open("../models/%s.json" %nameModel, "w") as json_file:
    json_file.write(model_json)
model.save_weights("../models/%s.h5" %nameModel)
model_json = encoder.to_json()
with open("../models/%s_ENCODER.json" %nameModel, "w") as json_file:
    json_file.write(model_json)
encoder.save_weights("../models/%s_ENCODER.h5" %nameModel)