import numpy as np
import os
import pandas as pd
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Input, Add, BatchNormalization, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras import regularizers
import keras
from time import time
from tqdm import tqdm
<<<<<<< HEAD
# Just checking on a single file
=======

# Just checking on a single file 
>>>>>>> 326d6e61b457feb42e7814d8d1ff0a0f4602ba54
# For all the 1000 file, you can use each file as a batch

dataPath = "/home/femi/Desktop/tensor_data/"
labelPath = "/home/femi/Desktop/tensor_label/"

filelist = os.listdir(dataPath)
data = []
label = []

print("Loading data")
<<<<<<< HEAD
j =0
numOfFilesToInput = 100

for i in tqdm(filelist, total=numOfFilesToInput):
    print(i)
=======
j = 0
numOfFilesToInput = 100 #Number of files to load at once
for i in tqdm(filelist, total=numOfFilesToInput):
>>>>>>> 326d6e61b457feb42e7814d8d1ff0a0f4602ba54
    data.append(np.load(dataPath+i))
    fileName = i.split('_')[0]
    label.append(np.load(labelPath+fileName+"_label.npy"))
    if j == numOfFilesToInput:
<<<<<<< HEAD
        break
    j +=1
=======
      break
    j+=1
>>>>>>> 326d6e61b457feb42e7814d8d1ff0a0f4602ba54

data = np.concatenate(data, axis=0)
label = np.concatenate(label, axis=0)

#print(data.shape)
#print(label.shape)

# Change label to one-hot encoding
# For all the file, you need to club all the labels files together and then change into one-hot encoding for sync
from sklearn.preprocessing import OneHotEncoder
protein_label_encoder = pd.factorize(label)
encoder = OneHotEncoder()
protein_labels_1hot = encoder.fit_transform(protein_label_encoder[0].reshape(-1,1))
onehot_array = protein_labels_1hot.toarray()

data1 = data.reshape((-1, 21, 19, 19, 19, 1))
print(data1.shape)
print(onehot_array.shape)

# For parallel 21 computations, Create 21 list to insert the model
data2 = [[] for _ in range(21)]
for sample in data1:
    for ind,val in enumerate(sample):
        data2[ind].append(val)
data2 = [np.array(i) for i in data2]

<<<<<<< HEAD
# Demo Architecture
#plot_losses = livelossplot.PlotLossesKeras()
=======
# Demo Architecture 
>>>>>>> 326d6e61b457feb42e7814d8d1ff0a0f4602ba54

def parallel_computation(inputs):
    convs = []
    for i in range(21):
        conv = Conv3D(3,kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu', input_shape=(19,19,19,1))(inputs[i])
        convs.append(conv)
    return keras.layers.Add()(convs)

inputs = [Input(shape=(19,19,19,1)) for _ in range(21)] #19x19x19
adds = parallel_computation(inputs)

conv1 = Conv3D(1,kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', input_shape=(19,19,19,3))(adds) #3x3x3
<<<<<<< HEAD
norm2 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001)(conv1)
conv2 = Conv3D(1,kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(norm2)
pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv2)


conv3 = Conv3D(1,kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', activity_regularizer=regularizers.l2(0.01))(pool1) #3x3x3
norm3 = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001)(conv3)
pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(norm3)

'''
conv3 = Conv3D(1,kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu')(pool2) #3x3x3
pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv3)
'''
#another convolution layer, max pooling, another convolution layer 3x3x3

flatten1 = Flatten()(pool2)
dense1 = Dense(300, activation='relu', activity_regularizer=regularizers.l2(0.01))(flatten1)
drop2 = Dropout(0.5)(dense1)
out = Dense(20, activation='softmax')(drop2)
=======
conv2 = Conv3D(1,kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(conv1)
pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv2)

#conv2 = Conv3D(1,kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu')(pool1) #3x3x3
#pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv2)

#conv3 = Conv3D(1,kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu')(pool2) #3x3x3
#pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv3)

#another convolution layer, max pooling, another convolution layer 3x3x3

flatten1 = Flatten()(pool1)
dense1 = Dense(200, activation='relu')(flatten1)
out = Dense(20, activation='softmax')(dense1)
>>>>>>> 326d6e61b457feb42e7814d8d1ff0a0f4602ba54
model = Model(input= inputs,output = out)

model.summary()

# Run it
model.compile(loss=keras.losses.categorical_crossentropy,
          optimizer=keras.optimizers.Adam(lr=0.001), #.0001 decrease learning rate
          metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

#Attempted to reduce learning rate to prevent it 'val_loss' from going up.
#reduce_lr can be used as a callback during fitting

#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr = 0.00025)

model.fit(data2, onehot_array,
      batch_size=20,
<<<<<<< HEAD
      epochs=100,
=======
      epochs=10,
>>>>>>> 326d6e61b457feb42e7814d8d1ff0a0f4602ba54
      verbose=1,
      callbacks=[tensorboard],
      validation_split=0.2)