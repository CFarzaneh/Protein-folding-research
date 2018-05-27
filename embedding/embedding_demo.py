import numpy as np
import os
import pandas as pd
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Input, Add
from keras.models import Model
#import livelossplot
import keras

# Just checking on a single file 
# For all the 1000 file, you can use each file as a batch

dataPath = "/Users/cfarzaneh/Desktop/tensor_data/"
labelPath = "/Users/cfarzaneh/Desktop/tensor_label/"

filelist = os.listdir(dataPath)
data = []
label = []

for i in filelist:
    print(i)
    data.append(np.load(dataPath+i))
    fileName = i.split('_')[0]
    label.append(np.load(labelPath+fileName+"_label.npy"))

data = np.concatenate(data, axis=0)
label = np.concatenate(label, axis=0)

print(data.shape)
print(label.shape)

# Change label to one-hot encoding
# For all the file, you need to club all the labels files together and then change into one-hot encoding for sync
from sklearn.preprocessing import OneHotEncoder
protein_label_encoder = pd.factorize(label)
encoder = OneHotEncoder()
protein_labels_1hot = encoder.fit_transform(protein_label_encoder[0].reshape(-1,1))
onehot_array = protein_labels_1hot.toarray()

print(onehot_array.shape)

data1 = data.reshape((-1, 21, 19, 19, 19, 1))
print(data1.shape)

# For parallel 21 computations, Create 21 list to insert the model
data2 = [[] for _ in range(21)]
for sample in data1:
    for ind,val in enumerate(sample):
        data2[ind].append(val)
data2 = [np.array(i) for i in data2]

# Demo Architecture 
#plot_losses = livelossplot.PlotLossesKeras()

def parallel_computation(inputs):
    convs = []
    for i in range(21):
        conv = Conv3D(3,kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu', input_shape=(19,19,19,1))(inputs[i])
        convs.append(conv)
    return keras.layers.Add()(convs)
    
inputs = [Input(shape=(19,19,19,1)) for _ in range(21)] #19x19x19
adds = parallel_computation(inputs)

conv1 = Conv3D(1,kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu', input_shape=(19,19,19,3))(adds) #3x3x3
pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv1)

conv2 = Conv3D(1,kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu')(pool1) #3x3x3
pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv2)

conv3 = Conv3D(1,kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu')(pool2) #3x3x3
pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv3)

#another convolution layer, max pooling, another convolution layer 3x3x3

flatten1 = Flatten()(pool3)
dense1 = Dense(100, activation='relu')(flatten1)
out = Dense(20, activation='softmax')(dense1)
model = Model(input= inputs,output = out)

model.summary()

# Run it
model.compile(loss=keras.losses.categorical_crossentropy,
          optimizer=keras.optimizers.Adam(lr=0.001), #.0001 decrease learning rate
          metrics=['accuracy'])

model.fit(data2, onehot_array,
      batch_size=10,
      epochs=20,
      verbose=1,
      validation_split=0.2)