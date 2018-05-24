import numpy as np
import os
import pandas as pd
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Input, Add
from keras.models import Model

# Just checking on a single file 
# For all the 1000 file, you can use each file as a batch
data = np.load("/home/atharva/Desktop/tensor_data/"+"pdb4g9e_data.npy")
label = np.load("/home/atharva/Desktop/tensor_label/"+"pdb4g9e_label.npy")

data.shape
label.shape

# Change label to one-hot encoding
# For all the file, you need to club all the labels files together and then change into one-hot encoding for sync
from sklearn.preprocessing import OneHotEncoder
protein_label_encoder = pd.factorize(label)
encoder = OneHotEncoder()
protein_labels_1hot = encoder.fit_transform(protein_label_encoder[0].reshape(-1,1))
onehot_array = protein_labels_1hot.toarray()

print(onehot_array.shape)
print(onehot_array[254])

data1 = data.reshape((548, 21, 18, 18, 18,1))

data1.shape

# For parallel 21 computations, Create 21 list to insert the model
data2 = [[] for _ in range(21)]
for sample in data1:
    for ind,val in enumerate(sample):
        data2[ind].append(val)
data2 = [np.array(i) for i in data2]

len(data2), data2[0].shape

# Demo Architecture 
import keras
def parallel_computation(inputs):
    convs = []
    for i in range(21):
        conv = Conv3D(3,kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu', input_shape=(18,18,18,1))(inputs[i])
        convs.append(conv)
    return keras.layers.Add()(convs)
    
inputs = [Input(shape=(18,18,18,1)) for _ in range(21)]
adds = parallel_computation(inputs)
conv1 = Conv3D(1,kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu', input_shape=(18,18,18,3))(adds)
pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv1)
flatten1 = Flatten()(pool1)
dense1 = Dense(100, activation='relu')(flatten1)
out = Dense(20, activation='softmax')(dense1)
model = Model(input= inputs,output = out)

model.summary()

# Run it
model.compile(loss=keras.losses.categorical_crossentropy,
          optimizer=keras.optimizers.Adam(lr=0.001),
          metrics=['accuracy'])

model.fit(data2, onehot_array,
      batch_size=10,
      epochs=5,
      verbose=1,
      validation_split=0.2)

