from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from numpy import array
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import keras
import mmap
import os

# Hyperparameters
Window_width = 2
Grid_cell_size = 1
Grid_X_size = 21
Grid_Y_size = 21
Grid_Z_size = 21
Coord_X_min = -10
Coord_Y_min = -10
Coord_Z_min = -10

batchOfTensors = []
proteinLabels = []

Path = "inputFiles"

if not os.path.exists(Path + '_out'):
	os.makedirs(Path + '_out')

def get_num_lines(file_path):
	fp = open(file_path, "r+")
	buf = mmap.mmap(fp.fileno(), 0)
	lines = 0
	while buf.readline():
		lines += 1
	return lines

filelist = os.listdir(Path)
for i in filelist:
	if i.endswith(".txt"):
		fileName = i.split('.')[0]
		print(i)
		if os.path.isfile(Path + '_out/' + fileName + '.txt'):
			with open(Path + '_out/' + fileName + '.txt', 'rb') as fp:
				batchOfTensors += pickle.load(fp)
			with open(Path + '_out/' + fileName + '_protein' + '.txt', 'rb') as fp:
				proteinLabels += pickle.load(fp)
		else:
			with open(Path + '/' + i, 'r') as f:
				fileTensor = []
				fileProteins = []
				for line in tqdm(f, total=get_num_lines(Path + '/' + i)):
					line = line.strip().split('\t')
					protein = line.pop(0)
					#print(protein)
					
					proteinLabels.append(protein)
					fileProteins.append(protein)

					coord_list = np.array([])
					for i in line:
						j = i.split(',')
						coord_list = np.concatenate([coord_list, j])

					coord_list = np.reshape(coord_list,(-1,4))
					nrows = coord_list.shape[0]

					tensor = np.zeros((Grid_X_size,Grid_Y_size,Grid_Z_size,22)) #4D-Tensor

					for x in range(0,Grid_X_size):
						Xx = (x*Grid_cell_size)+Coord_X_min
						for y in range(0,Grid_Y_size):
							Yy = (y*Grid_Y_size)+Coord_Y_min
							for z in range(0,Grid_Z_size):
								Zz = (z*Grid_cell_size)+Coord_Z_min
								for m in range(0,nrows):
									Channel = int(coord_list[m][0])
									DistanceSq = ((Xx - float(coord_list[m][1]))**2)+((Yy - float(coord_list[m][2]))**2)+((Zz - float(coord_list[m][3]))**2)
									Contribution = np.exp(-DistanceSq/(2*(Window_width**2)))
									tensor[x][y][z][Channel] += Contribution

					batchOfTensors.append(tensor)
					fileTensor.append(tensor)
				with open(Path + '_out/' + fileName + '.txt', 'wb') as fp:
					pickle.dump(fileTensor, fp)
				with open(Path + '_out/' + fileName + '_protein' + '.txt', 'wb') as fp:
					pickle.dump(fileProteins, fp)

input_data = np.array(batchOfTensors)
print(input_data.shape)

proteinLabels = np.array(proteinLabels)
protein_label_encoder = pd.factorize(proteinLabels)
encoder = OneHotEncoder()
protein_labels_1hot = encoder.fit_transform(protein_label_encoder[0].reshape(-1,1))
onehot_array = protein_labels_1hot.toarray()
print(onehot_array.shape)

input_shape = input_data[0].shape

model = Sequential()
model.add(Conv3D(22, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu', input_shape=input_shape))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1)))
model.add(Conv3D(22, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1))) 

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(len(set(proteinLabels)), activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
		  optimizer=keras.optimizers.Adam(lr=0.01),
		  metrics=['accuracy'])

model.fit(input_data, onehot_array,
	  batch_size=10,
	  epochs=3,
	  verbose=1,
	  validation_split=0.2)

model.summary()