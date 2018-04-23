import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense

fin = open("pdb1a1x.txt", 'r')

# Read in the data
for line in fin:
	line = line.strip().split('\t')
	protein = line.pop(0)
	print(protein)

	coord_list = np.array([])
	for i in line:
		j = i.split(',')
		coord_list = np.concatenate([coord_list, j])

	coord_list = np.reshape(coord_list,(-1,4))
	# print(coord_list)
	break #For now we're going to read only the first protein

nrows = coord_list.shape[0]

# Hyperparameters
Window_width = 2
Grid_cell_size = 1
Grid_X_size = 21
Grid_Y_size = 21
Grid_Z_size = 21
Coord_X_min = -10
Coord_Y_min = -10
Coord_Z_min = -10

tensor = np.zeros((Grid_X_size,Grid_Y_size,Grid_Z_size,21)) #4D-Tensor

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

				#Channel 0->20, funcyional atom 1-21

# # 5d = number of samples
# model = Sequential()
# # I think 21 should be nrows
# # 5th mini batch size
# model.add(Conv3D(21, kernel_size=(5, 5, 5, 5), strides=(1, 1, 1, 1),
#                  activation='relu',
#                  input_shape=(Grid_X_size,Grid_Y_size,Grid_Z_size,21)))
# model.add(MaxPooling3D(pool_size=(2, 2, 2, 2), strides=(1, 1, 1, 1)))
# model.add(Conv3D(21, (3, 3, 3), activation='relu')) # Need to fix this
# model.add(MaxPooling3D(pool_size=(2, 2, 2))) 
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dense(20, activation='softmax'))
# # we have 20 classify
# #loss cross entopy!, 100 epochs

# model.fit(tensor, y_train,
#           batch_size=10,
#           epochs=10,
#           verbose=1,
#           validation_data=(x_test, y_test),
#           callbacks=[history])