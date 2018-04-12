import os
import numpy as np

fin = open("pdb1a1x.txt", 'r')

# Read in the data
for line in fin:
	line = line.strip().split('\t')
	protein = line.pop(0)
	print(protein)

	arr = np.array([])
	for i in line:
		j = i.split(',')
		arr = np.concatenate([arr, j])

	arr = np.reshape(arr,(-1,4))
	print(arr)
	break #For now we're going to read only the first protein

# Hyperparameters
Window_width = 2
Grid_cell_size = 1
Grid_X_size = 21
Grid_Y_size = 21
Grid_Z_size = 21
Coord_X_min = -10
Coord_Y_min = -10
Coord_Z_min = -10

tensor = np.zeros((Grid_X_size,Grid_Y_size,Grid_Z_size,21))

for x in range(0,Grid_X_size):
	Xx = (x*Grid_cell_size)+Coord_X_min
	for y in range(0,Grid_Y_size):
		Yy = (y*Grid_Y_size)+Coord_Y_min
		for z in range(0,Grid_Z_size):
			Zz = (z*Grid_cell_size)+Coord_Z_min
			for m in range(0,21):
				#Channel = 
				tensor[x][y][z][m] = 1