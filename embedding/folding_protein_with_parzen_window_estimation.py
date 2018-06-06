from tqdm import tqdm
import numpy as np
import copy
import mmap
import os

# Hyperparameters
window_width = 2
grid_cell_size = 1
grid_X_size = 19
grid_Y_size = 19
grid_Z_size = 19
coord_X_min = -9
coord_Y_min = -9
coord_Z_min = -9

def protein_grid(grid_X_size,grid_Y_size,grid_Z_size,coord_X_min,coord_Y_min,coord_Z_min):

	# All the possible x,y,z coordinates for x,y,z axis in protein grid.
	x_possible = [(x*grid_cell_size)+coord_X_min for x in range(grid_X_size)]
	y_possible = [(y*grid_cell_size)+coord_Y_min for y in range(grid_Y_size)]
	z_possible = [(z*grid_cell_size)+coord_Z_min for z in range(grid_Z_size)]

	# Extract all the possible 3 coordinates for 3D protein grid.
	protein_coord_grid = [[x,y,z]for x in x_possible for y in y_possible for z in z_possible]

	# Reshape to extract 4D structure.
	protein_coord_grid = np.array(protein_coord_grid).reshape((grid_X_size,grid_Y_size,grid_Z_size,3))

	# Returning 4D protein coordinate grid with shape (18,18,18,3).
	return protein_coord_grid

def zero_dict_channel_21(grid_X_size,grid_Y_size,grid_Z_size):
	# Create an empty dictionary.
	tensor_dict_channel_21 = {}

	# Initialize all the 21 functional atoms by 3D zero grid of shape (18,18,18).
	for i in range(1,22):
		tensor_dict_channel_21[i] = np.zeros((grid_X_size,grid_Y_size,grid_Z_size))

	# Returning functional atoms dictionary with 3D zero grid (18,18,18) as value.
	# It's a 4D strucure of shape (21,18,18,18).
	return tensor_dict_channel_21

def change_dict_tensor(protein_dict,protein_grid_4D,zero_dict_4D,window_width):
	# Input Parameters:
	# protein_dict contains all the atoms present in protein with their x,y,z coordinates.
	# protein_grid_4D (18,18,18,3) contains 4D protein coordinate grid.
	# zero_dict_4D (21,18,18,18) contains functional atoms dictionary with 3D zero grid of shape (18,18,18) as value.
	# window_width for parzen window estimation

	# Deep_copy the functional atoms dictionary with 3D zero grid of shape (18,18,18) as value.
	# It's a 4D strucure of shape (21,18,18,18), we will update this using parzen window.
	tensor_dict = copy.deepcopy(zero_dict_4D)

	# Update all the functional atoms 3D grid, which has values in the protein by adding the
	# contribution to their corresponding 3D grid.

	for atom in protein_dict:

		# Intialize 3D zero grid to update a functional atom 3D grid.
		contribution = np.zeros(tensor_dict[atom].shape)

		# Check all the occurences of the functional atom to update the functional atom 3D grid.
		for coordinates in protein_dict[atom]:

			# Parzen window implementation.
			distance_sq = np.sum(np.square(protein_grid_4D - coordinates),axis=3)
			contribution += np.exp(-distance_sq/(2*(window_width**2)))

		# Update the functional atom 3D grid.
		tensor_dict[atom] += contribution

	# Create an empty tensor
	tensor = []

	# Add the 3D grid of all the 21 functional atoms
	for i in range(1,22):
		tensor.append(tensor_dict[i])

	# Returning 4D numpy array of tensor with shape (21,18,18,18)
	return np.array(tensor)

def get_num_lines(file_path):
	fp = open(file_path, "r+")
	buf = mmap.mmap(fp.fileno(), 0)
	lines = 0
	while buf.readline():
		lines += 1
	return lines

def file_data(file_path,file_name,protein_grid_3D,zero_dict_4D,window_width,data_path,label_path):
	# Input parameters:
	# file_name contains name of the text file.
	# protein_grid_3D,zero_dict_4D and window_width  are used when call change_dict_tensor function for each protein.

	# Create empty lists for protein labels and tensors(21,18,18,18) present in the file.
	protein_label = []
	protein_data = []

	# Read text file
	with open(file_path + file_name + ".txt","r") as fp:

		# Each line in the file represents a protein.
		for line in tqdm(fp.read().splitlines(), total=get_num_lines(file_path + file_name + ".txt")):

			# Create an empty dictionary
			protein_dict = {}
			line = line.strip().split("\t")

			# Store Label in the protein_label list
			protein_label += [line.pop(0)]

			# Update the protein_dict dictionary by adding (x,y,z) coordinates belonging to functional atoms.
			for atom in line:
				atom = atom.split(",")
				protein_dict[int(atom[0])] = protein_dict.get(int(atom[0]),[]) + [[float(cord) for cord in atom[1:]]]

			# Store tensor in the protein_data list
			protein_data.append(change_dict_tensor(protein_dict,protein_grid_3D,zero_dict_4D,window_width))

	# Storing the data and label files in numpy format extracting from the text file
	#return np.array(protein_data),np.array(protein_label)

	np.save(data_path+ file_name + "_data.npy", np.array(protein_data))
	np.save(label_path+ file_name + "_label.npy", np.array(protein_label))

# Extracting a dictionary to extract the 3D zero matrix for all the 21 atoms
zero_channel21_4D_val = zero_dict_channel_21(grid_X_size,grid_Y_size,grid_Z_size)

# Extracting a 4D protein coordinate grid with shape (18,18,18,3)
protein_grid_3D_val = protein_grid(grid_X_size,grid_Y_size,grid_Z_size,coord_X_min,coord_Y_min,coord_Z_min)

# Path of text files and storing tensors/labels
#path = "/home/atharva/Desktop/2.coor/"
path = "/home/femi/Desktop/2.coor/"
path_tensor = "/home/femi/Desktop/tensor_data/"
path_label = "/home/femi/Desktop/tensor_label/"

# List of all the files (10163 files)
file_list = os.listdir(path)
count = 0

#ignore first 5, last 5!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#  I have converted 1000 files into 5D structure ( 5D numpy array) and able to extract
# 460,733 samples of shape (21,18,18,18)
for txt_file in tqdm(file_list[:1000]):
	if txt_file.endswith(".txt"):
		txt_file = txt_file.split('.')[0]
		# Change txt file into numpy array
		file_data(path,txt_file,protein_grid_3D_val,zero_channel21_4D_val,window_width,path_tensor,path_label)
	count +=1
