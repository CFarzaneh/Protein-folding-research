import os
from tqdm import tqdm
from shutil import copyfile
import subprocess
import sys

dataPath = "/media/cameron/HDD2/tensor_data/"
labelPath = "/media/cameron/HDD2/tensor_label/"

filelist = os.listdir(dataPath)

numFiles = 1000 

for i,file in enumerate(tqdm(filelist, total=numFiles)):
	saveDataPath= "/home/cameron/Desktop/tensor_data/"
	savelabelPath = "/home/cameron/Desktop/tensor_label/"

	copyfile(dataPath+file,saveDataPath+file)
	fileName = file.split('_')[0]
	copyfile(labelPath+fileName+'_label.npy',savelabelPath+fileName+'_label.npy')

	f = open("output/"+fileName+"out.txt", "w")
	subprocess.call(["python3", "embedding_demo.py"], stdout=f)

	os.remove(saveDataPath+file)
	os.remove(savelabelPath+fileName+'_label.npy')

	if i == numFiles-1:
		break
