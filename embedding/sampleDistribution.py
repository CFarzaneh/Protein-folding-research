import os
import matplotlib.pyplot as plt

fileTest = str(input("Please input the full path to the file you want to analyze: "))
fileTest = fileTest.replace("'",'')
fileTest = fileTest.replace(' ','')

myList = []
with open(fileTest, 'r') as fp:
	for aminoacid in fp.read().splitlines():
		aminoacid = aminoacid.strip().split("\t")
		aminoacid = aminoacid.pop(0)
		myList.append(aminoacid)
myDict = {k:myList.count(k) for k in set(myList)}
print(myDict)

plt.bar(myDict.keys(), myDict.values())
plt.show()
