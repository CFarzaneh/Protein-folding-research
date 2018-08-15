import numpy as np
import tensorflow as tf
import random as rn
import os

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import pandas as pd
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Input, Add, BatchNormalization, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras import regularizers, optimizers
import keras
from time import time
from tqdm import tqdm
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import mmap
import math
plt.switch_backend('agg')
# For all the 1000 file, you can use each file as a batch

def load_data():
    labelPath = "/media/cameron/HDD2/new_tensor_label/"

    labellist = sorted(os.listdir(labelPath))
    aminoAcids = []
    for i in labellist:
        x = np.load(labelPath+i,'r')
        for j in x:
            aminoAcids.append(j)

    return aminoAcids

def parallel_computation(inputs):
    convs = []
    for i in range(21):
        conv = Conv3D(3,kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu', input_shape=(19,19,19,1))(inputs[i])
        convs.append(conv)
    return keras.layers.Add()(convs)

def create_model():
    
    inputs = [Input(shape=(19,19,19,1)) for _ in range(21)] #19x19x19
    adds = parallel_computation(inputs)

    conv1 = Conv3D(7,kernel_size=(3, 3, 3), strides=(1, 1, 1), padding = 'same', activation='relu', input_shape=(19,19,19,3))(adds) #3x3x3
    conv2 = Conv3D(14,kernel_size=(3, 3, 3), strides=(1, 1, 1), padding = 'same', activation='relu')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv2)

    conv3 = Conv3D(21,kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(pool1)
    conv4 = Conv3D(28,kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(conv3)

    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv4)
  
    flatten1 = Flatten()(pool3)
    dense1 = Dense(100, activation='relu')(flatten1)
    out = Dense(20, activation='softmax')(dense1)
    model = Model(input= inputs,output = out)

    #model.summary()
    
    #This is used to load the model
    import os.path
    if os.path.isfile('my_model.h5') == True:
        model = load_model('my_model.h5')
        print("Model loaded")
        model.summary()
    
    #Run it
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

    return model


def train_and_evaluate_model(model, trainGen, testGen, trainAmount, testAmount):

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
   
    model.fit_generator(generator=trainGen,
        steps_per_epoch=math.ceil(trainAmount/10),
        epochs=8,
        verbose=1,
        validation_data=testGen,
        validation_steps=math.ceil(testAmount/10))
    
    # Save the model
    model.save('my_model.h5')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def myAwesomeDataGenerator(indicies,batch_size=10):
    dataPath = "/media/cameron/HDD2/new_tensor_data/"
    labelPath = "/media/cameron/HDD2/new_tensor_label/"
    filelist = sorted(os.listdir(dataPath))
    lengthOfFiles = len(filelist)

    while True:
        i = 0
        k = 0
        
        lastFile = False
        samples = []
        labels = []
       
        sizes = []

        for v,file in enumerate(filelist):
            loadedFile = np.load(dataPath+file,'r')
            fileName = file.split('_')[0]
            numSamples = get_num_lines('/media/cameron/HDD2/2.label/'+fileName+'.txt')
            #print(fileName,'num of samples:',numSamples)
            sizes.append(numSamples)
            loadedLabels = np.load(labelPath+fileName+"_label.npy",'r')
        
            if v == lengthOfFiles-1:
                lastFile = True

            for l,tensor in enumerate(loadedFile):
                if i in indicies.tolist():
                    samples.append(tensor)
                    labels.append(loadedLabels[l])
                    k+=1
                i+=1

                if k == batch_size or (l == numSamples-1 and k != 0 and lastFile == True):
                    samples = np.stack(samples, axis=0)
                    labels = np.stack(labels, axis=0)
                
                    classes=['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR']
                    #classes=['PHE','TYR']
                    protein_label_encoder = pd.factorize(classes)
                    encoder = OneHotEncoder()
                    protein_labels_1hot = encoder.fit_transform(protein_label_encoder[0].reshape(-1,1))
                    onehot_array = protein_labels_1hot.toarray()

                    d1 = dict(zip(classes,onehot_array.tolist()))
                    theFinalLabels = []
                    for aminoacid in labels:
                        encoding = d1[aminoacid]
                        theFinalLabels.append(encoding)

                    nicelabels = np.array(theFinalLabels)
                    nicelabels = nicelabels.reshape((-1,len(classes)))
                    #Labels Done.

                    data = samples.reshape((-1, 21, 19, 19, 19, 1))

                    # For parallel 21 computations, Create 21 list to insert the model
                    data2 = [[] for _ in range(21)]
                    for sample in data:
                        for ind,val in enumerate(sample):
                            data2[ind].append(val)
                    data2 = [np.array(i) for i in data2]

                    #print(nicelabels)
                    yield data2, nicelabels

                    k = 0
                    samples = []
                    labels = []

if __name__ == "__main__":
   
    aminoAcids = load_data()
    n_samples = len(aminoAcids)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1382)

    for train_index, test_index in skf.split(X=np.zeros(n_samples),y=aminoAcids):
        '''
        #This is for pairwise testing!!
        theNewTrain = []
        theNewTest = []
        for l,amino in enumerate(aminoAcids):
            if amino == 'TYR' or amino == 'PHE':
                if l in train_index:
                    theNewTrain.append(l)
                else:
                    theNewTest.append(l)
          
        train_index = np.array(theNewTrain)
        test_index  = np.array(theNewTest)
        n_samples = len(theNewTrain) + len(theNewTest)
        ''' 

        trainGenerator = myAwesomeDataGenerator(train_index)
        validationGenerator = myAwesomeDataGenerator(test_index[:100])
        validationGenerator1 = myAwesomeDataGenerator(test_index[:100])
        validationGenerator2 = myAwesomeDataGenerator(test_index[:100])
        
        model = None
        model = create_model()

        np.set_printoptions(threshold=np.nan)
        print('Here are the weights of the softmax layer:\n',model.layers[-1].get_weights()[0])
       
        #print('Running pairwise test on TYR and PHE')
        #print('Total samples',n_samples,'training with',n_samples-len(test_index.tolist()), 'validating with',len(test_index.tolist()))
        #train_and_evaluate_model(model,trainGenerator,validationGenerator,n_samples-len(test_index.tolist()),len(test_index.tolist()))

        ########################### 
        # Confusion matrix stuff: #
        ###########################

        layer_name = 'flatten_1'
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        predictions1 = intermediate_layer_model.predict_generator(validationGenerator,steps=math.ceil(len(test_index[:100].tolist())/10),verbose=0)

        layer_name = 'dense_1'
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        predictions2 = intermediate_layer_model.predict_generator(validationGenerator1,steps=math.ceil(len(test_index[:100].tolist())/10),verbose=0)

        predictions = model.predict_generator(validationGenerator2,steps=math.ceil(len(test_index[:100].tolist())/10),verbose=0)

        classes=['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR']
        predictionss = predictions.argmax(axis=-1)

        print('Output from the layers')

        for o,aminoA in enumerate(test_index[:100].tolist()):
            print('True label:',aminoAcids[aminoA],'Predicted label:',classes[predictionss[o]])
            print(predictions1[o].tolist())
            print(predictions2[o].tolist())
            print(predictions[o].tolist())

        predictions = predictions.argmax(axis=-1)
        trueOutputs = validationLabels.argmax(axis=-1)

        cm = confusion_matrix(trueOutputs,predictions)
        classes=['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR']
        plot_confusion_matrix(cm,classes)

        plt.savefig('confusionMatrix.png')

        break
