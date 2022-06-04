import psutil
from imageloader import imageloader
import numpy as np
from numpy import genfromtxt
from ImageDataset import ImageDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from neural_net import Net
import torch.optim as optim
from statistics import mean
from torch.optim.lr_scheduler import StepLR
import pickle
import os


numclasses = 1


for j in range(1,11):
    numclasses = j
    print(numclasses)

    path = './F-MNISTSC_' + str(numclasses) + '.pickle'
    filename = 'F-MNISTSC_' + str(numclasses) + '.pickle'
    if(not(os.path.exists(path))):
        print("Pickle file does not exist")

        #F-MNIST Set
        if(numclasses == 1):
            airplane = imageloader.loadimageset(name='bag', dataset = 'F-MNIST/train')
        if(numclasses == 2):
            automobile = imageloader.loadimageset(name='boot', dataset = 'F-MNIST/train')
        if(numclasses == 3):
            bird = imageloader.loadimageset(name='coat', dataset = 'F-MNIST/train')
        if(numclasses == 4):        
            cat = imageloader.loadimageset(name='dress', dataset = 'F-MNIST/train')
        if(numclasses == 5):        
            deer = imageloader.loadimageset(name='pullover', dataset = 'F-MNIST/train')
        if(numclasses == 6):        
            dog = imageloader.loadimageset(name='sandal', dataset = 'F-MNIST/train')
        if(numclasses == 7):           
            frog = imageloader.loadimageset(name='shirt', dataset = 'F-MNIST/train')
        if(numclasses == 8):        
            horse = imageloader.loadimageset(name='sneaker', dataset = 'F-MNIST/train')
        if(numclasses == 9):        
            ship = imageloader.loadimageset(name='trouser', dataset = 'F-MNIST/train')
        if(numclasses == 10):        
            truck = imageloader.loadimageset(name='tshirt', dataset = 'F-MNIST/train')

        data = []
        #Dataset
        if(numclasses ==  1):
            data = np.array(airplane)
        if(numclasses == 2):    
            data = np.array(automobile)
        if(numclasses ==  3):    
            data = np.array(bird)
        if(numclasses ==  4):    
            data = np.array(cat)
        if(numclasses ==  5):    
            data = np.array(deer)
        if(numclasses ==  6):    
            data = np.array(dog)
        if(numclasses ==  7):    
            data = np.array(frog)
        if(numclasses ==  8):    
            data = np.array(horse)
        if(numclasses ==  9):    
            data = np.array(ship)
        if(numclasses ==  10):    
            data = np.array(truck)



        #make labels
        labeldata = []
        if(numclasses == 1):
            for i in range(0,data.shape[0]):
                labeldata.append(0)
        if(numclasses ==  2):    
            for i in range(0,data.shape[0]):
                labeldata.append(1)
        if(numclasses ==  3):            
            for i in range(0,data.shape[0]):
                labeldata.append(2)
        if(numclasses ==  4):    
            for i in range(0,data.shape[0]):
                labeldata.append(3)
        if(numclasses == 5):    
            for i in range(0,data.shape[0]):
                labeldata.append(4)
        if(numclasses == 6):    
            for i in range(0,data.shape[0]):
                labeldata.append(5)
        if(numclasses == 7):    
            for i in range(0,data.shape[0]):
                labeldata.append(6)
        if(numclasses == 8):    
            for i in range(0,data.shape[0]):
                labeldata.append(7)
        if(numclasses == 9):    
            for i in range(0,data.shape[0]):
                labeldata.append(8)
        if(numclasses == 10):    
            for i in range(0,data.shape[0]):
                labeldata.append(9)


        #setup datasets
        dataset = ImageDataset(data,labeldata)


        #Pickle Save

        # Store data (serialize)
        with open(filename, 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


