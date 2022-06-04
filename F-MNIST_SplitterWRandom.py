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


for i in range(1,11):
    numclasses = i
    print(numclasses)
    path = './F-MNISTSCR_' + str(numclasses) + '.pickle'
    filename = 'F-MNISTSCR_' + str(numclasses) + '.pickle'
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


 
        #CIFAR10random Set
        if(numclasses > 1):
            random_airplane = imageloader.loadimageset(name='random_bag', dataset = 'F-MNIST/train')
            random_airplane = np.array(random_airplane)
        if(numclasses > 2):
            random_automobile = imageloader.loadimageset(name='random_boot', dataset = 'F-MNIST/train')
            random_automobile = np.array(random_automobile)
        if(numclasses > 3):
            random_bird = imageloader.loadimageset(name='random_coat', dataset = 'F-MNIST/train')
            random_bird = np.array(random_bird)
        if(numclasses > 4):        
            random_cat = imageloader.loadimageset(name='random_dress', dataset = 'F-MNIST/train')
            random_cat = np.array(random_cat)
        if(numclasses > 5):        
            random_deer = imageloader.loadimageset(name='random_pullover', dataset = 'F-MNIST/train')
            random_deer = np.array(random_deer)
        if(numclasses > 6):        
            random_dog = imageloader.loadimageset(name='random_sandal', dataset = 'F-MNIST/train')
            random_dog = np.array(random_dog)
        if(numclasses > 7):           
            random_frog = imageloader.loadimageset(name='random_shirt', dataset = 'F-MNIST/train')
            random_frog = np.array(random_frog)
        if(numclasses > 8):        
            random_horse = imageloader.loadimageset(name='random_sneaker', dataset = 'F-MNIST/train')
            random_horse = np.array(random_horse)
        if(numclasses > 9):        
            random_ship = imageloader.loadimageset(name='random_trouser', dataset = 'F-MNIST/train')
            random_ship = np.array(random_ship)



        #Dataset
        if(numclasses == 1):
            airplane = np.array(airplane)
            data = airplane
        if(numclasses == 2):    
            automobile = np.array(automobile)
            data = np.concatenate((random_airplane, automobile))
        if(numclasses == 3):    
            bird = np.array(bird)
            data = np.concatenate((random_airplane, random_automobile, bird))
        if(numclasses == 4):    
            cat = np.array(cat)
            data = np.concatenate((random_airplane, random_automobile, random_bird, cat))
        if(numclasses == 5):    
            deer = np.array(deer)
            data = np.concatenate((random_airplane, random_automobile, random_bird, random_cat, deer))
        if(numclasses == 6):    
            dog = np.array(dog)
            data = np.concatenate((random_airplane, random_automobile, random_bird, random_cat, random_deer, dog))
        if(numclasses == 7):    
            frog = np.array(frog)
            data = np.concatenate((random_airplane, random_automobile, random_bird, random_cat, random_deer, random_dog, frog))
        if(numclasses == 8):    
            horse = np.array(horse)
            data = np.concatenate((random_airplane, random_automobile, random_bird, random_cat, random_deer, random_dog, random_frog, horse))
        if(numclasses == 9):    
            ship = np.array(ship)
            data = np.concatenate((random_airplane, random_automobile, random_bird, random_cat, random_deer, random_dog, random_frog, random_horse, ship))
        if(numclasses == 10):    
            truck = np.array(truck)
            data = np.concatenate((random_airplane, random_automobile, random_bird, random_cat, random_deer, random_dog, random_frog, random_horse, random_ship, truck))



        #make labels
        labeldata = []


        #random items
        if(numclasses > 1):
            for i in range(0,random_airplane.shape[0]):
                labeldata.append(0)
        if(numclasses > 2):    
            for i in range(0,random_automobile.shape[0]):
                labeldata.append(1)
        if(numclasses > 3):            
            for i in range(0,random_bird.shape[0]):
                labeldata.append(2)
        if(numclasses > 4):    
            for i in range(0,random_cat.shape[0]):
                labeldata.append(3)
        if(numclasses > 5):    
            for i in range(0,random_deer.shape[0]):
                labeldata.append(4)
        if(numclasses > 6):    
            for i in range(0,random_dog.shape[0]):
                labeldata.append(5)
        if(numclasses > 7):    
            for i in range(0,random_frog.shape[0]):
                labeldata.append(6)
        if(numclasses > 8):    
            for i in range(0,random_horse.shape[0]):
                labeldata.append(7)
        if(numclasses > 9):    
            for i in range(0,random_ship.shape[0]):
                labeldata.append(8)





        #current single class
        if(numclasses == 1):
            for i in range(0,airplane.shape[0]):
                labeldata.append(0)
        if(numclasses == 2):    
            for i in range(0,automobile.shape[0]):
                labeldata.append(1)
        if(numclasses == 3):            
            for i in range(0,bird.shape[0]):
                labeldata.append(2)
        if(numclasses == 4):    
            for i in range(0,cat.shape[0]):
                labeldata.append(3)
        if(numclasses == 5):    
            for i in range(0,deer.shape[0]):
                labeldata.append(4)
        if(numclasses == 6):    
            for i in range(0,dog.shape[0]):
                labeldata.append(5)
        if(numclasses == 7):    
            for i in range(0,frog.shape[0]):
                labeldata.append(6)
        if(numclasses == 8):    
            for i in range(0,horse.shape[0]):
                labeldata.append(7)
        if(numclasses == 9):    
            for i in range(0,ship.shape[0]):
                labeldata.append(8)
        if(numclasses == 10):    
            for i in range(0,truck.shape[0]):
                labeldata.append(9)



        #setup datasets
        dataset = ImageDataset(data,labeldata)


        #Pickle Save

        # Store data (serialize)
        with open(filename, 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


