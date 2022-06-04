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


for numclasses in range(1,11):
	print(numclasses)
	path = './F-MNIST_' + str(numclasses) + '.pickle'
	filename = 'F-MNIST_' + str(numclasses) + '.pickle'
	if(not(os.path.exists(path))):
			
		print("Pickle file does not exist")
		print("Current Class: " + str(numclasses))


		#F-MNIST Set
		if(numclasses >= 1):
			airplane = imageloader.loadimageset(name='bag', dataset = 'F-MNIST/train')
		if(numclasses >= 2):
			automobile = imageloader.loadimageset(name='boot', dataset = 'F-MNIST/train')
		if(numclasses >= 3):
			bird = imageloader.loadimageset(name='coat', dataset = 'F-MNIST/train')
		if(numclasses >= 4):        
			cat = imageloader.loadimageset(name='dress', dataset = 'F-MNIST/train')
		if(numclasses >= 5):        
			deer = imageloader.loadimageset(name='pullover', dataset = 'F-MNIST/train')
		if(numclasses >= 6):        
			dog = imageloader.loadimageset(name='sandal', dataset = 'F-MNIST/train')
		if(numclasses >= 7):           
			frog = imageloader.loadimageset(name='shirt', dataset = 'F-MNIST/train')
		if(numclasses >= 8):        
			horse = imageloader.loadimageset(name='sneaker', dataset = 'F-MNIST/train')
		if(numclasses >= 9):        
			ship = imageloader.loadimageset(name='trouser', dataset = 'F-MNIST/train')
		if(numclasses >= 10):        
			truck = imageloader.loadimageset(name='tshirt', dataset = 'F-MNIST/train')

		data = []
		#Dataset
		if(numclasses >= 1):
			airplane = np.array(airplane)
			data = airplane
			#data = np.concatenate((airplane))
		if(numclasses >= 2):    
			automobile = np.array(automobile)
			data = np.concatenate((airplane, automobile))
		if(numclasses >= 3):    
			bird = np.array(bird)
			data = np.concatenate((airplane, automobile, bird))
		if(numclasses >= 4):    
			cat = np.array(cat)
			data = np.concatenate((airplane, automobile, bird, cat))
		if(numclasses >= 5):    
			deer = np.array(deer)
			data = np.concatenate((airplane, automobile, bird, cat, deer))
		if(numclasses >= 6):    
			dog = np.array(dog)
			data = np.concatenate((airplane, automobile, bird, cat, deer, dog))
		if(numclasses >= 7):    
			frog = np.array(frog)
			data = np.concatenate((airplane, automobile, bird, cat, deer, dog, frog))
		if(numclasses >= 8):    
			horse = np.array(horse)
			data = np.concatenate((airplane, automobile, bird, cat, deer, dog, frog, horse))
		if(numclasses >= 9):    
			ship = np.array(ship)
			data = np.concatenate((airplane, automobile, bird, cat, deer, dog, frog, horse, ship))
		if(numclasses >= 10):    
			truck = np.array(truck)
			data = np.concatenate((airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck))



		#make labels
		labeldata = []
		if(numclasses >= 1):
			for i in range(0,airplane.shape[0]):
				labeldata.append(0)
		if(numclasses >= 2):    
			for i in range(0,automobile.shape[0]):
				labeldata.append(1)
		if(numclasses >= 3):            
			for i in range(0,bird.shape[0]):
				labeldata.append(2)
		if(numclasses >= 4):    
			for i in range(0,cat.shape[0]):
				labeldata.append(3)
		if(numclasses >= 5):    
			for i in range(0,deer.shape[0]):
				labeldata.append(4)
		if(numclasses >= 6):    
			for i in range(0,dog.shape[0]):
				labeldata.append(5)
		if(numclasses >= 7):    
			for i in range(0,frog.shape[0]):
				labeldata.append(6)
		if(numclasses >= 8):    
			for i in range(0,horse.shape[0]):
				labeldata.append(7)
		if(numclasses >= 9):    
			for i in range(0,ship.shape[0]):
				labeldata.append(8)
		if(numclasses >= 10):    
			for i in range(0,truck.shape[0]):
				labeldata.append(9)


		#setup datasets
		dataset = ImageDataset(data,labeldata)


		#Pickle Save

		# Store data (serialize)
		with open(filename, 'wb') as handle:
			pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)