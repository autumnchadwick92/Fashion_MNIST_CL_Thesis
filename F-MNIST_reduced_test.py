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


memflow = []
print('Virtual Memory at Start')
print(psutil.virtual_memory())
memflow.append(psutil.virtual_memory())


if(not(os.path.exists('./CIFAR10REDUCED.pickle'))):
    print("Pickle file does not exist")

    #CIFAR10 Set
    airplane = imageloader.loadimageset(name='avg_airplane', dataset = 'CIFAR10')
    automobile = imageloader.loadimageset(name='avg_automobile', dataset = 'CIFAR10')
    bird = imageloader.loadimageset(name='avg_bird', dataset = 'CIFAR10')
    cat = imageloader.loadimageset(name='avg_cat', dataset = 'CIFAR10')
    deer = imageloader.loadimageset(name='avg_deer', dataset = 'CIFAR10')
    dog = imageloader.loadimageset(name='avg_dog', dataset = 'CIFAR10')
    frog = imageloader.loadimageset(name='avg_frog', dataset = 'CIFAR10')
    horse = imageloader.loadimageset(name='avg_horse', dataset = 'CIFAR10')
    ship = imageloader.loadimageset(name='avg_ship', dataset = 'CIFAR10')
    truck = imageloader.loadimageset(name='avg_truck', dataset = 'CIFAR10')


    #Dataset
    airplane = np.array(airplane)
    automobile = np.array(automobile)
    bird = np.array(bird)
    cat = np.array(cat)
    deer = np.array(deer)
    dog = np.array(dog)
    frog = np.array(frog)
    horse = np.array(horse)
    ship = np.array(ship)
    truck = np.array(truck)


    data = np.concatenate((airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck))


    #make labels
    labeldata = []
    for i in range(0,airplane.shape[0]):
        labeldata.append(0)
    for i in range(0,automobile.shape[0]):
        labeldata.append(1)
    for i in range(0,bird.shape[0]):
        labeldata.append(2)
    for i in range(0,cat.shape[0]):
        labeldata.append(3)
    for i in range(0,deer.shape[0]):
        labeldata.append(4)
    for i in range(0,dog.shape[0]):
        labeldata.append(5)
    for i in range(0,frog.shape[0]):
        labeldata.append(6)
    for i in range(0,horse.shape[0]):
        labeldata.append(7)
    for i in range(0,ship.shape[0]):
        labeldata.append(8)
    for i in range(0,truck.shape[0]):
        labeldata.append(9)


    #setup datasets
    dataset = ImageDataset(data,labeldata)


    #Pickle Save

    # Store data (serialize)
    with open('CIFAR10REDUCED.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
# Load data (deserialize)
    with open('CIFAR10REDUCED.pickle', 'rb') as handle:
        dataset = pickle.load(handle)


print(dataset.__getitem__(1))

dataloader = DataLoader(dataset,shuffle = True ,batch_size=10)

model = Net()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer,gamma=0.1,step_size=10)

numepochs = 2

for i in range(numepochs):

    for x, y in (dataloader):
        x = x.unsqueeze(1) 
        x = x.float()     
        #y = y.long()
        y = y.float()
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(x)
        loss = loss_fn(outputs, (y))
        loss.backward()
        optimizer.step()

        print(loss)
    scheduler.step()   




#Calculate Final Accuracy Training Set
lo1 = []


for x, y in (dataloader):
    x = x.unsqueeze(1) 
    x = x.float()     
    y = y.long()
 
    output = model(x)

    lo1.append(loss_fn(output, (y)).item())


print("Full Model")
print(mean(lo1))


if(not(os.path.exists('./CIFAR10TEST.pickle'))):
    print("Pickle file does not exist")

    #Load Validation Set

    #CIFAR10TEST Set
    airplane = imageloader.loadimageset(name='airplane', dataset = 'CIFAR10TEST')
    automobile = imageloader.loadimageset(name='automobile', dataset = 'CIFAR10TEST')
    bird = imageloader.loadimageset(name='bird', dataset = 'CIFAR10TEST')
    cat = imageloader.loadimageset(name='cat', dataset = 'CIFAR10TEST')
    deer = imageloader.loadimageset(name='deer', dataset = 'CIFAR10TEST')
    dog = imageloader.loadimageset(name='dog', dataset = 'CIFAR10TEST')
    frog = imageloader.loadimageset(name='frog', dataset = 'CIFAR10TEST')
    horse = imageloader.loadimageset(name='horse', dataset = 'CIFAR10TEST')
    ship = imageloader.loadimageset(name='ship', dataset = 'CIFAR10TEST')
    truck = imageloader.loadimageset(name='truck', dataset = 'CIFAR10TEST')

    #Dataset
    airplane = np.array(airplane)
    automobile = np.array(automobile)
    bird = np.array(bird)
    cat = np.array(cat)
    deer = np.array(deer)
    dog = np.array(dog)
    frog = np.array(frog)
    horse = np.array(horse)
    ship = np.array(ship)
    truck = np.array(truck)


    data = np.concatenate((airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck))


    #make labels
    labeldata = []
    for i in range(0,airplane.shape[0]):
        labeldata.append(0)
    for i in range(0,automobile.shape[0]):
        labeldata.append(1)
    for i in range(0,bird.shape[0]):
        labeldata.append(2)
    for i in range(0,cat.shape[0]):
        labeldata.append(3)
    for i in range(0,deer.shape[0]):
        labeldata.append(4)
    for i in range(0,dog.shape[0]):
        labeldata.append(5)
    for i in range(0,frog.shape[0]):
        labeldata.append(6)
    for i in range(0,horse.shape[0]):
        labeldata.append(7)
    for i in range(0,ship.shape[0]):
        labeldata.append(8)
    for i in range(0,truck.shape[0]):
        labeldata.append(9)


    #setup datasets
    dataset = ImageDataset(data,labeldata)


    #Pickle Save

    # Store data (serialize)
    with open('CIFAR10TEST.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
# Load data (deserialize)
    with open('CIFAR10TEST.pickle', 'rb') as handle:
        dataset = pickle.load(handle)


dataloader = DataLoader(dataset,shuffle = True ,batch_size=10)






#Calculate Final Accuracy Validation Set
lo1 = []


for x, y in (dataloader):
    x = x.unsqueeze(1) 
    x = x.float()     
    y = y.long()
 
    output = model(x)

    lo1.append(loss_fn(output, (y)).item())


print("Full Model")
print(mean(lo1))





