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
from csv import writer







def acc_calc(output,target):
    acc = 0
    output = output.detach().numpy()
    target = target.numpy()

    if(np.argmax(output) == (target)):
        acc = 1

    return acc





#Test Information 
Test_Folder = './Results/'
Dataset_Name = "CIFAR10"
Test_Version = "Full10Test4convlayers"
numepochs = 50

path_to_file = Test_Folder + Dataset_Name + '_' + Test_Version + '_epochs_' + str(numepochs) + '.csv'
header = ['Train/Test','Epoch','Mean Loss','Accuracy']
with open(path_to_file,'w',newline='') as f:
    writer_obj = writer(f)
    writer_obj.writerow(header)
    f.close()


memflow = []
print('Virtual Memory at Start')
print(psutil.virtual_memory())
memflow.append(psutil.virtual_memory())


if(not(os.path.exists('./CIFAR10.pickle'))):
    print("Pickle file does not exist")

    #CIFAR10 Set
    airplane = imageloader.loadimageset(name='airplane', dataset = 'CIFAR10')
    automobile = imageloader.loadimageset(name='automobile', dataset = 'CIFAR10')
    bird = imageloader.loadimageset(name='bird', dataset = 'CIFAR10')
    cat = imageloader.loadimageset(name='cat', dataset = 'CIFAR10')
    deer = imageloader.loadimageset(name='deer', dataset = 'CIFAR10')
    dog = imageloader.loadimageset(name='dog', dataset = 'CIFAR10')
    frog = imageloader.loadimageset(name='frog', dataset = 'CIFAR10')
    horse = imageloader.loadimageset(name='horse', dataset = 'CIFAR10')
    ship = imageloader.loadimageset(name='ship', dataset = 'CIFAR10')
    truck = imageloader.loadimageset(name='truck', dataset = 'CIFAR10')


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
    with open('CIFAR10.pickle', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
# Load data (deserialize)
    with open('CIFAR10.pickle', 'rb') as handle:
        dataset = pickle.load(handle)



    dataloader = DataLoader(dataset,shuffle = True ,batch_size=10)

    model = Net()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer,gamma=0.1,step_size=5)


    for i in range(numepochs):
        dataloader = DataLoader(dataset,shuffle = True ,batch_size=10)
        for x, y in (dataloader):
            x = x.unsqueeze(1) 
            x = x.float()     
            y = y.long()
            #y = y.float()
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(x)
            loss = loss_fn(outputs, (y))
            loss.backward()
            optimizer.step()
        scheduler.step()   



        dataloader = DataLoader(dataset,shuffle = True ,batch_size=1)


        #Calculate Final Accuracy Training Set
        lo1 = []
        acc = []


        for x, y in (dataloader):
            x = x.unsqueeze(1) 
            x = x.float()     
            y = y.long()
        
            output = model(x)

            lo1.append(loss_fn(output, y).item())
            acc.append(acc_calc(output,y))



        print("Training Results")
        print("Number of Epochs: " + str(i))
        print('Avg Loss: ' + str(mean(lo1)))
        print('Accuracy: ' + str(mean(acc)))


        rowinfo = ['Training',str(i),str(mean(lo1)),str(mean(acc))]

        with open(path_to_file,'a',newline='') as f:
            writer_obj = writer(f)
            writer_obj.writerow(rowinfo)
            f.close()


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


    dataloader = DataLoader(dataset,shuffle = True ,batch_size=1)


    #Calculate Final Accuracy Training Set
    lo1 = []
    acc = []


    for x, y in (dataloader):
        x = x.unsqueeze(1) 
        x = x.float()     
        y = y.long()
    
        output = model(x)

        lo1.append(loss_fn(output, (y)).item())
        acc.append(acc_calc(output,y))



    print("Test Results")
    print("Number of Epochs: " + str(numepochs))
    print('Avg Loss: ' + str(mean(lo1)))
    print('Accuracy: ' + str(mean(acc)))


    rowinfo = ['Test',str(numepochs),str(mean(lo1)),str(mean(acc))]

    with open(path_to_file,'a',newline='') as f:
        writer_obj = writer(f)
        writer_obj.writerow(rowinfo)
        f.close()




