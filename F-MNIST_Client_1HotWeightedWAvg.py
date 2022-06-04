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



for epnum in range (1,4):
    
    #Test Information 
    Test_Folder = './Results/'
    Dataset_Name = "F-MNIST"
    numepochs = epnum
    Test_Version = "Client" + str(numepochs) + "HotWeightedWAvg_3ConvLrg"



    path_to_file = Test_Folder + Dataset_Name + '_' + Test_Version + '_epochs_' + str(numepochs) + '.csv'
    header = ['Train/Test','NumClasses','Mean Loss','Accuracy']
    with open(path_to_file,'w',newline='') as f:
        writer_obj = writer(f)
        writer_obj.writerow(header)
        f.close()



    memflow = []
    print('Virtual Memory at Start')
    print(psutil.virtual_memory())
    memflow.append(psutil.virtual_memory())

    for j in range(2,11):
        numclasses = j

        #Print line info
        plinfo = "F-MNIST, Num Class: " + str(numclasses)

        #Image Files
        path = './F-MNISTSCA_' + str(numclasses) + '.pickle'
        filename = 'F-MNISTSCA_' + str(numclasses) + '.pickle'
        #nn files
        startnnfilename = './F-MNIST_' + str(numclasses-1) + '.pt'

    # Load data (deserialize)
        with open(filename, 'rb') as handle:
            dataset = pickle.load(handle)

        print(dataset.__getitem__(1))

        dataloader = DataLoader(dataset,shuffle = True ,batch_size=10)

        model = torch.load(startnnfilename)
        modelbase = model

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.0001)
        scheduler = StepLR(optimizer,gamma=0.1,step_size=10)


        for i in range(numepochs):
        
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

            lo1.append(loss_fn(output, (y)).item())
            acc.append(acc_calc(output,y))



        print("Training Results")
        print("Number of classes currently: " + str(j))
        print("Number of Epochs: " + str(numepochs))
        print('Avg Loss: ' + str(mean(lo1)))
        print('Accuracy: ' + str(mean(acc)))


        rowinfo = ['Training',str(j),str(mean(lo1)),str(mean(acc))]



        with open(path_to_file,'a',newline='') as f:
            writer_obj = writer(f)
            writer_obj.writerow(rowinfo)
            f.close()





        path = './F-MNISTTEST_' + str(numclasses) + '.pickle'
        filename = 'F-MNISTTEST_' + str(numclasses) + '.pickle'
        if(not(os.path.exists(path))):
            print("Pickle file does not exist")

            #CIFAR10 Set
            if(numclasses >= 1):
                airplane = imageloader.loadimageset(name='airplane', dataset = 'CIFAR10TEST')
            if(numclasses >= 2):
                automobile = imageloader.loadimageset(name='automobile', dataset = 'CIFAR10TEST')
            if(numclasses >= 3):
                bird = imageloader.loadimageset(name='bird', dataset = 'CIFAR10TEST')
            if(numclasses >= 4):        
                cat = imageloader.loadimageset(name='cat', dataset = 'CIFAR10TEST')
            if(numclasses >= 5):        
                deer = imageloader.loadimageset(name='deer', dataset = 'CIFAR10TEST')
            if(numclasses >= 6):        
                dog = imageloader.loadimageset(name='dog', dataset = 'CIFAR10TEST')
            if(numclasses >= 7):           
                frog = imageloader.loadimageset(name='frog', dataset = 'CIFAR10TEST')
            if(numclasses >= 8):        
                horse = imageloader.loadimageset(name='horse', dataset = 'CIFAR10TEST')
            if(numclasses >= 9):        
                ship = imageloader.loadimageset(name='ship', dataset = 'CIFAR10TEST')
            if(numclasses >= 10):        
                truck = imageloader.loadimageset(name='truck', dataset = 'CIFAR10TEST')

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
        else:
        # Load data (deserialize)
            with open(filename, 'rb') as handle:
                dataset = pickle.load(handle)


        dataloader = DataLoader(dataset,shuffle = True ,batch_size=1)



        #Calculate Final Accuracy Validation Set
        lo1 = []
        acc = []

        #Setup weight distributions
        baseweights = np.zeros(10)
        clientweights = np.zeros(10)

        for i in range(0,10):
            #adjust averages (early weights, base is weighted more)
            if(i < (numclasses-1)):
                baseweights[i] = 0.9
                clientweights[i] = 0.1

            #current class, client weighted more
            if(i == (numclasses-1)):
                baseweights[i] = 0.1
                clientweights[i] = 0.9           


        baseweights = torch.tensor(baseweights)
        clientweights = torch.tensor(clientweights)

        for x, y in (dataloader):
            x = x.unsqueeze(1) 
            x = x.float()     
            y = y.long()
        
            output = model(x)
            output2 = modelbase(x)

            output = output*baseweights
            output2 = output2*clientweights

            comboutput = output + output2

            lo1.append(loss_fn(comboutput, (y)).item())
            acc.append(acc_calc(comboutput,y))



        print("Test Results")
        print("Number of classes currently: " + str(j))
        print("Number of Epochs: " + str(numepochs))
        print('Avg Loss: ' + str(mean(lo1)))
        print('Accuracy: ' + str(mean(acc)))


        rowinfo = ['Test',str(j),str(mean(lo1)),str(mean(acc))]

        with open(path_to_file,'a',newline='') as f:
            writer_obj = writer(f)
            writer_obj.writerow(rowinfo)
            f.close()