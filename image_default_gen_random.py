import numpy as np
#from PIL import Image
import cv2
from PIL import Image

import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import random


print(os.getcwd())
folderlist = ['bag','boot','coat','dress','pullover','sandal','shirt','sneaker','trouser','tshirt']

for terms in range (0,10):

  #path = 'images/F-MNIST/train/bag'
  path = 'images/F-MNIST/train/' + folderlist[terms]
  
  print(path)


  # this list holds all the image filename
  ds = []
  dsnames = []

  # creates a ScandirIterator aliased as files
  with os.scandir(path) as files:
    # loops through each file in the directory
      for file in files:
          if file.name.endswith('.png'):
            # adds only the image files to the ds list
              ds.append(np.asarray(cv2.cvtColor(cv2.imread((path+'/'+file.name)),cv2.COLOR_BGR2GRAY)).flatten())
              dsnames.append(str(file.name))



  converted_data = np.array(ds)
  print(converted_data.shape)
  #Creating Clusters
  k = 100
  clusters = KMeans(k, random_state = None)
  clusters.fit(converted_data)

  image_cluster = pd.DataFrame(dsnames,columns=['image'])
  image_cluster["clusterid"] = clusters.labels_

  print(image_cluster)

  #Combine sets together per cluster
  currentds = []


  
  avgfolderlist = ['random_bag','random_boot','random_coat','random_dress','random_pullover','random_sandal','random_shirt','random_sneaker','random_trouser','random_tshirt']

  avgpath = 'images/F-MNIST/train/' + avgfolderlist[terms]


  for i in range(0,k):
      currentset = (image_cluster.loc[image_cluster['clusterid']==i])


      filename = np.random.choice(currentset['image'])
      
      mat = np.asarray(cv2.cvtColor(cv2.imread((path+'/'+filename)),cv2.COLOR_BGR2GRAY))


      #show final image              
      mat = mat.astype(np.uint8)
      img = Image.fromarray(mat, 'L')
      #img.show()
      img.save(avgpath+'/'+str(i)+'.png',format = "png")
      
        


