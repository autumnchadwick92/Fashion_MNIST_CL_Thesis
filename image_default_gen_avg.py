import numpy as np
#from PIL import Image
import cv2
from PIL import Image

import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd

print(os.getcwd())


folderlist = ['bag','boot','coat','dress','pullover','sandal','shirt','sneaker','trouser','tshirt']

for terms in range (1,10):

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
  
  avgfolderlist = ['avg_bag','avg_boot','avg_coat','avg_dress','avg_pullover','avg_sandal','avg_shirt','avg_sneaker','avg_trouser','avg_tshirt']

  avgpath = 'images/F-MNIST/train/' + avgfolderlist[terms]
  #avgpath = 'images/F-MNIST/train/avg_bag'

  for i in range(0,k):
      currentset = (image_cluster.loc[image_cluster['clusterid']==i])
      for filename in currentset['image']:
          currentds.append(np.asarray(cv2.cvtColor(cv2.imread((path+'/'+filename)),cv2.COLOR_BGR2GRAY)))


      #making average image
      avgimage = np.zeros((28,28))
      count = 0
      for image in currentds:

          
        count = count + 1
        for j in range(0,image.shape[0]):
              for g in range(0,image.shape[1]):
                avgimage[j,g] += image[j,g]



      #show final image              
      mat = (avgimage/count)
      mat = mat.astype(np.uint8)
      img = Image.fromarray(mat, 'L')
      #img.show()
      img.save(avgpath+'/'+str(i)+'.png',format = "png")
      

          


