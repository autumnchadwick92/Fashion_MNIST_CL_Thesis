import os
import cv2
from ImageDataset import ImageDataset

class imageloader:

    @classmethod
    def loadimageset(cls,name,dataset):
    
        images = []
        for filename in os.listdir(r'./images/' + dataset + '/' + name + '/'):
            if(not(filename.startswith('.'))):    
                if(os.path.exists(r'./images/' + dataset + '/' + name+ '/'+filename)):
                    
                    #print('./images/' + dataset + '/' + name+ '/'+filename)
                    
                    img = cv2.imread(os.path.join((r'./images/' + dataset + '/' + name+ '/'),filename))

                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    #cv2.imshow('test', img)
                    img = cv2.resize(img,(28,28))


                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                        break

                    images.append(img)


        cv2.destroyAllWindows()

        return images


