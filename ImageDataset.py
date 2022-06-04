from torch.utils.data import Dataset
import torch

class ImageDataset(Dataset):
    def __init__(self, imagesarray , labelarray):
        self.images = torch.tensor(imagesarray)
        self.labels = torch.tensor(labelarray)
        self.data_len = len(imagesarray)
        
    def __getitem__(self, index):
        img = self.images[index]
        #img = img.reshape(-1,32*32)
        label = self.labels[index]
        return (img, label)

    def __len__(self):
        return self.data_len