from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

class CassavaDataset(Dataset):
    def __init__(self,image_dir,mask_dir,transform=None):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.transform=transform
        self.images=os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        img_path=os.path.join(self.image_dir,self.images[index])
        mask_path=os.path.join(self.image_dir,self.images[index].replace(".jpg", "_mask.gif"))
        image=np.array(Image.open(img_path).convert("RGB"))
        mask=np.array(Image.open(mask_path).convert("L"),dtype=np.float32)

        return image,mask

