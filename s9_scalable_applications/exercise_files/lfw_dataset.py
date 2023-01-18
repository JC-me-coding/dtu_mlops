"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
import glob
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'



class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        '''# TODO: fill out with what you need   r'C:/Users/juca/Downloads/lfw/*/* '''
        self.transform = transform
        self.images = []

        for f in glob.iglob(path_to_folder + '//*//*'):
            keep = copy.deepcopy(Image.open(f))
            self.images.append(keep)
        
    def __len__(self):
        return len(self.images) #None # TODO: fill out
    
    def __getitem__(self, index: int) -> torch.Tensor:
        '''# TODO: fill out'''
        self.img = self.images[index]

        return self.transform(self.img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='', type=str)
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-num_workers', default=0, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    if args.visualize_batch:
        '''# TODO: visualize a batch of images'''
        transf = transforms.ToPILImage()

        def show(imgs):
            if not isinstance(imgs, list):
                imgs = [imgs]
            fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
            for i, img in enumerate(imgs):
                img = img.detach()
                img = transf(img)
                #img = F.to_pil_image(img)
                axs[0, i].imshow(np.asarray(img))
                axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            
            plt.show()
        img_list = [dataset.__getitem__(0)]
        grid=make_grid(img_list)
        show(grid)
        print(dataset.__len__())

        pass
        
        
        #img = transf(dataset.__getitem__(0))
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print('Timing: {np.mean(res)}+-{np.std(res)}')
