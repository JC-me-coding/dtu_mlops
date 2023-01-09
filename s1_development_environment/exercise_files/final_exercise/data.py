import torch
import numpy as np

def mnist():
    root_folder = r'C:/git/dtu_mlops/s1_development_environment/exercise_files/final_exercise/'
    
    tr0 = np.load(root_folder + 'data/corruptmnist/train_0.npz')
    tr1 = np.load(root_folder + 'data/corruptmnist/train_1.npz')
    tr2 = np.load(root_folder + 'data/corruptmnist/train_2.npz')
    tr3 = np.load(root_folder + 'data/corruptmnist/train_3.npz')
    tr4 = np.load(root_folder + 'data/corruptmnist/train_4.npz')
    
    train_imgs = np.concatenate((tr0['images'],tr1['images'],tr2['images'],tr3['images'],tr4['images']))
    train_imgs = torch.from_numpy(train_imgs)
    
    train_labels = np.concatenate((tr0['labels'],tr1['labels'],tr2['labels'],tr3['labels'],tr4['labels']))
    train_labels = torch.from_numpy(train_labels)
    
    test_set = np.load('data/corruptmnist/test.npz')
    test_imgs = test_set['images']
    test_imgs = torch.from_numpy(test_imgs)
        
    test_labels = test_set['labels']
    test_labels = torch.from_numpy(test_labels)
    train = [[train_imgs[i],train_labels[i]] for i in range(len(train_imgs))]
    test = [[test_imgs[i],test_labels[i]] for i in range(len(test_imgs))]
    
    return train, test