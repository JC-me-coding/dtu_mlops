import torch
import numpy as np

def mnist():
    train0 = np.load('data/corruptmnist/train_0.npz')
    train1 = np.load('data/corruptmnist/train_1.npz')
    train2 = np.load('data/corruptmnist/train_2.npz')
    train3 = np.load('data/corruptmnist/train_3.npz')
    train4 = np.load('data/corruptmnist/train_4.npz')
    train_img_stamp = np.concatenate((train0['images'],train1['images'],train2['images'],train3['images'],train4['images']))

    mean_px = train_img_stamp.mean().astype(np.float32)
    std_px = train_img_stamp.std().astype(np.float32)
    train_img_stamp = (train_img_stamp - mean_px)/(std_px)

    train_imgs = torch.from_numpy(train_img_stamp)
    train_label_stamp = np.concatenate((train0['labels'],train1['labels'],train2['labels'],train3['labels'],train4['labels']))
    train_labels = torch.from_numpy(train_label_stamp)
    loadtest = np.load('data/corruptmnist/test.npz')
    test_img_stamp = loadtest['images']
    test_imgs = torch.from_numpy(test_img_stamp)

    mean_px = test_img_stamp.mean().astype(np.float32)
    std_px = test_img_stamp.std().astype(np.float32)
    test_img_stamp = (test_img_stamp - mean_px)/(std_px)
    
    tslabelstmp = loadtest['labels']
    tslabels = torch.from_numpy(tslabelstmp)
    train = [[train_imgs[i],train_labels[i]] for i in range(len(train_imgs))]
    test = [[test_imgs[i],tslabels[i]] for i in range(len(test_imgs))]
    return train, test