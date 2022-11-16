from matplotlib import pyplot as plt
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os import listdir
from os.path import isfile, join
import os
import re

class DrivingDataset(Dataset):
    
    def __init__(self, root_dir, categorical = False, classes=-1, transform=None):
        """
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = [f for f in listdir(self.root_dir) if f.endswith('jpg')]
        self.categorical = categorical
        self.classes = classes
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        basename = self.filenames[idx]
        img_name = os.path.join(self.root_dir, basename)
        image = io.imread(img_name)

        m = re.search('expert_([0-9]+)_([0-9]+)_([-+]?\d*\.\d+|\d+).jpg', basename)
        steering_command = np.array(float(m.group(3)), dtype=np.float32)

        if self.categorical:
            steering_command = int(((steering_command + 1.0)/2.0) * (self.classes - 1)) 
            
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'cmd': steering_command}
        

def get_sample_weights(dataset: DrivingDataset):
    """
        The method generated sampling weighted so that the straight steering and turning classes have equal representation
    """
    re_pattern = 'expert_([0-9]+)_([0-9]+)_([-+]?\d*\.\d+|\d+).jpg'
    steering_command = torch.tensor([float(re.search(re_pattern, basename).group(3)) for basename in dataset.filenames])
    straight_count = torch.sum((steering_command >= -0.052632) & (steering_command < 0.157894))
    turn_count = len(dataset) - straight_count
    weights = torch.ones_like(steering_command) #* straight_count
    # The range corresponds to the steering class 9-11 if 20 steering classes are used
    # Making straight turns half as likely. (Not using more weights because we already have weighted loss)
    weights[(steering_command >= -0.052632) & (steering_command < 0.157894)] = 0.5 #turn_count
    return weights


def plot_histogram(dataset: DrivingDataset, args):
    """Probably Unnecessary"""
    re_pattern = 'expert_([0-9]+)_([0-9]+)_([-+]?\d*\.\d+|\d+).jpg'
    steering_command = torch.tensor([float(re.search(re_pattern, basename).group(3)) for basename in dataset.filenames])
    counts, bins = np.histogram(steering_command.detach().numpy(), bins=args.n_steering_classes)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(bins[:-1], bins, weights=counts)
    fig.savefig(f"./results/{args.folder_name}/dataset_iterations.png")
    plt.close(fig)
