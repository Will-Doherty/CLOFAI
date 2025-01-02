import torch
from torch.utils.data import ConcatDataset, random_split, Dataset
from torchvision import datasets, transforms
import os
from PIL import Image
from datasets import Dataset
import numpy as np

def load_single_task_data(dataset_dict, data_type, task_name):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    task_data = dataset_dict[data_type].filter(lambda x: x['task'] == task_name)
    data_pairs = list(zip(task_data['image'], task_data['label']))
    return CustomImageDataset(data_pairs, transform=transform)

def load_all_task_data(dataset_dict, tasks):
    train_data = ConcatDataset([load_single_task_data(dataset_dict, 'train', task) for task in tasks])
    test_data = ConcatDataset([load_single_task_data(dataset_dict, 'test', task) for task in tasks])
    return train_data, test_data

def load_cumulative_task_data(data_path, task_names, data_type='train', shuffle=True):
    datasets = [load_single_task_data(data_path, data_type, task) for task in task_names]
    combined_dataset = ConcatDataset(datasets)
    if shuffle:
        total_length = len(combined_dataset)
        shuffled_dataset, _ = random_split(combined_dataset, [total_length, 0], generator=torch.Generator().manual_seed(42))
        return shuffled_dataset
    else:
        return combined_dataset
    
class ReplayDataset(Dataset):
    def __init__(self, replay_samples):
        self.replay_samples = replay_samples

    def __len__(self):
        return len(self.replay_samples)

    def __getitem__(self, idx):
        return self.replay_samples[idx]
    
class CustomImageDataset:
    def __init__(self, data, transform=None, label_type='float'):
        self.images = []
        self.targets = []
        
        for img, label in data:
            self.images.append(img)
            self.targets.append(int(label) if label_type == 'long' else label)
            
        dtype = torch.long if label_type == 'long' else torch.float
        self.targets = torch.tensor(self.targets, dtype=dtype)
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.targets[idx]
        image = Image.fromarray(np.array(image))
        
        if self.transform:
            image = self.transform(image)
            
        return image, label