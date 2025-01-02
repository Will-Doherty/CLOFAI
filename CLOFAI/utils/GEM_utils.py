import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
import os
from CLOFAI.data.data_processing import CustomImageDataset
from time import time
from torch.utils.data import DataLoader
from torchvision import transforms

class CustomBCELoss(nn.Module):
    def __init__(self):
        super(CustomBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, inputs, targets):
        inputs = inputs.squeeze()
        targets = targets.float()
        targets = torch.clamp(targets, min=0, max=1)
        return self.bce_loss(inputs, targets)
    
def load_single_image(filepath, label):
    return (filepath, label)

def load_images_from_folder(folder, label):
    image_files = [os.path.join(folder, f) for f in os.listdir(folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    load_func = partial(load_single_image, label=label)
    
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_func, image_files))
    
    return images

def load_datasets(dataset_dict, tasks):
    train_datasets = []
    test_datasets = []

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    for task in tasks:
        print(f"Loading {task}...")
        
        train_task_data = dataset_dict['train'].filter(lambda x: x['task'] == task)
        test_task_data = dataset_dict['test'].filter(lambda x: x['task'] == task)
        
        train_data = list(zip(train_task_data['image'], train_task_data['label']))
        test_data = list(zip(test_task_data['image'], test_task_data['label']))

        np.random.shuffle(train_data)
        np.random.shuffle(test_data)

        train_datasets.append(CustomImageDataset(train_data, transform=transform, label_type='long'))
        test_datasets.append(CustomImageDataset(test_data, transform=transform, label_type='long'))
        
    return train_datasets, test_datasets

def forward_pass(data_loader, strategy):
    total_correct = 0
    total_samples = 0

    strategy.model.eval()

    with torch.no_grad():
        for batch in data_loader:
            inputs, actuals = batch[0], batch[1]
            inputs, actuals = inputs.to(strategy.device), actuals.to(strategy.device)
            probs = strategy.model(inputs)
            preds = probs > 0.5
            total_correct += preds.eq(actuals.view_as(preds)).sum().item()
            total_samples += len(inputs)

    accuracy = total_correct / total_samples
    strategy.model.train()
    return accuracy

def GEM_training(benchmark, cl_gem, cfg):
    accuracy_matrix = []
    start_time = time()
    for task_idx, task in enumerate(benchmark.train_stream):
        print(f"Training on task {task_idx + 1}")
        cl_gem.train(task)

        task_accuracies = []
        for test_task_idx, test_task in enumerate(benchmark.test_stream):
            test_data_loader = DataLoader(test_task.dataset, batch_size=len(test_task.dataset), shuffle=False,
                                        worker_init_fn=lambda _: torch.manual_seed(cfg.SEED))
            task_accuracy = forward_pass(test_data_loader, cl_gem)
            print(f"Accuracy on test dataset {test_task_idx + 1}: {task_accuracy:.2%}")
            task_accuracies.append(task_accuracy)
        accuracy_matrix.append(task_accuracies)

    end_time = time() - start_time
    return end_time, accuracy_matrix