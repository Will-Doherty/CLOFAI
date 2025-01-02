import torch
from torch.utils.data import DataLoader
from CLOFAI.data.data_processing import load_single_task_data, load_cumulative_task_data
from time import time
import random
from torch.utils.data import ConcatDataset
from CLOFAI.utils.EWC_utils import ewc_loss
from CLOFAI.data.data_processing import CustomImageDataset

def test_model(task_name, model, dataset_dict, batch_size, SEED, device):
    model.eval()
    test_data = load_single_task_data(dataset_dict, 'test', task_name)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, worker_init_fn=lambda _: torch.manual_seed(SEED))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy on {task_name}: {accuracy:.2%}')
    return accuracy

def training_loop(cfg, params, train_data, optimal_weights=None):
    params.model.train()
    train_loader = DataLoader(
        train_data, 
        batch_size=params.batch_size, 
        shuffle=True, 
        worker_init_fn=lambda _: torch.manual_seed(cfg.SEED)
    )

    for epoch in range(params.num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(cfg.device), data[1].to(cfg.device)
            params.optimizer.zero_grad()
            outputs = params.model(inputs)
            loss = params.criterion(outputs, labels.float().unsqueeze(1))
            
            if hasattr(cfg, 'lambda_param') and cfg.lambda_param > 0 and optimal_weights is not None: # for EWC only
                ewc_loss_fn = ewc_loss(cfg.lambda_param, params.model, inputs, cfg.samples_replayed, optimal_weights, cfg)
                ewc_loss_value = ewc_loss_fn(params.model)
                total_loss = loss + ewc_loss_value
            else:
                total_loss = loss

            total_loss.backward()
            params.optimizer.step()
            
            running_loss += total_loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()

            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.6f}, accuracy: {100 * correct / total:.2f}%')
            running_loss = 0.0
            correct = 0
            total = 0

def train_on_single_task(task_name, cfg, params):
    print(f"Training on {task_name}")
    train_data = load_single_task_data(cfg.dataset_dict, 'train', task_name)
    training_loop(cfg, params, train_data)

def train_on_cumulative_tasks(tasks_to_train_on, cfg, params):
    print(f"Training on tasks: {tasks_to_train_on}")
    combined_train_data = load_cumulative_task_data(cfg.dataset_dict, tasks_to_train_on, 'train')
    training_loop(cfg, params, combined_train_data)

def train_on_all_tasks(cfg, params, CL_method, num_replay_samples=100):
    accuracy_matrix = []
    start_time = time()
    optimal_weights_per_task = [] # just for EWC

    for i in range(len(cfg.tasks)):
        current_task = cfg.tasks[i]
        
        if CL_method == 'naive':
            train_on_single_task(current_task, cfg, params)
        
        elif CL_method == 'cumulative':
            tasks_to_train_on = cfg.tasks[:i + 1]
            train_on_cumulative_tasks(tasks_to_train_on, cfg, params)
        
        elif CL_method == 'replay':
            replayed_tasks = cfg.tasks[:i] if i > 0 else []
            train_on_task_online_replay(
                current_task,
                replayed_tasks,
                cfg,
                params,
                num_replay_samples
            )

        elif CL_method == "EWC":
            previous_weights = optimal_weights_per_task[-1] if optimal_weights_per_task else None
            train_on_task_EWC(current_task, cfg, params, previous_weights)
            current_optimal_weights = [p.clone().detach() for p in params.model.parameters() if p.requires_grad]
            optimal_weights_per_task.append(current_optimal_weights)
        
        else:
            raise ValueError(f"Unknown CL_method: {CL_method}")

        accuracies = []
        for task in cfg.tasks:
            acc = test_model(task, params.model, cfg.dataset_dict, 
                            params.batch_size, cfg.SEED, cfg.device)
            accuracies.append(acc)
        
        accuracy_matrix.append(accuracies)

    torch.save(params.model.state_dict(), 
                f'{cfg.weights_path}/{CL_method}_classifier.pth')

    end_time = time() - start_time
    return end_time, accuracy_matrix

def get_replay_samples(task_data, num_samples):
    indices = random.sample(range(len(task_data)), num_samples)
    replay_samples = torch.utils.data.Subset(task_data, indices)
    return replay_samples

def train_on_task_online_replay(task_name, replayed_tasks, cfg, params, num_replay_samples):
    print(f"Training on {task_name} with replay from {replayed_tasks}")
    current_task_data = load_single_task_data(cfg.dataset_dict, 'train', task_name)
    
    if not replayed_tasks:
        return training_loop(cfg, params, current_task_data)
    
    replay_datasets = []
    for task in replayed_tasks:
        task_data = load_single_task_data(cfg.dataset_dict, 'train', task)
        replay_subset = get_replay_samples(task_data, num_replay_samples)
        replay_datasets.append(replay_subset)
    
    all_datasets = [current_task_data] + replay_datasets
    combined_data = ConcatDataset(all_datasets)
    training_loop(cfg, params, combined_data)

def train_on_task_EWC(task_name, cfg, params, optimal_weights=None):
    print(f"Training on {task_name}")
    task_data = load_single_task_data(cfg.dataset_dict, 'train', task_name)
    training_loop(cfg, params, task_data, optimal_weights)