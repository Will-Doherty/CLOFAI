import torch.nn as nn
import torch
import torch.optim as optim
from dataclasses import dataclass
from CLOFAI.training.training_setup import parse_arguments, set_seeds
from CLOFAI.data.data_processing import load_all_task_data
from CLOFAI.model.model_def import ModifiedEfficientNet
from CLOFAI.utils.results_logging import save_results
from CLOFAI.training.training_fns import train_on_all_tasks
from datasets import load_dataset

@dataclass
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 123
    tasks = ["task1", "task2", "task3", "task4", "task5"]
    args = parse_arguments("EWC")
    dataset_dict = load_dataset("willd98/CLOFAI")
    weights_path = args.weights_and_results_path
    results_path = args.weights_and_results_path
    samples_replayed = args.samples_replayed
    lambda_param = args.lambda_param

@dataclass
class TrainingHyperparameters:
    model: nn.Module
    batch_size = 128
    num_epochs = 1
    learning_rate = 0.0001
    weight_decay = 0.0
    criterion = nn.BCELoss()

    def __post_init__(self):
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

cfg = Config()
set_seeds(cfg.SEED)

model = ModifiedEfficientNet()
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
model.to(cfg.device)

params = TrainingHyperparameters(model)
train_data, test_data = load_all_task_data(cfg.dataset_dict, cfg.tasks)
end_time, accuracy_matrix = train_on_all_tasks(cfg, params, "EWC")
save_results(cfg, accuracy_matrix, end_time, "EWC")
