import torch
import torch.nn as nn
import torch.optim as optim
from avalanche.benchmarks import nc_benchmark
from avalanche.training import GEM
from dataclasses import dataclass
from CLOFAI.training.training_setup import parse_arguments, set_seeds
from CLOFAI.model.model_def import ModifiedEfficientNet
from CLOFAI.utils.GEM_utils import CustomBCELoss, load_datasets, GEM_training
from CLOFAI.utils.results_logging import save_results
from datasets import load_dataset

@dataclass
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 123
    tasks = ["task1", "task2", "task3", "task4", "task5"]
    args = parse_arguments("Online Replay")
    dataset_dict = load_dataset("willd98/CLOFAI")
    weights_path = args.weights_and_results_path
    results_path = args.weights_and_results_path
    samples_replayed = args.samples_replayed

@dataclass
class TrainingHyperparameters:
    model: nn.Module
    batch_size = 128
    num_epochs = 3
    learning_rate = 0.0001
    weight_decay = 0.0
    criterion = CustomBCELoss()

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
train_datasets, test_datasets = load_datasets(cfg.dataset_dict, cfg.tasks)
benchmark = nc_benchmark(
    train_datasets,
    test_datasets,
    task_labels=False,
    n_experiences=len(cfg.tasks),
    shuffle=False,
    one_dataset_per_exp=True,
    class_ids_from_zero_in_each_exp=True
)
cl_gem = GEM(model=model, optimizer=params.optimizer, criterion=params.criterion, patterns_per_exp=cfg.samples_replayed, train_mb_size=params.batch_size,
             train_epochs=params.num_epochs, device=cfg.device, evaluator=None)

end_time, accuracy_matrix = GEM_training(benchmark, cl_gem, cfg)
save_results(cfg, accuracy_matrix, end_time, "GEM")
