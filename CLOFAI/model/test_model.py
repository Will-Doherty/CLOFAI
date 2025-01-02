import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from CLOFAI.data.data_processing import CustomImageDataset
from torchvision import transforms
import os
import torchvision.models as models
import argparse
import sys
from datasets import load_dataset
from CLOFAI.model.model_def import ModifiedEfficientNet

parser = argparse.ArgumentParser()
# parser.add_argument('--use_pretrained', action='store_true', help='If set, use pretrained classifier weights. Otherwise, initialize weights randomly.')
parser.add_argument('--results_path', type=str, help='Path to store results')
parser.add_argument('--weights_file', type=str, help='Path to the stored weights file')

args = parser.parse_args()

if args.results_path is None:
    print("Error: --results_path argument is required")
    parser.print_help()
    sys.exit(1)

if args.weights_file is None:
    print("Error: --weights_file argument is required")
    parser.print_help()
    sys.exit(1)

if not os.path.exists(args.results_path):
    print(f"Error: Chosen path to store results ({args.results_path}) does not exist")
    sys.exit(1)

# if not os.path.exists(args.weights_file):
#     print(f"Error: Chosen model weights not found in {args.weights_file}")
#     sys.exit(1)

dataset_dict = load_dataset("willd98/CLOFAI")
results_path = args.results_path
weights_file = args.weights_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 123
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

tasks = ["task1", "task2", "task3", "task4", "task5"]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_task_data(task_name, data_type='train'):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    task_data = dataset_dict[data_type].filter(lambda x: x['task'] == task_name)
    data_pairs = list(zip(task_data['image'], task_data['label']))
    return CustomImageDataset(data_pairs, transform=transform)

test_data = ConcatDataset([load_task_data(task, 'test') for task in tasks])

original_model = models.efficientnet_b0(weights=None)
num_features = original_model.classifier[1].in_features
model = ModifiedEfficientNet(original_model, num_features)
model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
model.to(device)

criterion = nn.BCELoss()
batch_size = 128

def test_model(task_name):
    model.eval()
    test_data = load_task_data(task_name, 'test')
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

accuracies = []
for i in range(len(tasks)):
    acc = test_model(tasks[i])
    accuracies.append(acc)

# save accuracy to a text file
def extract_weights_name(path):
    last_slash = path.rindex('/')
    last_dot = path.rindex('.')
    return path[last_slash + 1:last_dot]

results_save_path = f'{results_path}test accuracy {extract_weights_name(weights_file)}.txt'
with open(results_save_path, 'w') as f:
    f.write("Accuracy on test dataset:\n")
    for i, acc in enumerate(accuracies, 1):
        f.write(f"task {i} accuracy = {acc:.2%}\n")

print(f"Results written to {results_path}")