import argparse
import sys
import os
import torch
import random

class ArgumentParser:
    def __init__(self, CL_method):
        self.CL_method = CL_method
        self.parser = argparse.ArgumentParser(description='Train and evaluate model')
        self.default_samples_replayed = 100
        self.default_EWC_param = 100000.0
        self.method_configs = {
            "Online Replay": self._setup_replay_arguments,
            "GEM": self._setup_replay_arguments,
            "EWC": self._setup_EWC_arguments
        }

    def _add_basic_arguments(self):
        """Add basic arguments that are common to all methods."""
        self.parser.add_argument('--weights_and_results_path', type=str, 
                               help='Path to store model weights and results')
        
    def _validate_basic_arguments(self, args):
        if args.weights_and_results_path is None:
            print("Error: --weights_and_results_path argument is required") 
            self.parser.print_help()
            sys.exit(1)

        if not os.path.exists(args.weights_and_results_path):
            print(f"Error: Path for weights and results ({args.weights_and_results_path}) does not exist")
            sys.exit(1)

    def _setup_replay_arguments(self):
        self.parser.add_argument('--samples_replayed', type=int, default=self.default_samples_replayed,
                               help='Size of replay buffer')
        
    def _setup_EWC_arguments(self):
        self.parser.add_argument('--samples_replayed', type=int, default=self.default_samples_replayed,
                               help='Size of replay buffer')
        self.parser.add_argument('--lambda_param', type=float, default=self.default_EWC_param,
                               help='Lambda parameter for EWC')

    def parse(self):
        self._add_basic_arguments()

        if self.CL_method in self.method_configs:
            self.method_configs[self.CL_method]()
        
        args = self.parser.parse_args()
        self._validate_basic_arguments(args)
        
        if self.CL_method in ["Online Replay", "GEM", "EWC"] and args.samples_replayed == self.default_samples_replayed:
            print(f"The default buffer size of {self.default_samples_replayed} is being used. To change, pass an integer to the --samples_replayed argument.")

        if self.CL_method == "EWC" and args.lambda_param == self.default_EWC_param:
            print(f"The default EWC lambda parameter of {self.default_EWC_param} is being used. To change, pass a value to the --lambda_param argument.")

        return args

def parse_arguments(CL_method):
    arg_parser = ArgumentParser(CL_method)
    args = arg_parser.parse()
    return args

def set_seeds(seed=123):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)