from dataclasses import dataclass
import argparse
from typing import Type


# Base configuration class (shared parameters for all models)
@dataclass
class BaseConfig:
    base_path: str
    output_path: str
    n_epochs: int = 30
    learning_rate: float = 0.001
    scheduler_step: int = 15
    stopper_patience: int = 10
    stopper_tol: float = 1e-4
    max_kl_weight: float = 1.0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'BaseConfig':
        return cls(**vars(args))


# Configuration for TPCNet (a specific architecture)
@dataclass
class TPCNetConfig(BaseConfig):
    conv_layers: int = 2
    kernel_number: int = 16
    kernel_width1: int = 41
    kernel_width2: int = 21
    kernel_multiplier: float = 1.2
    pooling_type: str = 'max'


# Configuration for ratioNN (another architecture)
@dataclass
class RatioNNConfig(BaseConfig):
    mha_number: int = 8
    transformer_number: int = 4
    pos_enc_type: str = 'sinusoidal'


# Function to select the correct configuration based on architecture name
def get_config(architecture: str) -> Type[BaseConfig]:
    if architecture == 'TPCNet':
        return TPCNetConfig
    elif architecture == 'ratioNN':
        return RatioNNConfig
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# Function to parse the arguments and return the config object
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single model training with specific hyperparameters.")

    # Required Arguments
    parser.add_argument('--base_path', type=str, required=True, help='Base path for data, checkpoints, etc.')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for the results.')

    # General Hyperparameters
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--scheduler_step', type=int, default=15, help='Step size for the learning rate scheduler.')
    parser.add_argument('--stopper_patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--stopper_tol', type=float, default=1e-4, help='Tolerance for early stopping.')
    parser.add_argument('--max_kl_weight', type=float, default=1.0, help='Maximum weight for the KL divergence loss.')

    # Optional Model Hyperparameters
    parser.add_argument('--conv_layers', type=int, default=None, help='Number of convolutional layers.')
    parser.add_argument('--kernel_number', type=int, default=None, help='Number of kernels in the convolutional layer.')
    parser.add_argument('--kernel_width1', type=int, default=None, help='Width of the first kernels in the convolutional layer.')
    parser.add_argument('--kernel_width2', type=int, default=None, help='Width of the second kernels in the convolutional layer.')
    parser.add_argument('--kernel_multiplier', type=float, default=None, help='Multiplier for the number of kernels in each block.')
    parser.add_argument('--pooling_type', type=str, default=None, help='Type of pooling to use (off, max, avg).')
    parser.add_argument('--mha_number', type=int, default=None, help='Number of heads in the Multi-Head Attention layer.')
    parser.add_argument('--transformer_number', type=int, default=None, help='Number of layers in the Transformer Encoder.')
    parser.add_argument('--pos_enc_type', type=str, default='off', help='Type of positional encoding to use (off, sinusoidal, stochastic, united).')

    # Architecture selection argument
    parser.add_argument('--architecture', type=str, required=True, choices=['TPCNet', 'ratioNN'], help='Model architecture.')

    return parser.parse_args()


# Class to handle configuration management
class ConfigManager:
    @staticmethod
    def get_config_and_arguments():
        args = parse_args()
        config_class = get_config(args.architecture)
        config = config_class.from_args(args)
        return config
