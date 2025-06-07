"""
Module providing a configuration dictionary for various neural network models and a utility function to instantiate them with specified parameters.
Attributes:
    MODEL_CONFIGS (dict): A dictionary mapping model names to their corresponding classes and default initialization parameters.
Functions:
    load_model(model_name, **kwargs):
        Instantiates and returns a model based on the provided model name and optional parameter overrides.
        Args:
            model_name (str): The key identifying the model to load from MODEL_CONFIGS.
            **kwargs: Additional keyword arguments to override or supplement the default model parameters.
        Returns:
            An instance of the specified model class, initialized with the combined parameters.
        Raises:
            ValueError: If the provided model_name does not exist in MODEL_CONFIGS.
"""

import copy
from .models import SauryModel, BayesHIModel, TPCNetAllPhases, RNNModel, LSTMSequencePredictor, TransformerWithAttentionAggregation, SimpleCNN, SimpleBNN, LSTMSequenceToSequence

MODEL_CONFIGS = {
    'saury': {
        'class': SauryModel,
        'params': {
            'cnnBlocks': 1,
            'kernelNumber': 12,
            'kernelWidth': 51,
            'MHANumber': 4,
            'transformerNumber': 1,
            'priorMu': 0.0,
            'priorSigma': 0.1,
            'posEncType': 'off'
        }
    },
    'bayeshi': {
        'class': BayesHIModel,
        'params': {
            'cnnBlocks': 1,
            'kernelNumber': 8,
            'kernelWidth1': 31,
            'kernelWidth2': 3,
            'kernelMult': 2,
            'pooling': 'off',
            'MHANumber': 4,
            'transformerNumber': 1,
            'priorMu': 0.0,
            'priorSigma': 0.1,
            'posEncType': 'sinusoidal'
        }
    },
    'tpcnet_all_phases': {
        'class': TPCNetAllPhases,
        'params':{
            'num_output': 4,
            'in_channels': 1,
            'input_row': 1,
            'input_column': 256
        }
    },
    'rnn_model': {
        'class': RNNModel,
        'params': {
            'input_dim': 1,
            'seq_len': 256,
            'd_model': 128,
            'nhead': 4,
            'num_layers': 4,
            'output_dim': 4
        }
    },
    'LSTMSequencePredictor': {
        'class': LSTMSequencePredictor,
        'params': {}
    },
    'TransformerWithAttentionAggregation': {
        'class': TransformerWithAttentionAggregation,
        'params': {}
    },
    'simpleCNN': {
        'class': SimpleCNN,
        'params': {}
    },
    'simpleBNN': {
        'class': SimpleBNN,
        'params': {}
    },
    'LSTMSequenceToSequence': {
        'class': LSTMSequenceToSequence,
        'params': {}
    }
}

def load_model(model_name, **kwargs):
    """
    Loads and initializes a model based on the specified model name.
    Parameters:
        model_name (str): The name of the model to load. Must be a key in the MODEL_CONFIGS dictionary.
        **kwargs: Additional keyword arguments to override or supplement the default model parameters.
    Returns:
        object: An instance of the specified model class, initialized with the combined parameters.
    Raises:
        ValueError: If the specified model_name is not found in MODEL_CONFIGS.
    Notes:
        - The function does not modify the original MODEL_CONFIGS dictionary.
        - The model is initialized with parameters from MODEL_CONFIGS[model_name]['params'], updated with any provided kwargs.
    """

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    model_class = config['class']
    # Update model parameters with any additional kwargs provided but don't change the actual dictionary
    params = copy.deepcopy(config['params'])
    params.update(kwargs)

    # model = model_class(**params, device='cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class(**params)

    return model
