import copy
from typing import overload, Literal
import torch
from .models import saury_model, bayeshi_model, tpcnet_all_phases, rnn_model, LSTMSequencePredictor, TransformerWithAttentionAggregation, SimpleCNN, SimpleBNN, LSTMSequenceToSequence

MODEL_CONFIGS = {
    'saury': {
        'class': saury_model,
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
        'class': bayeshi_model,
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
        'class': tpcnet_all_phases,
        'params':{
            'num_output': 4,
            'in_channels': 1,
            'input_row': 1,
            'input_column': 256
        }
    },
    'rnn_model': {
        'class': rnn_model,
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

# # Use overloads to provide tooltips for the load_model function
# @overload
# def load_model(model_name: Literal['saury'], **kwargs) -> saury_model: ...
# @overload
# def load_model(model_name: Literal['bayeshi'], **kwargs) -> bayeshi_model: ...
# @overload
# def load_model(model_name: Literal['tpcnet_all_phases'], **kwargs) -> tpcnet_all_phases: ...
# @overload
# def load_model(model_name: Literal['rnn_model'], **kwargs) -> rnn_model: ...
# @overload
# def load_model(model_name: Literal['LSTMSequencePredictor'], **kwargs) -> LSTMSequencePredictor: ...
# @overload
# def load_model(model_name: Literal['TransformerWithAttentionAggregation'], **kwargs) -> TransformerWithAttentionAggregation: ...

def load_model(model_name, **kwargs):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_CONFIGS.keys())}")

    
    config = MODEL_CONFIGS[model_name]
    model_class = config['class']
    # Update model parameters with any additional kwargs provided but don't change the actual dictionary
    params = copy.deepcopy(config['params'])
    params.update(kwargs)
    
    model = model_class(**params, device='cuda' if torch.cuda.is_available() else 'cpu')

    return model