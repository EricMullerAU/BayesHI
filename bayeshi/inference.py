import torch
from .models import saury_model, bayeshi_model, tpcnet_all_phases

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
    }
}

def load_model(model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_CONFIGS.keys())}")

    
    config = MODEL_CONFIGS[model_name]
    model_class = config['class']
    model = model_class(**config['params'], device='cuda' if torch.cuda.is_available() else 'cpu')

    return model