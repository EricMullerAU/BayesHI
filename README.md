# BayesHI
Bayesian Neural Network for analysis of neutral hydrogen emission spectra to examine thermal phase structures in the ISM.

## Features
- Multiple model architectures
- Bayesian inference via Bayes by Backpropagation
- Configurable via command line or configuration file
- Weights and biases integration
- Pre-trained models and example usage

## Installation
```bash
git clone https://github.com/EricMullerAU/BayesHI.git
cd BayesHI
pip install -r requirements.txt
```

## Usage

The standard pipeline, including analysis and plotting, can be run completely from the main script.
```bash
python main.py --architecture <model_architecture> --data_path <path_to_data> --output_path <output_directory>
```


### Training
To train a model, you can use the provided training script. You can specify the configuration file or use command line arguments to override specific parameters.
```bash
python scripts/train.py --config configs/model_config.yaml
```

### Inference
```bash
python scripts/predict.py --model_path path/to/model --data_path path/to/data
```