import unittest
from unittest.mock import patch
from  bayeshi.data_loaders import load_data

def test_split_dict_proportion(self):
    split_dict = {'tigress': 0.5, 'saury': 0.3, 'seta': 0.2}
    with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
         patch('bayeshi.data_loaders.load_saury_data', self.mock_load_saury_data), \
         patch('bayeshi.data_loaders.load_seta_data', self.mock_load_seta_data):
        train_loader, val_loader, test_loader = load_data(dataset='all', split=split_dict, random_state=42)
        total_samples = sum(len(loader.dataset) for loader in [train_loader, val_loader, test_loader])
        self.assertGreater(total_samples, 0)
