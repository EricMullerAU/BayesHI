import unittest
from unittest.mock import patch
from bayeshi import load_model
from torch import randn

# Test the shape handling of all the models, using a simple dummy input tensor
class TestModels(unittest.TestCase):
    def setUp(self):
        pass
        
    def test_dimension_handling(self):
        model = load_model('BLSTMSequenceToSequence')
        test_input = randn(16, 1, 256)  # Batch size of 10, sequence length of 256, 1 feature

        output = model(test_input)
        self.assertEqual(output.shape, (16, 256))  # Assuming the output shape is the same as input
        
        test_loader = [(test_input, randn(16, 256))]  # Dummy target tensor

        prediction = model.predict(test_loader, numPredictions=1)
        self.assertEqual(prediction.shape, (16, 256))  # Assuming the prediction shape is the same as input