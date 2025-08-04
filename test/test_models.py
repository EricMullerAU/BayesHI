import unittest
from unittest.mock import patch
from bayeshi import load_model
from torch import randn

# Test the shape handling of all the models, using a simple dummy input tensor
class TestModels(unittest.TestCase):
    def setUp(self):
        pass
        
    def test_blstm_dimension_handling(self):
        model = load_model('BLSTMSequenceToSequence')
        test_input = randn(16, 1, 256)  # Batch size of 10, sequence length of 256, 1 feature

        output = model(test_input)
        self.assertEqual(output.shape, (16, 256))  # Assuming the output shape is the same as input
        
        test_loader = [(test_input, randn(16, 256))]  # Dummy target tensor

        prediction = model.predict(test_loader, n_predictions=1)
        self.assertEqual(prediction.shape, (16, 256))  # Assuming the prediction shape is the same as input\
    
    def test_blstm_training(self):
        model = load_model('BLSTMSequenceToSequence')
        train_loader = [(randn(16, 256), randn(16, 256))]
        val_loader = [(randn(16, 256), randn(16, 256))]
        
        # Check that the model trains and prints a line that it is training
        with patch('builtins.print') as mock_print:
            model.fit(train_loader, val_loader, checkpoint_path=None, n_epochs=1, learning_rate=0.001)
            # The expected output of the training is something like the following:
            # Initial learning rate: [0.001]
            # Epoch [1/8], Train Loss: 0.0020, Validation Loss: 0.0018, took 75.61s
            mock_print.assert_any_call('Initial learning rate: [0.001]')
            assert any('Epoch [1/1], Train Loss: ' in str(call) for call in mock_print.call_args_list)
    
if __name__ == '__main__':
    unittest.main()