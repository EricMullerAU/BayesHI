'''
Unit tests for the data loading functionality in bayeshi.data_loaders.
This module tests various aspects of the load_data function, including:
- Different simulation split modes (all, equal, integer, dictionary)
- Randomness and cube diversity in TIGRESS data
- Handling of different y value types (fractions, emission, absorption)
- Error handling for invalid inputs
- Ensuring reproducibility with random states
'''

import unittest
from unittest.mock import patch
import numpy as np
import warnings
from bayeshi.data_loaders import load_data

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Mock data setup
        self.n_cubes = 11
        self.samples_per_cube = 100
        self.tigress_data = np.vstack([
            np.random.rand(self.samples_per_cube, 256) + i
            for i in range(self.n_cubes)
        ])
        self.tigress_labels = np.vstack([
            np.random.rand(self.samples_per_cube, 4)
            for _ in range(self.n_cubes)
        ])
        self.saury_data = np.random.rand(80, 256)
        self.saury_labels = np.random.rand(80, 4)
        self.seta_data = np.random.rand(60, 256)
        self.seta_labels = np.random.rand(60, 4)

    def mock_load_tigress_data(self, path, sim, x_values, y_values, verbose=False):
        # Return mock data with cube IDs encoded in the first column
        return self.tigress_data, self.tigress_labels

    def mock_load_saury_data(self, path, x_values, y_values, verbose=False):
        return self.saury_data, self.saury_labels

    def mock_load_seta_data(self, path, sim, x_values, y_values, verbose=False):
        return self.seta_data, self.seta_labels

    def mock_data_for_y_values(self, samples, y_values, n_features=256):
        """Generate mock data for a given y_values option."""
        if y_values == 'fractions':
            return np.random.rand(samples, 4)
        elif y_values in ['emission', 'absorption']:
            return np.random.rand(samples, n_features)
        else:
            raise ValueError("Invalid y_values")

    # --- Split mode tests ---

    def test_split_all(self):
        """Test using all available data points."""
        with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', self.mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', self.mock_load_seta_data):

            train_loader, val_loader, test_loader = load_data(split='all')
            total_samples = sum(len(loader.dataset) for loader in [train_loader, val_loader, test_loader])
            self.assertGreater(total_samples, 0)

    def test_split_equal(self):
        """Test equal sampling from each simulation."""
        with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', self.mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', self.mock_load_seta_data):

            train_loader, val_loader, test_loader = load_data(split='equal')
            total_samples = sum(len(loader.dataset) for loader in [train_loader, val_loader, test_loader])
            self.assertGreater(total_samples, 0)

    def test_split_int_valid(self):
        """Test integer split with valid number."""
        with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', self.mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', self.mock_load_seta_data):

            split_num = 90  # 30 per simulation (30*3=90)
            train_loader, val_loader, test_loader = load_data(split=split_num)
            self.assertEqual(
                sum(len(loader.dataset) for loader in [train_loader, val_loader, test_loader]),
                split_num
            )

    def test_split_int_invalid(self):
        """Test integer split with insufficient samples."""
        with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', self.mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', self.mock_load_seta_data):

            split_num = 200  # Requires more than available in seta (which has 60)
            with self.assertRaisesRegex(ValueError, "Not enough samples in"):
                load_data(dataset='all', split=split_num)

    # --- Dictionary split tests ---

    def test_split_dict_proportion(self):
        """Test dictionary split with proportions."""
        with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', self.mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', self.mock_load_seta_data):

            split_dict = {'tigress': 0.5, 'saury': 0.3, 'seta': 0.2}
            train_loader, val_loader, test_loader = load_data(dataset='all', split=split_dict, random_state=42)
            total_samples = sum(len(loader.dataset) for loader in [train_loader, val_loader, test_loader])
            self.assertGreater(total_samples, 0)

    def test_split_dict_percent(self):
        """Test dictionary split with percentages."""
        with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', self.mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', self.mock_load_seta_data):

            split_dict = {'tigress': 50, 'saury': 30, 'seta': 20}
            train_loader, val_loader, test_loader = load_data(dataset='all', split=split_dict, random_state=42)
            total_samples = sum(len(loader.dataset) for loader in [train_loader, val_loader, test_loader])
            self.assertGreater(total_samples, 0)

    def test_split_dict_absolute(self):
        """Test dictionary split with absolute numbers (and check warning)."""
        with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', self.mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', self.mock_load_seta_data):

            split_dict = {'tigress': 50, 'saury': 30, 'seta': 25}
            with self.assertWarns(UserWarning):
                train_loader, val_loader, test_loader = load_data(dataset='all', split=split_dict, random_state=42)
            total_samples = sum(len(loader.dataset) for loader in [train_loader, val_loader, test_loader])
            self.assertEqual(total_samples, 50 + 30 + 25)

    def test_split_dict_missing_key(self):
        """Test error when dictionary is missing a key."""
        with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', self.mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', self.mock_load_seta_data):

            split_dict = {'tigress': 50, 'saury': 30}  # Missing 'seta'
            with self.assertRaisesRegex(ValueError, "split dictionary must contain 'tigress', 'saury', and 'seta' keys"):
                load_data(dataset='all', split=split_dict)

    def test_split_dict_non_numeric(self):
        """Test error when dictionary value is not a number."""
        with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', self.mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', self.mock_load_seta_data):

            split_dict = {'tigress': 50, 'saury': '30', 'seta': 20}  # 'saury' is string
            with self.assertRaisesRegex(ValueError, "split\\['saury'\\] must be a number"):
                load_data(dataset='all', split=split_dict)

    def test_split_dict_proportion_invalid(self):
        """Test error when proportion split does not sum to 1."""
        with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', self.mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', self.mock_load_seta_data):

            split_dict = {'tigress': 0.5, 'saury': 0.4, 'seta': 0.2}  # Sums to 1.1
            with self.assertRaisesRegex(ValueError, "if using proportions for data splitting, the sum of the split values must be 1.0"):
                load_data(dataset='all', split=split_dict)

    def test_split_dict_percent_invalid(self):
        """Test error when percentage split does not sum to 100."""
        with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', self.mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', self.mock_load_seta_data):

            split_dict = {'tigress': 50, 'saury': 30, 'seta': 25}  # Sums to 105
            with self.assertWarns(UserWarning):
                train_loader, val_loader, test_loader = load_data(dataset='all', split=split_dict)
            total_samples = sum(len(loader.dataset) for loader in [train_loader, val_loader, test_loader])
            self.assertEqual(total_samples, 50 + 30 + 25)

    # --- Randomness and cube diversity tests ---

    def test_tigress_cube_diversity_in_split(self):
        """Test that TIGRESS spectra in split come from different cubes."""
        with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', self.mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', self.mock_load_seta_data):

            split_num = 100
            train_loader, _, _ = load_data(dataset=['tigress'], split=split_num, random_state=42)
            selected_data = train_loader.dataset.tensors[0].numpy()
            cube_ids = np.floor(selected_data[:, 0]).astype(int)
            unique_cubes = np.unique(cube_ids)
            print(f"Unique cubes in selected TIGRESS spectra: {unique_cubes}")
            self.assertGreater(len(unique_cubes), 1,
                "All selected TIGRESS spectra came from the same cube")

    def test_tigress_cube_sampling_is_random(self):
        """Test that random sampling sometimes leaves out some TIGRESS cubes."""
        with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', self.mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', self.mock_load_seta_data):

            split_num = 30
            n_trials = 20
            unique_cubes_counts = []
            for _ in range(n_trials):
                train_loader, _, _ = load_data(
                    dataset=['tigress'],
                    split=split_num,
                    random_state=None  # Use different random state each time
                )
                selected_data = train_loader.dataset.tensors[0].numpy()
                cube_ids = np.floor(selected_data[:, 0]).astype(int)
                unique_cubes = np.unique(cube_ids)
                unique_cubes_counts.append(len(unique_cubes))
            self.assertLess(min(unique_cubes_counts), self.n_cubes,
                "Every split included spectra from all TIGRESS cubes; sampling may not be random")
            print(f"Unique cubes counts in {n_trials} trials: {unique_cubes_counts}")

    def test_randomness_reproducible(self):
        """Test that setting the random_state makes selections reproducible."""
        # Use fixed mock data for this test
        fixed_data = np.arange(100*256, dtype=float).reshape(100, 256)
        fixed_labels = np.arange(100*4, dtype=float).reshape(100, 4)
        def fixed_mock_load_tigress_data(path, sim, x_values, y_values, verbose=False):
            return fixed_data, fixed_labels
        def fixed_mock_load_saury_data(path, x_values, y_values, verbose=False):
            return fixed_data[:80], fixed_labels[:80]
        def fixed_mock_load_seta_data(path, sim, x_values, y_values, verbose=False):
            return fixed_data[:60], fixed_labels[:60]

        with patch('bayeshi.data_loaders.load_tigress_data', fixed_mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', fixed_mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', fixed_mock_load_seta_data):

            train_loader1, _, _ = load_data(split='equal', random_state=42)
            data1 = train_loader1.dataset.tensors[1].numpy()
            train_loader2, _, _ = load_data(split='equal', random_state=42)
            data2 = train_loader2.dataset.tensors[1].numpy()
            np.testing.assert_array_equal(data1, data2)

    # --- y_values tests ---

    def test_y_values_fractions(self):
        """Test that y_data shape is (n, 4) when y_values is 'fractions'."""
        def mock_load_tigress_data(path, sim, x_values, y_values, verbose=False):
            x = np.random.rand(100, 256)
            y = self.mock_data_for_y_values(100, y_values)
            return x, y
        def mock_load_saury_data(path, x_values, y_values, verbose=False):
            x = np.random.rand(80, 256)
            y = self.mock_data_for_y_values(80, y_values)
            return x, y
        def mock_load_seta_data(path, sim, x_values, y_values, verbose=False):
            x = np.random.rand(60, 256)
            y = self.mock_data_for_y_values(60, y_values)
            return x, y

        with patch('bayeshi.data_loaders.load_tigress_data', mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', mock_load_seta_data):

            train_loader, val_loader, test_loader = load_data(
                dataset='all', y_values='fractions', split='all'
            )
            self.assertEqual(train_loader.dataset.tensors[1].shape[1], 4)
            self.assertEqual(val_loader.dataset.tensors[1].shape[1], 4)
            self.assertEqual(test_loader.dataset.tensors[1].shape[1], 4)

    def test_y_values_emission_absorption(self):
        """Test that y_data shape matches x_data when y_values is 'emission' or 'absorption'."""
        for y_val in ['emission', 'absorption']:
            def mock_load_tigress_data(path, sim, x_values, y_values, verbose=False):
                x = np.random.rand(100, 256)
                y = self.mock_data_for_y_values(100, y_val, 256)
                return x, y
            def mock_load_saury_data(path, x_values, y_values, verbose=False):
                x = np.random.rand(80, 256)
                y = self.mock_data_for_y_values(80, y_val, 256)
                return x, y
            def mock_load_seta_data(path, sim, x_values, y_values, verbose=False):
                x = np.random.rand(60, 256)
                y = self.mock_data_for_y_values(60, y_val, 256)
                return x, y

            with patch('bayeshi.data_loaders.load_tigress_data', mock_load_tigress_data), \
                 patch('bayeshi.data_loaders.load_saury_data', mock_load_saury_data), \
                 patch('bayeshi.data_loaders.load_seta_data', mock_load_seta_data):

                train_loader, val_loader, test_loader = load_data(
                    dataset='all', y_values=y_val, split='all'
                )
                self.assertEqual(train_loader.dataset.tensors[1].shape[1], 256)
                self.assertEqual(val_loader.dataset.tensors[1].shape[1], 256)
                self.assertEqual(test_loader.dataset.tensors[1].shape[1], 256)

    def test_shapes(self):
        """Test that the x and y data shapes are (n_samples, 4) and (n_samples, 256) for fractions or emission/absorption respectively."""
        for x_val in ['emission', 'absorption']:
            for y_val in ['fractions', 'emission', 'absorption']:
                
                n_spectra = 200
                target_n_spectra = round((1 - 0.05 - 0.05) * n_spectra)
                torch_loader, *_ = load_data(dataset='saury', x_values=x_val, y_values=y_val, split=n_spectra, test_size = 0.05, val_size = 0.05)
                x_data, y_data = torch_loader.dataset.tensors
                if y_val == 'fractions':
                    self.assertEqual(x_data.shape, (target_n_spectra, 256))
                    self.assertEqual(y_data.shape, (target_n_spectra, 4))
                elif y_val in ['emission', 'absorption']:
                    self.assertEqual(x_data.shape, (target_n_spectra, 256))
                    self.assertEqual(y_data.shape, (target_n_spectra, 256))
    
    def test_train_val_test_split(self):
        """Test that train, val, and test splits are correctly sized."""
        for test_split in [0.05, 0.1, 0.5]:
            for val_split in [0.05, 0.1, 0.5]:
                if test_split + val_split >= 1.0:
                    with self.assertRaises(ValueError):
                        load_data(
                            dataset='saury', split=200, test_size=test_split, val_size=val_split
                        )
                else:
                    train_loader, val_loader, test_loader = load_data(
                        dataset='saury', split=200, test_size=test_split, val_size=val_split
                    )
                    total_samples = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
                    self.assertEqual(total_samples, 200)
                    self.assertEqual(len(train_loader.dataset), round(200 - (200 * (test_split + val_split))))
                    self.assertEqual(len(val_loader.dataset), round(200 * val_split))
                    self.assertEqual(len(test_loader.dataset), round(200 * test_split))

if __name__ == '__main__':
    unittest.main()
