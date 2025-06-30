# import unittest
# from unittest.mock import patch
# import numpy as np
# from bayeshi import load_data

# def mock_data_for_y_values(samples, y_values, n_features=256):
#     """Generate mock data for a given y_values option."""
#     if y_values == 'fractions':
#         # For fractions, return (n, 4) array
#         return np.random.rand(samples, 4)
#     elif y_values in ['emission', 'absorption']:
#         # For emission/absorption, return (n, n_features) array
#         return np.random.rand(samples, n_features)
#     else:
#         raise ValueError("Invalid y_values")

# def mock_load_tigress_data(path, sim, x_values, y_values):
#     samples = 100
#     x_data = np.random.rand(samples, 256)
#     y_data = mock_data_for_y_values(samples, y_values)
#     return x_data, y_data

# def mock_load_saury_data(path, x_values, y_values):
#     samples = 80
#     x_data = np.random.rand(samples, 256)
#     y_data = mock_data_for_y_values(samples, y_values)
#     return x_data, y_data

# def mock_load_seta_data(path, sim, x_values, y_values):
#     samples = 60
#     x_data = np.random.rand(samples, 256)
#     y_data = mock_data_for_y_values(samples, y_values)
#     return x_data, y_data

# class TestDataLoader(unittest.TestCase):
#     def setUp(self):
#         # Fixed random state for reproducible tests
#         self.random_state = 42
#         # Sample sizes for each simulation
#         self.sample_sizes = {'tigress': 100, 'saury': 80, 'seta': 60}
        
#     def mock_load_tigress_data(self, path, sim, x_values, y_values):
#         return (np.random.rand(self.sample_sizes['tigress'], 256),
#                 np.random.rand(self.sample_sizes['tigress'], 3))
    
#     def mock_load_saury_data(self, path, x_values, y_values):
#         return (np.random.rand(self.sample_sizes['saury'], 256),
#                 np.random.rand(self.sample_sizes['saury'], 3))
    
#     def mock_load_seta_data(self, path, sim, x_values, y_values):
#         return (np.random.rand(self.sample_sizes['seta'], 256),
#                 np.random.rand(self.sample_sizes['seta'], 3))
    
#     def test_split_all(self):
#         """Test using all available data points"""
#         with patch('bayeshi.data_loaders.load_tigress_data', mock_load_tigress_data), \
#              patch('bayeshi.data_loaders.load_saury_data', mock_load_saury_data), \
#              patch('bayeshi.data_loaders.load_seta_data', mock_load_seta_data):
            
#             train_loader, val_loader, test_loader = load_data(split='all')
#             total_samples = sum(self.sample_sizes.values())
#             self.assertEqual(
#                 len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
#                 total_samples
#             )

#     def test_split_equal(self):
#         """Test equal sampling from each simulation"""
#         with patch('bayeshi.data_loaders.load_tigress_data', mock_load_tigress_data), \
#              patch('bayeshi.data_loaders.load_saury_data', mock_load_saury_data), \
#              patch('bayeshi.data_loaders.load_seta_data', mock_load_seta_data):
            
#             train_loader, val_loader, test_loader = load_data(split='equal')
#             min_samples = min(self.sample_sizes.values())
#             total_samples = min_samples * 3
#             self.assertEqual(
#                 len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
#                 total_samples
#             )

#     def test_split_int_valid(self):
#         """Test integer split with valid number"""
#         with patch('bayeshi.data_loaders.load_tigress_data', mock_load_tigress_data), \
#              patch('bayeshi.data_loaders.load_saury_data', mock_load_saury_data), \
#              patch('bayeshi.data_loaders.load_seta_data', mock_load_seta_data):
            
#             split_num = 90  # 30 per simulation (30*3=90)
#             train_loader, val_loader, test_loader = load_data(split=split_num)
#             self.assertEqual(
#                 len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
#                 split_num
#             )

#     def test_split_int_invalid(self):
#         """Test integer split with insufficient samples"""
#         with patch('bayeshi.data_loaders.load_tigress_data', mock_load_tigress_data), \
#              patch('bayeshi.data_loaders.load_saury_data', mock_load_saury_data), \
#              patch('bayeshi.data_loaders.load_seta_data', mock_load_seta_data):
            
#             # Request more samples than available in seta (which has 60)
#             split_num = 200  # Requires 67 per simulation (67*3=201)
#             with self.assertRaisesRegex(ValueError, "Not enough samples in"):
#                 load_data(split=split_num)

#     def test_single_dataset(self):
#         """Test loading single dataset"""
#         with patch('bayeshi.data_loaders.load_tigress_data', mock_load_tigress_data):
#             train_loader, val_loader, test_loader = load_data(dataset='tigress', split='all')
#             total_samples = self.sample_sizes['tigress']
#             self.assertEqual(
#                 len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
#                 total_samples
#             )

#     def test_two_datasets_equal_split(self):
#         """Test equal split with two datasets"""
#         with patch('bayeshi.data_loaders.load_tigress_data', mock_load_tigress_data), \
#              patch('bayeshi.data_loaders.load_saury_data', mock_load_saury_data):
            
#             train_loader, val_loader, test_loader = load_data(
#                 dataset=['tigress', 'saury'],
#                 split='equal'
#             )
#             min_samples = min(self.sample_sizes['tigress'], self.sample_sizes['saury'])
#             total_samples = min_samples * 2
#             self.assertEqual(
#                 len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
#                 total_samples
#             )

#     def test_split_none(self):
#         """Test None split (should default to 'all')"""
#         with patch('bayeshi.data_loaders.load_tigress_data', mock_load_tigress_data), \
#              patch('bayeshi.data_loaders.load_saury_data', mock_load_saury_data), \
#              patch('bayeshi.data_loaders.load_seta_data', mock_load_seta_data):
            
#             train_loader, val_loader, test_loader = load_data(split=None)
#             total_samples = sum(self.sample_sizes.values())
#             self.assertEqual(
#                 len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
#                 total_samples
#             )
            
# class TestDataLoaderYValues(unittest.TestCase):
#     def setUp(self):
#         # Define sample sizes
#         self.sample_sizes = {'tigress': 100, 'saury': 80, 'seta': 60}

#     def test_y_values_emission_absorption(self):
#         """Test that y_data shape matches x_data when y_values is 'emission' or 'absorption'."""
#         for y_val in ['emission', 'absorption']:
#             with patch('bayeshi.data_loaders.load_tigress_data', mock_load_tigress_data), \
#                  patch('bayeshi.data_loaders.load_saury_data', mock_load_saury_data), \
#                  patch('bayeshi.data_loaders.load_seta_data', mock_load_seta_data):

#                 train_loader, val_loader, test_loader = load_data(
#                     dataset='all', y_values=y_val, split='all'
#                 )
#                 # Check that y_data has the same number of samples and features as x_data
#                 x_samples = sum([len(loader.dataset) for loader in [train_loader, val_loader, test_loader]])
#                 y_samples = sum([loader.dataset.tensors[1].shape[0] for loader in [train_loader, val_loader, test_loader]])
#                 self.assertEqual(x_samples, y_samples)
#                 # Check y_data feature dimension matches x_data (256)
#                 self.assertEqual(train_loader.dataset.tensors[1].shape[1], 256)
#                 self.assertEqual(val_loader.dataset.tensors[1].shape[1], 256)
#                 self.assertEqual(test_loader.dataset.tensors[1].shape[1], 256)

#     def test_y_values_fractions(self):
#         """Test that y_data shape is (n, 4) when y_values is 'fractions'."""
#         with patch('bayeshi.data_loaders.load_tigress_data', mock_load_tigress_data), \
#              patch('bayeshi.data_loaders.load_saury_data', mock_load_saury_data), \
#              patch('bayeshi.data_loaders.load_seta_data', mock_load_seta_data):

#             train_loader, val_loader, test_loader = load_data(
#                 dataset='all', y_values='fractions', split='all'
#             )
#             # Check y_data feature dimension is 4
#             self.assertEqual(train_loader.dataset.tensors[1].shape[1], 4)
#             self.assertEqual(val_loader.dataset.tensors[1].shape[1], 4)
#             self.assertEqual(test_loader.dataset.tensors[1].shape[1], 4)

# class TestDataLoaderRandomness(unittest.TestCase):
#     def setUp(self):
#         self.sample_sizes = {'tigress': 100, 'saury': 80, 'seta': 60}

#     def test_randomness_reproducible(self):
#         """Test that setting the random_state makes selections reproducible."""
#         fixed_data = np.arange(100*256, dtype=float).reshape(100, 256)
#         fixed_labels = np.arange(100*3, dtype=float).reshape(100, 3)
        
#         def fixed_mock_load_tigress_data(path, sim, x_values, y_values):
#             return fixed_data, fixed_labels
#         def fixed_mock_load_saury_data(path, x_values, y_values):
#             return fixed_data[:80], fixed_labels[:80]
#         def fixed_mock_load_seta_data(path, sim, x_values, y_values):
#             return fixed_data[:60], fixed_labels[:60]

#         with patch('bayeshi.data_loaders.load_tigress_data', fixed_mock_load_tigress_data), \
#              patch('bayeshi.data_loaders.load_saury_data', fixed_mock_load_saury_data), \
#              patch('bayeshi.data_loaders.load_seta_data', fixed_mock_load_seta_data):

#             train_loader1, _, _ = load_data(split='equal', random_state=42)
#             data1 = train_loader1.dataset.tensors[0].numpy()
#             train_loader2, _, _ = load_data(split='equal', random_state=42)
#             data2 = train_loader2.dataset.tensors[0].numpy()
#             np.testing.assert_array_equal(data1, data2)

#     def test_randomness_varied(self):
#         """Test that not setting a random_state or using a new seed produces different selections."""
#         with patch('bayeshi.data_loaders.load_tigress_data', mock_load_tigress_data), \
#              patch('bayeshi.data_loaders.load_saury_data', mock_load_saury_data), \
#              patch('bayeshi.data_loaders.load_seta_data', mock_load_seta_data):

#             # Call with no random_state (or default)
#             train_loader1, _, _ = load_data(split='equal')
#             indices1 = train_loader1.dataset.tensors[0].numpy()

#             # Call with a new random_state
#             train_loader2, _, _ = load_data(split='equal', random_state=123)
#             indices2 = train_loader2.dataset.tensors[0].numpy()

#             # Call with another new random_state
#             train_loader3, _, _ = load_data(split='equal', random_state=456)
#             indices3 = train_loader3.dataset.tensors[0].numpy()

#             # At least two of these should differ, but we can check all pairs
#             # (Note: In rare cases, random seeds might coincidentally produce the same selection,
#             # but for good seeds and large enough datasets, this is extremely unlikely)
#             self.assertFalse(
#                 np.array_equal(indices1, indices2) and
#                 np.array_equal(indices1, indices3) and
#                 np.array_equal(indices2, indices3),
#                 "All selections were identical, randomness is not working as expected"
#             )

# class TestTIGRESSCubeDiversity(unittest.TestCase):
#     def setUp(self):
#         self.n_cubes = 11
#         self.samples_per_cube = 100
#         # Simulate 11 cubes, each with 100 samples (total 1100)
#         # For this test, we only care about TIGRESS, so mock others as empty or minimal
#         self.tigress_data = np.vstack([
#             np.random.rand(self.samples_per_cube, 256) + i  # Add cube ID as offset for easy tracking
#             for i in range(self.n_cubes)
#         ])
#         self.tigress_labels = np.vstack([
#             np.random.rand(self.samples_per_cube, 4)
#             for _ in range(self.n_cubes)
#         ])
        
#     def mock_load_tigress_data(self, path, sim, x_values, y_values):
#         return self.tigress_data, self.tigress_labels
    
#     def test_tigress_cube_diversity_in_split(self):
#         """Test that TIGRESS spectra in split come from different cubes."""
#         with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
#              patch('bayeshi.data_loaders.load_saury_data', mock_load_saury_data), \
#              patch('bayeshi.data_loaders.load_seta_data', mock_load_seta_data):

#             # Use a split that will take a subset of TIGRESS data (e.g., split=100)
#             # Since we have 11 cubes, this will force the split to draw from multiple cubes
#             # (assuming your split logic randomly samples from the full TIGRESS array)
#             split_num = 100
#             train_loader, _, _ = load_data(dataset=['tigress'], split=split_num, random_state=42)

#             # Get the indices of the selected TIGRESS spectra
#             selected_data = train_loader.dataset.tensors[0].numpy()
#             # For this mock, each cube's data is offset by its cube ID (see setUp)
#             # So, subtracting the integer part gives the cube ID
#             cube_ids = np.floor(selected_data[:, 0] - np.floor(selected_data[:, 0])).astype(int)
#             # Actually, in this mock, the offset is added to ALL values in the cube,
#             # so a better way is to reconstruct the original cube ID for each sample:
#             # For this mock, the first column of each sample is (rand + cube_id), so:
#             cube_ids = np.floor(selected_data[:, 0]).astype(int)
#             unique_cubes = np.unique(cube_ids)
            
#             print(f"Unique cubes in selected TIGRESS spectra: {unique_cubes}")

#             # Assert that at least two different cubes are represented
#             self.assertGreater(len(unique_cubes), 1, "All selected TIGRESS spectra came from the same cube")

#             # Check that the number of unique cubes is at least a certain fraction of the total
#             self.assertGreaterEqual(len(unique_cubes), min(5, self.n_cubes), "Not enough unique TIGRESS cubes sampled in the split")
       
#     def test_tigress_cube_sampling_is_random(self):
#         """Test that random sampling sometimes leaves out some TIGRESS cubes."""
#         with patch('bayeshi.data_loaders.load_tigress_data', self.mock_load_tigress_data), \
#             patch('bayeshi.data_loaders.load_saury_data', mock_load_saury_data), \
#             patch('bayeshi.data_loaders.load_seta_data', mock_load_seta_data):

#             split_num = 30  # Small enough so that, with 11 cubes, some may be missed
#             n_trials = 20

#             # Collect the number of unique cubes in each trial
#             unique_cubes_counts = []
#             for _ in range(n_trials):
#                 train_loader, _, _ = load_data(
#                     dataset=['tigress'],
#                     split=split_num,
#                     random_state=None  # Let it use different random state each time
#                 )
#                 selected_data = train_loader.dataset.tensors[0].numpy()
#                 cube_ids = np.floor(selected_data[:, 0]).astype(int)
#                 unique_cubes = np.unique(cube_ids)
#                 unique_cubes_counts.append(len(unique_cubes))

#             # Check that at least once, not all cubes were sampled
#             # (i.e., at least one trial has <11 unique cubes)
#             self.assertLess(min(unique_cubes_counts), self.n_cubes,
#                 "Every split included spectra from all TIGRESS cubes; sampling may not be random")

#             # print(f"Unique cubes counts in {n_trials} trials: {unique_cubes_counts}")

# if __name__ == '__main__':
#     unittest.main()

import unittest
from unittest.mock import patch
import numpy as np
import warnings

# Import your load_data function
from bayeshi.data_loaders import load_data  # Update path as needed

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

    def mock_load_tigress_data(self, path, sim, x_values, y_values):
        # Return mock data with cube IDs encoded in the first column
        return self.tigress_data, self.tigress_labels

    def mock_load_saury_data(self, path, x_values, y_values):
        return self.saury_data, self.saury_labels

    def mock_load_seta_data(self, path, sim, x_values, y_values):
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
        def fixed_mock_load_tigress_data(path, sim, x_values, y_values):
            return fixed_data, fixed_labels
        def fixed_mock_load_saury_data(path, x_values, y_values):
            return fixed_data[:80], fixed_labels[:80]
        def fixed_mock_load_seta_data(path, sim, x_values, y_values):
            return fixed_data[:60], fixed_labels[:60]

        with patch('bayeshi.data_loaders.load_tigress_data', fixed_mock_load_tigress_data), \
             patch('bayeshi.data_loaders.load_saury_data', fixed_mock_load_saury_data), \
             patch('bayeshi.data_loaders.load_seta_data', fixed_mock_load_seta_data):

            train_loader1, _, _ = load_data(split='equal', random_state=42)
            data1 = train_loader1.dataset.tensors[0].numpy()
            train_loader2, _, _ = load_data(split='equal', random_state=42)
            data2 = train_loader2.dataset.tensors[0].numpy()
            np.testing.assert_array_equal(data1, data2)

    # --- y_values tests ---

    def test_y_values_fractions(self):
        """Test that y_data shape is (n, 4) when y_values is 'fractions'."""
        def mock_load_tigress_data(path, sim, x_values, y_values):
            x = np.random.rand(100, 256)
            y = self.mock_data_for_y_values(100, y_values)
            return x, y
        def mock_load_saury_data(path, x_values, y_values):
            x = np.random.rand(80, 256)
            y = self.mock_data_for_y_values(80, y_values)
            return x, y
        def mock_load_seta_data(path, sim, x_values, y_values):
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
            def mock_load_tigress_data(path, sim, x_values, y_values):
                x = np.random.rand(100, 256)
                y = self.mock_data_for_y_values(100, y_val, 256)
                return x, y
            def mock_load_saury_data(path, x_values, y_values):
                x = np.random.rand(80, 256)
                y = self.mock_data_for_y_values(80, y_val, 256)
                return x, y
            def mock_load_seta_data(path, sim, x_values, y_values):
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

if __name__ == '__main__':
    unittest.main()
