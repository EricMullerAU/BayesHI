import numpy as np
import warnings
from astropy.io import fits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# def load_data(data_path='/scratch/mk27/em8117/', x_values = 'emission', y_values='fractions', dataset = 'all', tigress_sim = 'all', seta_sim = 'both', split = None, batch_size=32, num_workers=4, noise=0.5, test_size = 0.2, val_size = 0.2, random_state=42, show_example=False):
#     if x_values not in ['emission', 'absorption']:
#         raise ValueError("x_values must be either 'emission' or 'absorption'")
    
#     if y_values not in ['fractions', 'absorption', 'emission']:
#         raise ValueError("y_values must be either 'fractions', 'absorption', or 'emission'")
    
#     x_data = np.array([])
#     y_data = np.array([])
    
#     if dataset == 'all':
#         # Load TIGRESS, Saury, and Seta emission data
#         tigress_x, tigress_y = load_tigress_data(data_path + 'TIGRESS/', tigress_sim, x_values, y_values)
#         saury_x, saury_y  = load_saury_data(data_path + 'Saury/', x_values, y_values)
#         seta_x, seta_y = load_seta_data(data_path + 'Seta/', seta_sim, x_values, y_values)
        
#         # Always want to preserve the second axis (velocity or fraction) for the data, so concat on axis 0
#         x_data = np.concatenate((tigress_x, saury_x, seta_x), axis=0)
#         y_data = np.concatenate((tigress_y, saury_y, seta_y), axis=0)
        
#     elif type(dataset) is list:
#         for sim in dataset:
#             if sim.startswith('tigress'):
#                 tigress_x, tigress_y = load_tigress_data(data_path + 'TIGRESS/', tigress_sim, x_values, y_values)
#                 x_data = np.append(x_data, tigress_x, axis=0)
#                 y_data = np.append(y_data, tigress_y, axis=0)
#             elif sim.startswith('saury'):
#                 saury_x, saury_y = load_saury_data(data_path + 'Saury/', x_values, y_values)
#                 x_data = np.append(x_data, saury_x, axis=0)
#                 y_data = np.append(y_data, saury_y, axis=0)
#             elif sim.startswith('seta'):
#                 seta_x, seta_y = load_seta_data(data_path + 'Seta/', seta_sim, x_values, y_values)
#                 x_data = np.append(x_data, seta_x, axis=0)
#                 y_data = np.append(y_data, seta_y, axis=0)
#             else:
#                 raise ValueError(f"Unknown dataset type: {sim}")

#     elif type(dataset) is str:
#         if dataset.startswith('tigress'):
#             x_data, y_data = load_tigress_data(data_path + 'TIGRESS/', tigress_sim, x_values, y_values)
#         elif dataset.startswith('saury'):
#             x_data, y_data = load_saury_data(data_path + 'Saury/', x_values, y_values)
#         elif dataset.startswith('seta'):
#             x_data, y_data = load_seta_data(data_path + 'Seta/', seta_sim, x_values, y_values)
#         else:
#             raise ValueError(f"Unknown dataset type: {dataset}")
#     else:
#         raise ValueError("dataset must be 'all', a list of dataset names, or a single dataset name string")
    
#     print('Shape of x_data:', x_data.shape)
#     print('Shape of y_data:', y_data.shape)

# def load_data(data_path='/scratch/mk27/em8117/', x_values='emission', y_values='fractions', 
#               dataset='all', tigress_sim='all', seta_sim='both', split=None, batch_size=32, 
#               num_workers=4, noise=0.5, test_size=0.2, val_size=0.2, random_state=42, 
#               show_example=False):

def load_data(data_path='/scratch/mk27/em8117/', x_values='emission', y_values='fractions',
              dataset='all', tigress_sim='all', seta_sim='both', split=None,
              batch_size=32, num_workers=4, noise=0.5, test_size=0.2, val_size=0.2,
              random_state=42, show_example=False):

    # Validate x_values and y_values
    if x_values not in ['emission', 'absorption']:
        raise ValueError("x_values must be either 'emission' or 'absorption'")
    if y_values not in ['fractions', 'absorption', 'emission']:
        raise ValueError("y_values must be either 'fractions', 'absorption', or 'emission'")

    # Initialize variables to store loaded data
    tigress_x, tigress_y = np.array([]), np.array([])
    saury_x, saury_y = np.array([]), np.array([])
    seta_x, seta_y = np.array([]), np.array([])

    # Load data based on dataset parameter
    if dataset == 'all':
        tigress_x, tigress_y = load_tigress_data(data_path + 'TIGRESS/', tigress_sim, x_values, y_values)
        saury_x, saury_y = load_saury_data(data_path + 'Saury/', x_values, y_values)
        seta_x, seta_y = load_seta_data(data_path + 'Seta/', seta_sim, x_values, y_values)
    elif isinstance(dataset, list):
        for sim in dataset:
            if sim.startswith('tigress'):
                tigress_x, tigress_y = load_tigress_data(data_path + 'TIGRESS/', tigress_sim, x_values, y_values)
            elif sim.startswith('saury'):
                saury_x, saury_y = load_saury_data(data_path + 'Saury/', x_values, y_values)
            elif sim.startswith('seta'):
                seta_x, seta_y = load_seta_data(data_path + 'Seta/', seta_sim, x_values, y_values)
            else:
                raise ValueError(f"Unknown dataset type: {sim}")
    elif isinstance(dataset, str):
        if dataset.startswith('tigress'):
            tigress_x, tigress_y = load_tigress_data(data_path + 'TIGRESS/', tigress_sim, x_values, y_values)
        elif dataset.startswith('saury'):
            saury_x, saury_y = load_saury_data(data_path + 'Saury/', x_values, y_values)
        elif dataset.startswith('seta'):
            seta_x, seta_y = load_seta_data(data_path + 'Seta/', seta_sim, x_values, y_values)
        else:
            raise ValueError(f"Unknown dataset type: {dataset}")
    else:
        raise ValueError("dataset must be 'all', a list of dataset names, or a single dataset name string")

    # Collect non-empty simulations
    sim_data = {
        'tigress': (tigress_x, tigress_y) if tigress_x.size > 0 else None,
        'saury': (saury_x, saury_y) if saury_x.size > 0 else None,
        'seta': (seta_x, seta_y) if seta_x.size > 0 else None
    }
    active_sims = [k for k, v in sim_data.items() if v is not None]
    if not active_sims:
        raise ValueError("No valid simulations loaded")

    # Process split argument
    if split == 'all':
        # Use all available data
        pass
    elif split == 'equal':
        # Use equal number of samples from each loaded simulation
        n_per_sim = min(v[0].shape[0] for v in sim_data.values() if v is not None)
        rng = np.random.RandomState(random_state)
        for sim in active_sims:
            x, y = sim_data[sim]
            idx = rng.permutation(x.shape[0])[:n_per_sim]
            sim_data[sim] = (x[idx], y[idx])
    elif isinstance(split, int):
        # Split as equally as possible across simulations
        n_total = split
        n_sims = len(active_sims)
        n_per_sim = n_total // n_sims
        remainder = n_total % n_sims
        print(f"Splitting {n_total} samples across {n_sims} simulations: {n_per_sim} per simulation, with {remainder} extra samples distributed")
        rng = np.random.RandomState(random_state)
        for i, sim in enumerate(active_sims):
            x, y = sim_data[sim]
            n = n_per_sim + (1 if i < remainder else 0)
            if n > x.shape[0]:
                raise ValueError(f"Not enough samples in {sim} for split={split}")
            idx = rng.permutation(x.shape[0])[:n]
            sim_data[sim] = (x[idx], y[idx])
    elif isinstance(split, dict):
        # Dictionary split (proportion, percent, or absolute)
        required_keys = {'tigress', 'saury', 'seta'}
        if not required_keys.issubset(split.keys()):
            raise ValueError("split dictionary must contain 'tigress', 'saury', and 'seta' keys")
        for key in required_keys:
            if not isinstance(split[key], (int, float)):
                raise ValueError(f"split['{key}'] must be a number")
        sum_split = sum(split.values())
        # Check if numbers are proportions, percentages, or absolute numbers
        if all(0 <= v <= 1 for v in split.values()):
            if np.isclose(sum_split, 1.0):
                is_proportion = True  # proportions
                # Print the splits as percentages for each simulation
                print("Splitting data as proportions:")
                for sim in active_sims:
                    print(f"{sim}: {split[sim] * 100:.2f}%")
            else:
                raise ValueError("if using proportions for data splitting, the sum of the split values must be 1.0")
        elif all(0 <= v <= 100 for v in split.values()):
            if np.isclose(sum_split, 100.0):
                is_proportion = False  # percentages
                # Print the splits as percentages for each simulation
                print("Splitting data as percentages:")
                for sim in active_sims:
                    print(f"{sim}: {split[sim]:.2f}%")
            else:
                # Treat as absolute numbers if not close to 100
                warnings.warn("Splitting data as absolute numbers:", UserWarning)
                for sim in active_sims:
                    print(f"{sim}: {split[sim]} spectra")
                is_proportion = None  # absolute numbers
        elif all(isinstance(v, int) for v in split.values()):
            print("Splitting data as absolute numbers:")
            for sim in active_sims:
                print(f"{sim}: {split[sim]} spectra")
            is_proportion = None  # absolute numbers
            if any(v < 0 for v in split.values()):
                raise ValueError("if using absolute numbers for data splitting, all values must be non-negative")
        else:
            raise ValueError("split values must be either proportions (0-1), percentages (0-100), or absolute numbers (non-negative integers)")
        rng = np.random.RandomState(random_state)
        for sim in active_sims:
            x, y = sim_data[sim]
            if is_proportion is not None:
                if is_proportion:
                    n = int(round(split[sim] * x.shape[0]))
                else:
                    n = int(round(split[sim] / 100.0 * x.shape[0]))
            else:
                n = min(split[sim], x.shape[0])
            idx = rng.permutation(x.shape[0])[:n]
            sim_data[sim] = (x[idx], y[idx])
    elif split is not None:
        raise ValueError("split must be 'all', 'equal', an integer, a dictionary, or None")

    # Concatenate the data
    x_arrays = [v[0] for v in sim_data.values() if v is not None]
    y_arrays = [v[1] for v in sim_data.values() if v is not None]
    x_data = np.concatenate(x_arrays, axis=0) if x_arrays else np.array([])
    y_data = np.concatenate(y_arrays, axis=0) if y_arrays else np.array([])

    # Concatenate the data
    x_arrays = [v[0] for v in sim_data.values() if v is not None]
    y_arrays = [v[1] for v in sim_data.values() if v is not None]
    x_data = np.concatenate(x_arrays, axis=0) if x_arrays else np.array([])
    y_data = np.concatenate(y_arrays, axis=0) if y_arrays else np.array([])
    
      
    x_data += np.random.randn(*x_data.shape) * noise
    
    print('Total number of spectra:', x_data.shape[0])

    # Remove lines of sight with NaNs or all zeros
    # nanIndices = np.isnan(spectra).any(axis=1) | np.isnan(fcnm) | np.isnan(funm) | np.isnan(fwnm) | np.isnan(rhi) | np.all(spectra == 0, axis=1)
    # spectra = spectra[~nanIndices]
    # fcnm = fcnm[~nanIndices]
    # funm = funm[~nanIndices]
    # fwnm = fwnm[~nanIndices]
    # rhi = rhi[~nanIndices]
    # los_removed = np.sum(nanIndices)
        
    # print(f'Removed {los_removed} lines of sight with NaNs')
    
    # Split the data into training, validation, and testing sets
    train_pct = int((1 - test_size - val_size) * 100)
    val_pct = int(val_size * 100)
    test_pct = int(test_size * 100)
    
    print(f'Splitting data into {train_pct}% train, {val_pct}% validation, and {test_pct}% test sets.')
    X_train, X_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=test_size + val_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state)
    
    train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
    val_dataset = TensorDataset(Tensor(X_val), Tensor(y_val))
    test_dataset = TensorDataset(Tensor(X_test), Tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    if show_example:
        import matplotlib.pyplot as plt
        random_indices = np.random.choice(spectra.shape[0], size=5, replace=False)
        plt.figure(figsize=(10, 6))
        for i in random_indices:
            if x_values == 'emission':
                plt.plot(spectra[i].T)
            elif x_values == 'absorption':
                plt.plot(np.exp(-spectra[i]).T)
        plt.xlabel('Channel')
        if x_values == 'emission':
            plt.ylabel(r'$T_B$ [K]')
        elif x_values == 'absorption':
            plt.ylabel(r'$e^{-\tau}$')
        plt.title('Example Spectra')
        plt.show()
    
    return train_loader, val_loader, test_loader

def load_tigress_data(data_path, sim_number='all', x_values='emission', y_values='fractions'):
    if sim_number == 'all':
        sim_number = np.arange(290, 391, 10)
    elif type(sim_number) is list:
        sim_number = [int(s) for s in sim_number]
    else:
        try:
            sim_number = [int(sim_number)]
        except:
            raise ValueError("sim_number should be 'all', a list of integers, or a single integer.")
        
    if x_values not in ['emission', 'absorption']:
        raise ValueError("x_values must be either 'emission' or 'absorption'")
    if y_values not in ['fractions', 'absorption', 'emission']:
        raise ValueError("y_values must be either 'fractions', 'absorption', or 'emission'")

    # x_data = np.array([])
    # y_data = np.array([])
    # Preallocate arrays to hold data
    # All TIGRESS cubes are 256 x 256 x 512 (v, z, x)
    total_spectra = len(sim_number) * 256 * 512
    x_data = np.empty((total_spectra, 256))  # 256 channels
    if y_values == 'fractions':
        y_data = np.empty((total_spectra, 4))
    else:
        y_data = np.empty((total_spectra, 256))
            
    for i, sim in enumerate(sim_number):
        if x_values == 'emission':
            spectra = fits.getdata(data_path + f'{sim}_Tb_FINAL.fits')[:, 3584//2-128:3584//2+128, :]
        elif x_values == 'absorption':
            spectra = fits.getdata(data_path + f'{sim}_Tau_FINAL.fits')[:, 3584//2-128:3584//2+128, :]
        else:
            raise ValueError("x_values must be either 'emission' or 'absorption'")
                
        # Change from (v, z, x) to (x*z, v)
        spectra = np.moveaxis(spectra, 0, -1)
        spectra = spectra.reshape(-1, spectra.shape[-1])
        
        # Add the spectra to x_data
        start_idx = (i * 256 * 512)
        end_idx = start_idx + spectra.shape[0]
        x_data[start_idx:end_idx, :] = spectra
        
        if y_values == 'fractions':
            fcnm = fits.getdata(data_path + f'{sim}_fcnm_FINAL.fits')[3584//2-128:3584//2+128, :]
            funm = fits.getdata(data_path + f'{sim}_funm_FINAL.fits')[3584//2-128:3584//2+128, :]
            fwnm = fits.getdata(data_path + f'{sim}_fwnm_FINAL.fits')[3584//2-128:3584//2+128, :]
            rhi = fits.getdata(data_path + f'{sim}_rhi_FINAL.fits')[3584//2-128:3584//2+128, :]
            
            # Stack from (z, x) to (z, x, 4)
            fractions = np.stack((fcnm, funm, fwnm, rhi), axis=2)
            # Change from (z, x, 4) to (x*z, 4)
            fractions = fractions.reshape(-1, 4)
            y_data_temp = fractions
            
        elif y_values == 'absorption':
            absorption = fits.getdata(data_path + f'{sim}_Tau_FINAL.fits')[:, 3584//2-128:3584//2+128, :]
            # Change from (v, z, x) to (x*z, 1)
            absorption = np.moveaxis(absorption, 0, -1)
            y_data_temp = absorption.reshape(-1, absorption.shape[-1])
        elif y_values == 'emission':
            emission = fits.getdata(data_path + f'{sim}_Tb_FINAL.fits')[:, 3584//2-128:3584//2+128, :]
            # Change from (v, z, x) to (x*z, 1)
            emission = np.moveaxis(emission, 0, -1)
            y_data_temp = emission.reshape(-1, emission.shape[-1])
        else:
            raise ValueError("y_values must be either 'fractions', 'absorption', or 'emission'")
    
        # Add the y_data to y_data
        y_data[start_idx:end_idx, :] = y_data_temp
            
    return x_data, y_data

def load_saury_data(data_path, x_values='emission', y_values='fractions'):
    if x_values not in ['emission', 'absorption']:
        raise ValueError("x_values must be either 'emission' or 'absorption'")
    if y_values not in ['fractions', 'absorption', 'emission']:
        raise ValueError("y_values must be either 'fractions', 'absorption', or 'emission'")

    if x_values == 'emission':
        x_data = fits.getdata(data_path + 'saury_Tb.fits')
    elif x_values == 'absorption':
        x_data = fits.getdata(data_path + 'saury_Tau.fits')
    else:
        raise ValueError("x_values must be either 'emission' or 'absorption'")
    
    # Change from (v, z, x) to (x*z, v)
    x_data = np.moveaxis(x_data, 0, -1)
    x_data = x_data.reshape(-1, x_data.shape[-1])
    
    if y_values == 'fractions':
        fcnm = fits.getdata(data_path + 'saury_fcnm.fits')
        funm = fits.getdata(data_path + 'saury_funm.fits')
        fwnm = fits.getdata(data_path + 'saury_fwnm.fits')
        rhi = fits.getdata(data_path + 'saury_rhi.fits')
        
        # Stack from (z, x) to (z, x, 4)
        fractions = np.stack((fcnm, funm, fwnm, rhi), axis=2)
        # Change from (z, x, 4) to (x*z, 4)
        y_data = fractions.reshape(-1, 4)
        
    elif y_values == 'absorption':
        absorption = fits.getdata(data_path + 'saury_Tau.fits')
        # Change from (v, z, x) to (x*z, 1)
        absorption = np.moveaxis(absorption, 0, -1)
        y_data = absorption.reshape(-1, absorption.shape[-1])
    elif y_values == 'emission':
        emission = fits.getdata(data_path + 'saury_Tb.fits')
        # Change from (v, z, x) to (x*z, 1)
        emission = np.moveaxis(emission, 0, -1)
        y_data = emission.reshape(-1, emission.shape[-1])
    else:
        raise ValueError("y_values must be either 'fractions', 'absorption', or 'emission'")
    
    return x_data, y_data

def load_seta_data(data_path, sim_type='both', x_values='emission', y_values='fractions'):
    if sim_type not in ['both', 'comp', 'sol']:
        raise ValueError("sim_type must be 'both', 'comp', or 'sol'")
    
    if x_values not in ['emission', 'absorption']:
        raise ValueError("x_values must be either 'emission' or 'absorption'")
    if y_values not in ['fractions', 'absorption', 'emission']:
        raise ValueError("y_values must be either 'fractions', 'absorption', or 'emission'")
    
    if sim_type == 'both':
        sims = ['comp', 'sol']
    else:
        sims = [sim_type]
        
    x_data = np.array([])
    y_data = np.array([])
    
    for sim in sims:
        if x_values == 'emission':
            spectra = fits.getdata(data_path + f'seta_{sim}_Tb.fits')
        elif x_values == 'absorption':
            spectra = fits.getdata(data_path + f'seta_{sim}_Tau.fits')
        else:
            raise ValueError("x_values must be either 'emission' or 'absorption'")
        
        # Change from (v, z, x) to (x*z, v)
        spectra = np.moveaxis(spectra, 0, -1)
        spectra = spectra.reshape(-1, spectra.shape[-1])
        
        if x_data.size == 0:
            x_data = spectra
        else:
            x_data = np.append(x_data, spectra, axis=0)
            
        if y_values == 'fractions':
            fcnm = fits.getdata(data_path + f'seta_{sim}_fcnm.fits')
            funm = fits.getdata(data_path + f'seta_{sim}_funm.fits')
            fwnm = fits.getdata(data_path + f'seta_{sim}_fwnm.fits')
            rhi = fits.getdata(data_path + f'seta_{sim}_rhi.fits')
            
            # Stack from (z, x) to (z, x, 4)
            fractions = np.stack((fcnm, funm, fwnm, rhi), axis=2)
            # Change from (z, x, 4) to (x*z, 4)
            fractions = fractions.reshape(-1, 4)
            y_data_temp = fractions
        elif y_values == 'absorption':
            absorption = fits.getdata(data_path + f'seta_{sim}_Tau.fits')
            # Change from (v, z, x) to (x*z, 1)
            absorption = np.moveaxis(absorption, 0, -1)
            y_data_temp = absorption.reshape(-1, absorption.shape[-1])
        elif y_values == 'emission':
            emission = fits.getdata(data_path + f'seta_{sim}_Tb.fits')
            # Change from (v, z, x) to (x*z, 1)
            emission = np.moveaxis(emission, 0, -1)
            y_data_temp = emission.reshape(-1, emission.shape[-1])
        else:
            raise ValueError("y_values must be either 'fractions', 'absorption', or 'emission'")
        
        if y_data.size == 0:
            y_data = y_data_temp
        else:
            y_data = np.append(y_data, y_data_temp, axis=0)
            
    return x_data, y_data
        

# def load_absorption_data(data_path='/scratch/mk27/em8117/R8_2pc/', sim_number = 'all', batch_size=32, num_workers=4, noise=0.5, test_size = 0.2, val_size = 0.2, random_state=42, show_example=False):
#     if sim_number == 'all':
#         sim_number = np.arange(290, 391, 10)
#         # Remove 340 from the list while it's broken
#         sim_number = np.delete(sim_number, np.where(sim_number == 340))
#     elif sim_number == '340':
#         raise NotImplementedError("Simulation 340 is currently broken and cannot be loaded.")
#     else:
#         sim_number = [int(sim_number)]
        
#     all_spectra = []
#     all_absorption = []
        
#     for sim in sim_number:
#         print(f'Loading data for simulation {sim}')
        
#         spectra = fits.getdata(f'/scratch/mk27/em8117/R8_2pc/0{sim}/{sim}_Tb_full.fits')[:,1600:1950,:]
#         absorption = fits.getdata(f'/scratch/mk27/em8117/R8_2pc/0{sim}/{sim}_Tau_full.fits')[:,1600:1950,:]
        
#         all_spectra.append(spectra)
#         all_absorption.append(absorption)
    
#     spectra = np.concatenate(all_spectra, axis=1) # Concatenate along the z-axis
#     abs_spectra = np.concatenate(all_absorption, axis=1)
    
#     spectra += np.random.randn(*spectra.shape) * noise

#     # Change from (v, z, x) to (x*z, v)
#     spectra = np.moveaxis(spectra, 0, -1)
#     spectra = spectra.reshape(-1, spectra.shape[-1])

#     abs_spectra = np.moveaxis(abs_spectra, 0, -1)
#     abs_spectra = abs_spectra.reshape(-1, abs_spectra.shape[-1])
    
#     if show_example:
#         import matplotlib.pyplot as plt
#         random_indices = np.random.choice(spectra.shape[0], size=5, replace=False)
#         plt.figure(figsize=(10, 6))
#         for i in random_indices:
#             plt.plot(spectra[i].T, label=f'Line of sight {i}')
#         plt.xlabel('Channel')
#         plt.ylabel(r'$T_B$ [K]')
#         plt.title('Example Spectra')
#         plt.legend()
#         plt.show()

#         plt.figure(figsize=(10, 6))
#         for i in random_indices:
#             plt.plot(abs_spectra[i].T, label=f'Line of sight {i}')
#         plt.xlabel('Channel')
#         plt.ylabel(r'$abs$ [K]')
#         plt.title('Example Spectra')
#         plt.legend()
#         plt.show()

#     # Remove lines of sight with NaNs or all zeros
#     nanIndices = np.isnan(spectra).any(axis=1) | np.all(spectra == 0, axis=1)
#     spectra = spectra[~nanIndices]
#     abs_spectra = abs_spectra[~nanIndices]
#     los_removed = np.sum(nanIndices)
        
#     print(f'Removed {los_removed} lines of sight with NaNs')
    
#     # Split the data into training, validation, and testing sets
#     train_pct = int((1 - test_size - val_size) * 100)
#     val_pct = int(val_size * 100)
#     test_pct = int(test_size * 100)
#     print(f'Splitting data into train, validation, and test sets with sizes: train={train_pct}%, val={val_pct}%, test={test_pct}%')
#     X_train, X_temp, y_train, y_temp = train_test_split(spectra, abs_spectra, test_size=test_size + val_size, random_state=random_state)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state)
    
#     train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
#     val_dataset = TensorDataset(Tensor(X_val), Tensor(y_val))
#     test_dataset = TensorDataset(Tensor(X_test), Tensor(y_test))
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def load_tigress_temp_data(data_path='/scratch/mk27/em8117/R8_2pc/', sim_number = 'all', batch_size=32, num_workers=4, noise=0.5, test_size = 0.2, val_size = 0.2, random_state=42, show_example=False):
    if sim_number == 'all':
        sim_number = np.arange(290, 391, 10)
        # Remove 340 from the list while it's broken
        sim_number = np.delete(sim_number, np.where(sim_number == 340))
    elif sim_number == '340':
        raise NotImplementedError("Simulation 340 is currently broken and cannot be loaded.")
    else:
        sim_number = [int(sim_number)]
        
    all_spectra = []
    all_temp = []
        
    for sim in sim_number:
        print(f'Loading data for simulation {sim}')
        
        spectra = fits.getdata(f'/scratch/mk27/em8117/R8_2pc/0{sim}/{sim}_Tb_full.fits')[:,1600:1950,:]
        temp = np.load(f'/scratch/mk27/em8117/R8_2pc/0{sim}/{sim}_temperature.npy')[1600:1950,:,:]
        
        all_spectra.append(spectra)
        all_temp.append(temp)
    
    spectra = np.concatenate(all_spectra, axis=1) # Concatenate along the z-axis
    temp = np.concatenate(all_temp, axis=1)
    
    print(f'Spectra shape: {spectra.shape}, Temperature shape: {temp.shape}')
    
    spectra += np.random.randn(*spectra.shape) * noise

    # Change from (v, z, x) to (x*z, v)
    spectra = np.moveaxis(spectra, 0, -1)
    spectra = spectra.reshape(-1, spectra.shape[-1])

    # Change from (z, y, x) to (x*z, y)
    temp = np.moveaxis(temp, 1, -1)
    temp = temp.reshape(-1, temp.shape[-1])
    print(f'Spectra reshaped to: {spectra.shape}, Temperature reshaped to: {temp.shape}')
    
    
    if show_example:
        random_indices = np.random.choice(spectra.shape[0], size=5, replace=False)
        plt.figure(figsize=(10, 6))
        for i in random_indices:
            plt.plot(spectra[i].T, label=f'Line of sight {i}')
        plt.xlabel('Channel')
        plt.ylabel(r'$T_B$ [K]')
        plt.title('Example Spectra')
        plt.legend()
        plt.show()
        plt.figure(figsize=(10, 6))
        for i in random_indices:
            plt.step(np.arange(temp.shape[1]), temp[i], label=f'Line of sight {i}')
        plt.xlabel('Channel')
        plt.ylabel(r'$T$ [K]')
        plt.title('Example Temperature')
        plt.legend()
        plt.show()
        
    # Remove lines of sight with NaNs or all zeros
    nanIndices = np.isnan(spectra).any(axis=1) | np.all(spectra == 0, axis=1)
    spectra = spectra[~nanIndices]
    temp = temp[~nanIndices]
    los_removed = np.sum(nanIndices)
    
    print(f'Removed {los_removed} lines of sight with NaNs')
    
    # Split the data into training, validation, and testing sets
    train_pct = int((1 - test_size - val_size) * 100)
    val_pct = int(val_size * 100)
    test_pct = int(test_size * 100)
    
    print(f'Splitting data into train, validation, and test sets with sizes: train={train_pct}%, val={val_pct}%, test={test_pct}%')
    X_train, X_temp, y_train, y_temp = train_test_split(spectra, temp, test_size=test_size + val_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state)
    
    train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
    val_dataset = TensorDataset(Tensor(X_val), Tensor(y_val))
    test_dataset = TensorDataset(Tensor(X_test), Tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader