import numpy as np
from astropy.io import fits
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

def load_tigress_data(data_path='/scratch/mk27/em8117/R8_2pc/', sim_number = 'all', batch_size=32, num_workers=4, noise=0.5, test_size = 0.2, val_size = 0.2, random_state=42, show_example=False):
    if sim_number == 'all':
        sim_number = np.arange(290, 391, 10)
        # Remove 340 from the list while it's broken
        sim_number = np.delete(sim_number, np.where(sim_number == 340))
    elif sim_number == '340':
        raise NotImplementedError("Simulation 340 is currently broken and cannot be loaded.")
    else:
        sim_number = [int(sim_number)]
        
    all_spectra = []
    all_fcnm = []
    all_funm = []
    all_fwnm = []
    all_rhi = []
        
    for sim in sim_number:
        print(f'Loading data for simulation {sim}')
        
        spectra = fits.getdata(f'/scratch/mk27/em8117/R8_2pc/0{sim}/{sim}_Tb_full.fits')[:,1600:1950,:]
        fcnm = fits.getdata(f'/scratch/mk27/em8117/R8_2pc/0{sim}/{sim}_fcnm_full.fits')[1600:1950,:]
        funm = fits.getdata(f'/scratch/mk27/em8117/R8_2pc/0{sim}/{sim}_funm_full.fits')[1600:1950,:]
        fwnm = fits.getdata(f'/scratch/mk27/em8117/R8_2pc/0{sim}/{sim}_fwnm_full.fits')[1600:1950,:]
        rhi = fits.getdata(f'/scratch/mk27/em8117/R8_2pc/0{sim}/{sim}_rhi_full.fits')[1600:1950,:]
        
        all_spectra.append(spectra)
        all_fcnm.append(fcnm)
        all_funm.append(funm)
        all_fwnm.append(fwnm)
        all_rhi.append(rhi)
    
    spectra = np.concatenate(all_spectra, axis=1) # Concatenate along the z-axis
    fcnm = np.concatenate(all_fcnm, axis=0)
    funm = np.concatenate(all_funm, axis=0)
    fwnm = np.concatenate(all_fwnm, axis=0)
    rhi = np.concatenate(all_rhi, axis=0)
    
    spectra += np.random.randn(*spectra.shape) * noise

    fcnm = fcnm.flatten()
    funm = funm.flatten()
    fwnm = fwnm.flatten()
    rhi = rhi.flatten()

    # Change from (v, z, x) to (x*z, v)
    spectra = np.moveaxis(spectra, 0, -1)
    spectra = spectra.reshape(-1, spectra.shape[-1])
    
    if show_example:
        import matplotlib.pyplot as plt
        random_indices = np.random.choice(spectra.shape[0], size=5, replace=False)
        plt.figure(figsize=(10, 6))
        for i in random_indices:
            plt.plot(spectra[i].T, label=f'Line of sight {i}')
        plt.xlabel('Channel')
        plt.ylabel(r'$T_B$ [K]')
        plt.title('Example Spectra')
        plt.legend()
        plt.show()

    # Remove lines of sight with NaNs or all zeros
    nanIndices = np.isnan(spectra).any(axis=1) | np.isnan(fcnm) | np.isnan(funm) | np.isnan(fwnm) | np.isnan(rhi) | np.all(spectra == 0, axis=1)
    spectra = spectra[~nanIndices]
    fcnm = fcnm[~nanIndices]
    funm = funm[~nanIndices]
    fwnm = fwnm[~nanIndices]
    rhi = rhi[~nanIndices]
    los_removed = np.sum(nanIndices)
        
    print(f'Removed {los_removed} lines of sight with NaNs')
    
    targets = np.stack((fcnm, funm, fwnm, rhi), axis=1)
    
    # Split the data into training, validation, and testing sets
    train_pct = int((1 - test_size - val_size) * 100)
    val_pct = int(val_size * 100)
    test_pct = int(test_size * 100)
    print(f'Splitting data into train, validation, and test sets with sizes: train={train_pct}%, val={val_pct}%, test={test_pct}%')
    X_train, X_temp, y_train, y_temp = train_test_split(spectra, targets, test_size=test_size + val_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state)
    
    train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
    val_dataset = TensorDataset(Tensor(X_val), Tensor(y_val))
    test_dataset = TensorDataset(Tensor(X_test), Tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

def load_saury_data(data_path='/scratch/fd08/em8117/MLdata/'):
    spectra = fits.getdata(data_path + 'Tb_saury_512_thick_sector_1.fits')
    fcnm = fits.getdata(data_path + 'fCNM_saury_512_thick_sector_1.fits')
    funm = fits.getdata(data_path + 'fUNM_saury_512_thick_sector_1.fits')
    fwnm = fits.getdata(data_path + 'fWNM_saury_512_thick_sector_1.fits')


def load_tigress_abs_data(data_path='/scratch/mk27/em8117/R8_2pc/', sim_number = 'all', batch_size=32, num_workers=4, noise=0.5, test_size = 0.2, val_size = 0.2, random_state=42, show_example=False):
    if sim_number == 'all':
        sim_number = np.arange(290, 391, 10)
        # Remove 340 from the list while it's broken
        sim_number = np.delete(sim_number, np.where(sim_number == 340))
    elif sim_number == '340':
        raise NotImplementedError("Simulation 340 is currently broken and cannot be loaded.")
    else:
        sim_number = [int(sim_number)]
        
    all_spectra = []
    all_absorption = []
        
    for sim in sim_number:
        print(f'Loading data for simulation {sim}')
        
        spectra = fits.getdata(f'/scratch/mk27/em8117/R8_2pc/0{sim}/{sim}_Tb_full.fits')[:,1600:1950,:]
        absorption = fits.getdata(f'/scratch/mk27/em8117/R8_2pc/0{sim}/{sim}_Tau_full.fits')[:,1600:1950,:]
        
        all_spectra.append(spectra)
        all_absorption.append(absorption)
    
    spectra = np.concatenate(all_spectra, axis=1) # Concatenate along the z-axis
    abs_spectra = np.concatenate(all_absorption, axis=1)
    
    spectra += np.random.randn(*spectra.shape) * noise

    # Change from (v, z, x) to (x*z, v)
    spectra = np.moveaxis(spectra, 0, -1)
    spectra = spectra.reshape(-1, spectra.shape[-1])

    abs_spectra = np.moveaxis(abs_spectra, 0, -1)
    abs_spectra = abs_spectra.reshape(-1, abs_spectra.shape[-1])
    
    if show_example:
        import matplotlib.pyplot as plt
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
            plt.plot(abs_spectra[i].T, label=f'Line of sight {i}')
        plt.xlabel('Channel')
        plt.ylabel(r'$abs$ [K]')
        plt.title('Example Spectra')
        plt.legend()
        plt.show()

    # Remove lines of sight with NaNs or all zeros
    nanIndices = np.isnan(spectra).any(axis=1) | np.all(spectra == 0, axis=1)
    spectra = spectra[~nanIndices]
    abs_spectra = abs_spectra[~nanIndices]
    los_removed = np.sum(nanIndices)
        
    print(f'Removed {los_removed} lines of sight with NaNs')
    
    # Split the data into training, validation, and testing sets
    train_pct = int((1 - test_size - val_size) * 100)
    val_pct = int(val_size * 100)
    test_pct = int(test_size * 100)
    print(f'Splitting data into train, validation, and test sets with sizes: train={train_pct}%, val={val_pct}%, test={test_pct}%')
    X_train, X_temp, y_train, y_temp = train_test_split(spectra, abs_spectra, test_size=test_size + val_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state)
    
    train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
    val_dataset = TensorDataset(Tensor(X_val), Tensor(y_val))
    test_dataset = TensorDataset(Tensor(X_test), Tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader