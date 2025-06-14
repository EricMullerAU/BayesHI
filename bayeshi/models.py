import math
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchbnn as bnn
from tqdm import tqdm
from pathlib import Path

root_dir = Path(__file__).resolve().parent

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        # Note that we use different division terms for sine and cosine to handle the case where d_model is odd.
        sin_div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        cos_div_term = torch.exp(torch.arange(1, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * sin_div_term)
        pe[:, 0, 1::2] = torch.cos(position * cos_div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class SinusoidalPositionalEncoding(nn.Module):
    '''
    Sinusoidal positional encoding from TPCNet.
    
    Simple encoding that uses an additive sine function to encode the position of each channel of the input.
    '''
    def __init__(self, dropout: float = 0.0, max_len: int = 256):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(1, 1, 1, max_len)
        pe[:, :, :, :] = torch.sin(torch.arange(max_len).float())
        self.register_buffer('pe', pe, persistent=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(3)]
        return self.dropout(x)

class StochasticPositionalEncoding(nn.Module):
    """ Stochastic positional encoding (SPE) with Gaussian noise. """
    def __init__(self, d_model, max_len=256, std=0.02):
        super(StochasticPositionalEncoding, self).__init__()
        self.std = std
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * std)

    def forward(self, x):
        noise = torch.randn_like(self.pe[:x.size(1), :]) * self.std
        return x + (self.pe[:x.size(1), :] + noise).unsqueeze(0).to(x.device)

class earlyStopper:
    def __init__(self, patience, tol):
        self.patience = patience
        self.tol = tol
        self.counter = 0
        self.bestLoss = float('inf')

    def check(self, loss):
        if loss < self.bestLoss - self.tol:
            self.bestLoss = loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter == self.patience:
            return True
        else:
            return False

class saury_model(nn.Module):
    def __init__(self, cnnBlocks=1, kernelNumber=12, kernelWidth=51, MHANumber=4, transformerNumber=1, priorMu=0.0, priorSigma=0.01, posEncType='off', device='cpu'):
        super(saury_model, self).__init__()
        self.device = device
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        in_channels = 1
        Hout, Wout = 1, 256  # Initial input size
        for i in range(cnnBlocks):
            self.conv_layers.append(
            bnn.BayesConv2d(
                prior_mu=priorMu,
                prior_sigma=priorSigma,
                in_channels=in_channels,
                out_channels=12*(i+1),
                kernel_size=(1, kernelWidth),
                padding=(0, 0)
            )
            )
            self.conv_layers.append(nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
            self.conv_layers.append(
            bnn.BayesConv2d(
                prior_mu=priorMu,
                prior_sigma=priorSigma,
                in_channels=12*(i+1),
                out_channels=12*(i+1),
                kernel_size=(1, 7),
                padding=(0, 0)
            )
            )
            in_channels = 12*(i+1)
            Hout, Wout = self.get_output_size(Hout, Wout, k=[1, kernelWidth], s=[1, 1], p=[0, 0], d=[1, 1])
            Wout = int(Wout / 2)
            Hout, Wout = self.get_output_size(Hout, Wout, k=[1, 7], s=[1, 1], p=[0, 0], d=[1, 1])
            
        if posEncType == 'sinusoidal':
            self.positional_encoding = PositionalEncoding(d_model=kernelNumber)
        elif posEncType == 'stochastic':
            self.positional_encoding = StochasticPositionalEncoding(d_model=kernelNumber)
        else:
            self.positional_encoding = None
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=kernelNumber, nhead=MHANumber, batch_first=False),
            num_layers=transformerNumber
        )
        self.flatten = nn.Flatten()
        self.decoder = bnn.BayesLinear(prior_mu=priorMu, prior_sigma=priorSigma,
                                        in_features=kernelNumber * Wout, out_features=4)

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = F.relu(x)
        x = x.squeeze(2).permute(0, 2, 1)  # (batch, width, channels)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.flatten(x)
        x = self.decoder(x)
        x = x / 100
        x = torch.cat((F.softmax(x[:, :3], dim=1), torch.clamp(x[:, 3], min=1).unsqueeze(1)), dim=1)
        return x

    def get_output_size(self, Hin, Win, k, s=[1, 1], p=[0, 0], d=[1, 1]):
        Hout = int((Hin + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1)
        Wout = int((Win + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1)
        return Hout, Wout

    def lossFunction(self, outputs, targets, KLweight):
        MSE = nn.MSELoss()
        BKLoss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        return MSE(outputs, targets) + KLweight * BKLoss(self)
    

    def fit(self, train_loader, val_loader, checkpoint_path, nEpochs = 50, learningRate = 0.0005, schedulerStep = 15, stopperPatience = 5, stopperTol = 1e-4, maxKLweight = 0.01, maxKLepoch = 50):
        self.to(self.device)
        criterion = self.lossFunction
        optimizer = optim.Adam(self.parameters(), lr=learningRate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=schedulerStep)
        earlyStop = earlyStopper(patience=stopperPatience, tol=stopperTol)
        trainErrors = []
        valErrors = []
        epochTimes = []
        bestValLoss = float('inf')
        if checkpoint_path != None and checkpoint_path[-4:] != '.pth':
            raise ValueError("Checkpoint path must end with .pth")
        # Also check that we wont overwrite the pretrained model
        if checkpoint_path == root_dir / 'weights/saury.pth':
            raise ValueError("Checkpoint path is the same as the pretrained model. Please change the checkpoint path to avoid overwriting the pretrained model.")

        print('Training Model')
        print('Initial learning rate:', scheduler.get_last_lr())
        trainedEpochs = 0

        for epoch in range(nEpochs):
            KLweight = maxKLweight * min(1, epoch / maxKLepoch)
            startTime = time.time()
            self.train()
            runningLoss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets, KLweight)
                loss.backward()
                optimizer.step()
                runningLoss += loss.item()

            trainLoss = runningLoss / len(train_loader)
            trainErrors.append(trainLoss)
            trainedEpochs += 1
            valLoss = self.evaluate(val_loader, nn.MSELoss())
            valErrors.append(valLoss)

            if earlyStop.check(valLoss):
                print(f'Early stopping at epoch {epoch + 1}')
                break

            lastLR = scheduler.get_last_lr()
            scheduler.step(valLoss)
            if lastLR != scheduler.get_last_lr():
                print(f'Learning rate changed to {scheduler.get_last_lr()}')

            print(f'Epoch [{epoch + 1}/{nEpochs}], Train Loss: {trainLoss:.4f}, Validation Loss: {valLoss:.4f}, took {time.time() - startTime:.2f}s')
            epochTimes.append(time.time() - startTime)

            if valLoss < bestValLoss:
                bestValLoss = valLoss
                if checkpoint_path is not None:
                    torch.save(self.state_dict(), checkpoint_path)

        return trainErrors, valErrors, trainedEpochs, epochTimes

    def evaluate(self, loader, criterion):
        self.eval()
        totalLoss = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                totalLoss += loss.item()
        avgLoss = totalLoss / len(loader)
        return avgLoss

    def predict(self, test_loader, numPredictions):
        self.eval()
        allPredictions = []
        for _ in tqdm(range(numPredictions), desc='Predicting', file=sys.stdout):
            predictions = []
            with torch.no_grad():
                for inputs, *_ in test_loader:
                    inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                    outputs = self(inputs)
                    predictions.append(outputs)
            allPredictions.append(torch.cat(predictions, dim=0))
        return torch.stack(allPredictions, dim=0)

    def load_weights(self, path = root_dir / 'weights/saury.pth'):
        print(f'Loading model from {path}')
        if self.device == 'cpu':
            self.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.load_state_dict(torch.load(path))
            self.to(self.device)
        print('Model loaded successfully')
    
class TPCNetCustomLoss(nn.Module):
    def __init__(self, weights=[1., 1.]):
        super(TPCNetCustomLoss, self).__init__()
        self.weights = weights
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.npars = len(weights)

    def forward(self, preds, targets):
        total_loss = 0.
        for i in range(self.npars):
            loss_yi = self.mse(preds[:, i], targets[:, i])
            total_loss += self.weights[i] * loss_yi
        # End - for

        return total_loss

class tpcnet_all_phases(nn.Module):
    def __init__(self, num_output=4, in_channels=1, input_row=1, input_column=256, drop_out_rate=0., device='cpu'):
        super(tpcnet_all_phases, self).__init__()

        p = [0, 0] # padding
        d = [1, 1] # dilation
        k = [1, 7] # kernel size for small layers
        s = [1, 1] # stride
        
        self.device = device
        
        self.num_features = 54
        self.input_row = input_row
        self.in_channels = in_channels
        self.input_column = input_column

        kernel_wid = 33
        
        self.drop_out_rate = drop_out_rate
        # self.lpe = lpe
        self.emb_pos_encoder = PositionalEncoding(dropout = self.drop_out_rate, d_model=9, max_len=self.input_column)
        self.spec_pos_encoder = SinusoidalPositionalEncoding(dropout=self.drop_out_rate, max_len=self.input_column)
        
        self.loss_fcn = TPCNetCustomLoss(weights=[1., 1., 1., 1.])

        # num_layer*8 + 8
        
        # CNN layers (outchannels = outchannels-8)
        kernelsize = (1,7) if (input_row < 2) else (2,7)
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=72, kernel_size=kernelsize, stride=1, padding=0, bias=True, padding_mode='zeros')
        self.bn1   = nn.BatchNorm2d(72)
        Hout, Wout = self.get_output_size(72, input_column, k=kernelsize, s=s, p=p, d=d)
        # print('>>> Conv2: ', Hout, Wout)
        
        self.conv2 = nn.Conv2d(in_channels=72, out_channels=64, kernel_size=(1,kernel_wid), stride=1, padding=0, bias=True, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(64)
        Hout, Wout = self.get_output_size(64, Wout, k=(1,kernel_wid), s=s, p=p, d=d)
        # print('>>> Conv2: ', Hout, Wout)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=56, kernel_size=kernelsize, stride=1, padding=0, bias=True, padding_mode='zeros')
        self.bn3 = nn.BatchNorm2d(56)
        Hout, Wout = self.get_output_size(56, Wout, k = kernelsize, s=s, p=p, d=d)
        # print('>>> Conv2: ', Hout, Wout)
        
        self.conv4 = nn.Conv2d(in_channels=56, out_channels=48,  kernel_size=(1,kernel_wid), stride=1, padding=0, bias=True, padding_mode='zeros')
        self.bn4 = nn.BatchNorm2d(48)
        Hout, Wout = self.get_output_size(48, Wout, k = (1,kernel_wid), s=s, p=p, d=d)
        # print('>>> Conv2: ', Hout, Wout)
        
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=40,  kernel_size=kernelsize, stride=1, padding=0, bias=True, padding_mode='zeros')
        self.bn5 = nn.BatchNorm2d(40)
        Hout, Wout = self.get_output_size(40, Wout, k = kernelsize, s=s, p=p, d=d)
        # print('>>> Conv2: ', Hout, Wout)
        
        self.conv6 = nn.Conv2d(in_channels=40, out_channels=32,  kernel_size=(1,kernel_wid), stride=1, padding=0, bias=True, padding_mode='zeros')
        self.bn6 = nn.BatchNorm2d(32)
        Hout, Wout = self.get_output_size(32, Wout, k = (1,kernel_wid), s=s, p=p, d=d)
        # print('>>> Conv2: ', Hout, Wout)
        
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=16,  kernel_size=kernelsize, stride=1, padding=0,  bias=True, padding_mode='zeros')
        self.bn7 = nn.BatchNorm2d(16)
        Hout, Wout = self.get_output_size(16, Wout, k = kernelsize, s=s, p=p, d=d)
        # print('>>> Conv2: ', Hout, Wout)
        
        self.conv8 = nn.Conv2d(in_channels=16, out_channels=8,  kernel_size=(1,kernel_wid), stride=1, padding=0, bias=True, padding_mode='zeros')
        self.bn8 = nn.BatchNorm2d(8)
        Hout, Wout = self.get_output_size(8, Wout, k = (1,kernel_wid), s=s, p=p, d=d)
        # print('>>> Conv2: ', Hout, Wout)


        # self.conv9 = nn.Conv2d(in_channels=8, out_channels=4,  kernel_size=kernelsize, stride=1, padding=0, bias=True, padding_mode='zeros')
        # self.bn9 = nn.BatchNorm2d(4)
        # Hout, Wout = self.get_output_size(4, Wout, k = kernelsize, s=s, p=p, d=d)
        # # print('>>> Conv2: ', Hout, Wout)

        # self.conv10 = nn.Conv2d(in_channels=4, out_channels=2,  kernel_size=(1,kernel_wid), stride=1, padding=0, bias=True, padding_mode='zeros')
        # self.bn10 = nn.BatchNorm2d(2)
        # Hout, Wout = self.get_output_size(2, Wout, k = (1,kernel_wid), s=s, p=p, d=d)
        # print('>>> Conv2: ', Hout, Wout)

        # self.conv11 = nn.Conv2d(in_channels=8, out_channels=4,  kernel_size=(1,3), stride=1, padding=0, bias=True, padding_mode='zeros')
        # self.bn11 = nn.BatchNorm2d(4)
        # Hout, Wout = self.get_output_size(4, Wout, k = (1,3), s=s, p=p, d=d)
        # print('>>> Conv2: ', Hout, Wout)

        # if(input_row<=2):
        #     self.linear = nn.Linear(456, 54)
        # else:
        #     self.linear = nn.Linear(int(1904*(input_row-1)), 54)

        self.linear = nn.Linear(Hout*Wout, 54)
        self.flatten = nn.Flatten()
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=9,
                nhead=3,
                dim_feedforward=36,
                dropout=self.drop_out_rate,
                batch_first=True,
            ),
            num_layers=4
        )
        
        self.decoder = nn.Linear(54, num_output)

        # init parameter
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x =  self.spec_pos_encoder(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # print(2, x.size())
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # print(3, x.size())
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        # print(x.size())
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        # print(x.size())
        #
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        # print(x.size())
        #
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        # print(x.size())
        #
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        # print(x.size())
        #
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        # print(x.size())

        # x = self.conv9(x)
        # x = self.bn9(x)
        # x = F.relu(x)
        # # print(x.size())

        # x = self.conv10(x)
        # x = self.bn10(x)
        # x = F.relu(x)
        # print(10, x.size())
        #x = torch.squeeze(x)
        #x = x.reshape(-1, x.shape[2], x.shape[1])

        # x = self.conv11(x)
        # x = self.bn11(x)
        # x = F.relu(x)

        #
        #print(3, x.size()) #= (20,8,1,238)
        x = self.flatten(x) #1904
        # print('Flatten: ', x.size()) # [10, 402]

        x = self.linear(x)
        # print('linear: ', x.size()) # [10, 54]
        
        #print(3, x.size())
        x = x.reshape(x.shape[0], -1, 9)
        # print('x reshape (before trans): ', x.size())
        
        # Add positional encoding to embedding matrix
        x = self.emb_pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        x = self.flatten(x)
    
        x = self.decoder(x)
        return x

    def get_output_size(self, Hin, Win, k, s=[1, 1], p=[0, 0], d=[1, 1]):
        Hout = int((Hin + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1)
        Wout = int((Win + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1)
        return Hout, Wout

    def fit(self, train_loader, val_loader, checkpoint_path, nEpochs = 60, learningRate = 5e-3):
        self.to(self.device)
        criterion = self.loss_fcn
        optimizer = optim.Adam(self.parameters(), lr=learningRate)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[nEpochs//2], gamma=0.1, last_epoch=-1)
        # earlyStop = earlyStopper(patience=stopperPatience, tol=stopperTol)
        trainErrors = []
        valErrors = []
        epochTimes = []
        bestValLoss = float('inf')
        #TODO: save with epoch number in name for interruped training, add handling to train if not at provided epochs
        # Check if the provided path ends with .pth
        if checkpoint_path != None and checkpoint_path[-4:] != '.pth':
            raise ValueError("Checkpoint path must end with .pth")
        # Also check that we wont overwrite the pretrained model
        if checkpoint_path == root_dir / 'weights/tpcnet.pth':
            raise ValueError("Checkpoint path is the same as the pretrained model. Please change the checkpoint path to avoid overwriting the pretrained model.")

        print('Training Model')
        # print('Initial learning rate:', scheduler.get_last_lr())
        trainedEpochs = 0

        for epoch in range(nEpochs):
            startTime = time.time()
            self.train()
            runningLoss = 0.0
            mse_running_loss = 0.0
            for inputs, targets in tqdm(train_loader, file=sys.stdout, desc=f'Epoch {epoch + 1}', unit='batch'):
                inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                # Calculate MSE loss to compare with custom loss
                mse_loss = nn.MSELoss()(outputs, targets)
                loss.backward()
                optimizer.step()
                runningLoss += loss.item()
                mse_running_loss += mse_loss.item()

            trainLoss = runningLoss / len(train_loader)
            mse_trainLoss = mse_running_loss / len(train_loader)
            trainErrors.append(trainLoss)
            trainedEpochs += 1
            valLoss = self.evaluate(val_loader, nn.MSELoss())
            valErrors.append(valLoss)
            
            if trainLoss == np.NaN or valLoss == np.NaN:
                print('NaN loss detected, stopping training')
                break
            
            # if earlyStop.check(valLoss):
            #     print(f'Early stopping at epoch {epoch + 1}')
            #     break

            # lastLR = scheduler.get_last_lr()
            # scheduler.step()
            # if lastLR != scheduler.get_last_lr():
            #     print(f'Learning rate changed to {scheduler.get_last_lr()}')

            print(f'Epoch [{epoch + 1}/{nEpochs}], Train Loss: {trainLoss:.4f}, MSE Train Loss: {mse_trainLoss:.4f}, Validation Loss: {valLoss:.4f}, took {time.time() - startTime:.2f}s')
            epochTimes.append(time.time() - startTime)

            # Lmao this saves every epoch instead of every run at the moment. Oops.
            if valLoss < bestValLoss:
                bestValLoss = valLoss
                if checkpoint_path is not None:
                    torch.save(self.state_dict(), checkpoint_path)

        return trainErrors, valErrors, trainedEpochs, epochTimes

    def evaluate(self, loader, criterion):
        self.eval()
        totalLoss = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                totalLoss += loss.item()
        avgLoss = totalLoss / len(loader)
        return avgLoss

    def predict(self, test_loader, numPredictions):
        if numPredictions  > 1:
            print('Warning: numPredictions > 1, but this model is not Bayesian, so all predictions will be the same. Re-run with numPredictions = 1 for faster inference.')
        self.eval()
        allPredictions = []
        for _ in tqdm(range(numPredictions), file=sys.stdout, desc='Predicting'):
            predictions = []
            with torch.no_grad():
                for inputs, *_ in test_loader:
                    inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                    outputs = self(inputs)
                    predictions.append(outputs)
            allPredictions.append(torch.cat(predictions, dim=0))
        return torch.stack(allPredictions, dim=0)

    def load_weights(self, path = root_dir / 'weights/tpcnet.pth'):
        print(f'Loading model from {path}')
        if self.device == 'cpu':
            self.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.load_state_dict(torch.load(path))
            self.to(self.device)
        print('Model loaded successfully')

class bayeshi_model(nn.Module):
    def __init__(self, cnnBlocks: int = 1, kernelNumber: int = 8, kernelWidth1: int = 31, kernelWidth2: int = 3, kernelMult: float = 2.0, pooling: str = 'off', MHANumber: int = 4, transformerNumber: int = 1, priorMu: float = 0.0, priorSigma: float = 0.01, posEncType: str = 'sinusoidal', device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.pooling = pooling
        if pooling not in ['max', 'avg', 'off']:
            raise ValueError("Pooling must be 'max', 'avg', or 'off'.")
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        # self.pool_layers = nn.ModuleList()
        in_channels = 1
        Hout, Wout = 1, 256  # Initial input size
        for i in range(cnnBlocks):
            # Check if the width of the kernel is larger than the input size
            #TODO: this doesn't catch the negative dimension tensor error
            if Wout < kernelWidth1 or self.get_output_size(Hout, Wout, k=[1, kernelWidth1], s=[1, 1], p=[0, 0], d=[1, 1])[1] < kernelWidth2:
                print('Dimensions too small for kernel size in kernel block', i, '. Ignoring block and continuing...')
                cnnBlocks = i
                break
            
            if i == 0:
                out_channels = kernelNumber
            else:
                out_channels = int(in_channels * kernelMult)
            print(f'Convolutional block {i + 1}: in_channels={in_channels}, out_channels={out_channels}, kernelWidth1={kernelWidth1}, kernelWidth2={kernelWidth2}, pooling={pooling}')
            self.conv_layers.append(
            bnn.BayesConv2d(
                prior_mu=priorMu,
                prior_sigma=priorSigma,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, kernelWidth1),
                padding=(0, 0)
            )
            )
            if self.pooling == 'max':
                self.conv_layers.append(nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
            elif self.pooling == 'avg':
                self.conv_layers.append(nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))
            self.conv_layers.append(
            bnn.BayesConv2d(
                prior_mu=priorMu,
                prior_sigma=priorSigma,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, kernelWidth2),
                padding=(0, 0)
            )
            )
            if self.pooling == 'max':
                self.conv_layers.append(nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
            elif self.pooling == 'avg':
                self.conv_layers.append(nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))
            
            in_channels = out_channels
            Hout, Wout = self.get_output_size(Hout, Wout, k=[1, kernelWidth1], s=[1, 1], p=[0, 0], d=[1, 1])
            if self.pooling != 'off':
                Wout = Wout // 2
            Hout, Wout = self.get_output_size(Hout, Wout, k=[1, kernelWidth2], s=[1, 1], p=[0, 0], d=[1, 1])
            if self.pooling != 'off':
                Wout = Wout // 2
            
        if posEncType == 'sinusoidal':
            self.positional_encoding = PositionalEncoding(d_model = out_channels)
        elif posEncType == 'stochastic':
            self.positional_encoding = StochasticPositionalEncoding(d_model = out_channels)
        else:
            self.positional_encoding = None
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = out_channels, nhead=MHANumber, batch_first=True),
            num_layers=transformerNumber
        )
        self.flatten = nn.Flatten()
        self.decoder = bnn.BayesLinear(prior_mu=priorMu, prior_sigma=priorSigma,
                                        in_features=out_channels * Wout, out_features=4)
        
        # Initialise weights using Kaiming normal initialization
        # Note this DOES NOT initialise the transformer layers
        for layer in self.conv_layers:
            if isinstance(layer, bnn.BayesConv2d):
                nn.init.kaiming_normal_(layer.weight_mu)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias_mu)
        
        nn.init.kaiming_normal_(self.decoder.weight_mu)
        if self.decoder.bias_mu is not None:
            nn.init.zeros_(self.decoder.bias_mu)
        

    def forward(self, x):
        for layer in self.conv_layers:
            if isinstance(layer, bnn.BayesConv2d):
                # Apply convolutional layer
                x = layer(x)
                if self.pooling == 'off':
                    x = F.relu(x)
            elif isinstance(layer, nn.MaxPool2d) or isinstance(conv, nn.AvgPool2d):
                # Apply pooling layer and then ReLU activation just for a bit more efficiency (can also do ReLu then pooling but this means half the activations are not used)
                x = layer(x)
                x = F.relu(x)
        # Once finished with the convolutions pass to the transformer
        # x = x.squeeze(2).permute(0, 2, 1)  # change from (batch, channels, height, width) to (batch, width, channels)
        x = x.squeeze(2).permute(2, 0, 1)  # change from (batch, channels, height, width) to (width, batch, channels)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        x = self.transformer(x)
        # Now reshape back to (batch, width * channels)
        x = x.permute(1, 0, 2)  # change from (width, batch, channels) to (batch, width, channels)
        # and flatten
        x = self.flatten(x)
        x = self.decoder(x)
        x = x / 100 # this fixes the issue of exploding values ruining the softmax, likely caused by the new Kaiming initialisation.
        x = torch.cat((F.softmax(x[:, :3], dim=1), torch.clamp(x[:, 3], min=1).unsqueeze(1)), dim=1)
        return x

    def get_output_size(self, Hin, Win, k, s=[1, 1], p=[0, 0], d=[1, 1]):
        Hout = int((Hin + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1)
        Wout = int((Win + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1)
        return Hout, Wout

    def lossFunction(self, outputs, targets, KLweight):
        MSE = nn.MSELoss()
        BKLoss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        return MSE(outputs, targets) + KLweight * BKLoss(self)

    def fit(self, train_loader, val_loader, checkpoint_path, nEpochs = 100, learningRate = 0.001, schedulerStep = 15, stopperPatience = 20, stopperTol = 0.0001, maxKLweight = 0.01, maxKLepoch = 100, early_stop = False):
        
        self.to(self.device)
        # Track a single set of weights through the training process
        criterion = self.lossFunction
        optimizer = optim.Adam(self.parameters(), lr=learningRate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=schedulerStep)
        if early_stop:
            earlyStop = earlyStopper(patience=stopperPatience, tol=stopperTol)
        trainErrors = []
        valErrors = []
        epochTimes = []
        bestValLoss = float('inf')
        #TODO: save with epoch number in name for interruped training, add handling to train if not at provided epochs
        # Check if the provided path ends with .pth
        if checkpoint_path != None and checkpoint_path[-4:] != '.pth':
            raise ValueError("Checkpoint path must end with .pth")
        # Also check that we wont overwrite the pretrained model
        if checkpoint_path == root_dir / 'weights/tigress.pth':
            raise ValueError("Checkpoint path is the same as the pretrained model. Please change the checkpoint path to avoid overwriting the pretrained model.")
        
        print('Training Model')
        print('Initial learning rate:', scheduler.get_last_lr())
        trainedEpochs = 0

        # import pdb
        for p in self.parameters():
            p.requires_grad = True  # Ensure all parameters are trainable
        for epoch in range(nEpochs):
            self.train()
            KLweight = maxKLweight * min(1, epoch / maxKLepoch)
            startTime = time.time()
            runningLoss = 0.0
            # asdf = 0
            for inputs, targets in tqdm(train_loader, file=sys.stdout, desc=f'Epoch {epoch + 1}', unit='batch'):
                # asdf += 1
                # if asdf > 10:
                #     break
                inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                # print('Outputs:', outputs[0])
                # print('Targets:', targets[0])
                loss = criterion(outputs, targets, KLweight)
                # print some gradients for debugging
                # print('Gradients:', [p.grad for p in self.parameters() if p.grad is not None][:5])  # Print first 5 gradients
                # print('Loss:', loss.item())
                loss.backward()
                optimizer.step()
                runningLoss += loss.item()
                # pdb.set_trace()

            # for i, p in enumerate(self.parameters()):
            #     print(i, p.sum())

            trainLoss = runningLoss / len(train_loader)
            trainErrors.append(trainLoss)
            trainedEpochs += 1
            valLoss = self.evaluate(val_loader)
            valErrors.append(valLoss)
    
            if early_stop and earlyStop.check(valLoss):
                print(f'Early stopping at epoch {epoch + 1}')
                break

            lastLR = scheduler.get_last_lr()
            scheduler.step(valLoss)
            if lastLR != scheduler.get_last_lr():
                print(f'Learning rate changed to {scheduler.get_last_lr()}')

            print(f'Epoch [{epoch + 1}/{nEpochs}], Train Loss: {trainLoss:.4f}, Validation Loss: {valLoss:.4f}, took {time.time() - startTime:.2f}s')
            epochTimes.append(time.time() - startTime)

            # Lmao this saves every epoch instead of every run at the moment. Oops.
            if valLoss < bestValLoss:
                bestValLoss = valLoss
                if checkpoint_path is not None:
                    torch.save(self.state_dict(), checkpoint_path)

        return trainErrors, valErrors, trainedEpochs, epochTimes

    def evaluate(self, loader, criterion=None):
        criterion = nn.MSELoss() if criterion is None else criterion
        self.eval()
        totalLoss = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                totalLoss += loss.item()
        avgLoss = totalLoss / len(loader)
        return avgLoss

    def predict(self, test_loader, numPredictions):
        self.eval()
        allPredictions = []
        for _ in tqdm(range(numPredictions), file=sys.stdout, desc='Predicting'):
            predictions = []
            with torch.no_grad():
                for inputs, *_ in test_loader:
                    inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                    outputs = self(inputs)
                    predictions.append(outputs)
            allPredictions.append(torch.cat(predictions, dim=0))
        return torch.stack(allPredictions, dim=0)

    def load_weights(self, path = root_dir / 'weights/tigress.pth'):
        print(f'Loading model from {path}')
        if self.device == 'cpu':
            self.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.load_state_dict(torch.load(path))
            self.to(self.device)
        print('Model loaded successfully')

class rnn_model(nn.Module):
    def __init__(self, input_dim=1, seq_len=256, d_model=128, nhead=4, num_layers=4,
                 output_dim=4, device='cpu'):
        super().__init__()
        self.device = torch.device(device)

        # Embed scalar inputs to d_model dimension
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional embeddings and CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model, device=self.device))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model, device=self.device))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.head = nn.Linear(d_model, output_dim)

        # Move submodules to device
        self.to(self.device)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len) = (B, 1, 256)
        """
        x = x.to(self.device)

        B, H, L = x.shape
        
        assert H == 1 and L == 256, f"Input tensor must have shape (B, 1, 256), got {x.shape}"
        
        x = x.view(B, L).unsqueeze(-1) # (B, 256, 1)

        x = self.embedding(x)  # (B, 256, d_model)

        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_token, x], dim=1)  # (B, 257, d_model)
        x = x + self.pos_embedding[:, :x.size(1), :]

        x = self.transformer(x)  # (B, 257, d_model)

        cls_out = x[:, 0, :]  # (B, d_model)
        output = self.head(cls_out)  # (B, 4)
        return output
    
    def fit(self, train_loader, val_loader, checkpoint_path, nEpochs=100, learningRate=0.001, schedulerStep=15, stopperPatience=20, stopperTol=0.0001):
        self.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learningRate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=schedulerStep)
        # earlyStop = earlyStopper(patience=stopperPatience, tol=stopperTol)
        trainErrors = []
        valErrors = []
        epochTimes = []
        bestValLoss = float('inf')
        
        if checkpoint_path != None and checkpoint_path[-4:] != '.pth':
            raise ValueError("Checkpoint path must end with .pth")
        
        print('Training Model')
        print('Initial learning rate:', scheduler.get_last_lr())
        trainedEpochs = 0

        for epoch in range(nEpochs):
            self.train()
            startTime = time.time()
            runningLoss = 0.0
            for inputs, targets in tqdm(train_loader, file=sys.stdout, desc=f'Epoch {epoch + 1}', unit='batch'):
                inputs = inputs.unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                runningLoss += loss.item()
            trainLoss = runningLoss / len(train_loader)
            trainErrors.append(trainLoss)
            trainedEpochs += 1
            valLoss = self.evaluate(val_loader, criterion)
            valErrors.append(valLoss)
            
            # if earlyStop.check(valLoss):
            #     print(f'Early stopping at epoch {epoch + 1}')
            #     break
            
            lastLR = scheduler.get_last_lr()
            scheduler.step(valLoss)
            if lastLR != scheduler.get_last_lr():
                print(f'Learning rate changed to {scheduler.get_last_lr()}')
                
            print(f'Epoch [{epoch + 1}/{nEpochs}], Train Loss: {trainLoss:.4f}, Validation Loss: {valLoss:.4f}, took {time.time() - startTime:.2f}s')
            epochTimes.append(time.time() - startTime)
            
            if valLoss < bestValLoss:
                bestValLoss = valLoss
                if checkpoint_path is not None:
                    torch.save(self.state_dict(), checkpoint_path)

        return trainErrors, valErrors, trainedEpochs, epochTimes
    
    def evaluate(self, loader, criterion=nn.MSELoss()):
        self.eval()
        totalLoss = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                totalLoss += loss.item()
        avgLoss = totalLoss / len(loader)
        return avgLoss
    
    def predict(self, test_loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for inputs, *_ in test_loader:
                inputs = inputs.unsqueeze(1).to(self.device)
                outputs = self(inputs)
                predictions.append(outputs)
        return torch.cat(predictions, dim=0)

    def load_weights(self, path):
        print(f'Loading model from {path}')
        if self.device == 'cpu':
            self.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.load_state_dict(torch.load(path))
            self.to(self.device)
        print('Model loaded successfully')

class LSTMSequencePredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, output_dim=4, device='cpu', aggregation='mean'):
        super().__init__()
        self.device = torch.device(device)
        self.aggregation = aggregation

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

        self.to(self.device)

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, 256)
        """
        x = x.to(self.device)
        B, H, L = x.shape
        assert H == 1, f"Expected input shape (B, 1, L), got {x.shape}"

        x = x.view(B, L).unsqueeze(-1)  # (B, L, 1)
        x = self.embedding(x)  # (B, L, hidden_dim)

        x, _ = self.lstm(x)  # (B, L, hidden_dim)
        tokenwise_outputs = self.head(x)  # (B, L, 4)

        if self.aggregation == 'mean':
            output = tokenwise_outputs.mean(dim=1)  # (B, 4)
        elif self.aggregation == 'last':
            output = tokenwise_outputs[:, -1, :]  # (B, 4)
        elif self.aggregation == 'max':
            output, _ = tokenwise_outputs.max(dim=1)  # (B, 4)
        else:
            raise ValueError(f"Unsupported aggregation mode: {self.aggregation}")

        return output, tokenwise_outputs  # (B, 4), (B, 256, 4)
    
    def fit(self, train_loader, val_loader, checkpoint_path, nEpochs=100, learningRate=0.001, schedulerStep=15, stopperPatience=20, stopperTol=0.0001):
        self.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learningRate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=schedulerStep)
        # earlyStop = earlyStopper(patience=stopperPatience, tol=stopperTol)
        trainErrors = []
        valErrors = []
        epochTimes = []
        bestValLoss = float('inf')
        
        if checkpoint_path != None and checkpoint_path[-4:] != '.pth':
            raise ValueError("Checkpoint path must end with .pth")
        
        print('Training Model')
        print('Initial learning rate:', scheduler.get_last_lr())
        trainedEpochs = 0

        for epoch in range(nEpochs):
            self.train()
            startTime = time.time()
            runningLoss = 0.0
            for inputs, targets in tqdm(train_loader, file=sys.stdout, desc=f'Epoch {epoch + 1}', unit='batch'):
                inputs = inputs.unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs, _ = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                runningLoss += loss.item()
            trainLoss = runningLoss / len(train_loader)
            trainErrors.append(trainLoss)
            trainedEpochs += 1
            valLoss = self.evaluate(val_loader, criterion)
            valErrors.append(valLoss)
            
            # if earlyStop.check(valLoss):
            #     print(f'Early stopping at epoch {epoch + 1}')
            #     break
            
            lastLR = scheduler.get_last_lr()
            scheduler.step(valLoss)
            if lastLR != scheduler.get_last_lr():
                print(f'Learning rate changed to {scheduler.get_last_lr()}')
                
            print(f'Epoch [{epoch + 1}/{nEpochs}], Train Loss: {trainLoss:.4f}, Validation Loss: {valLoss:.4f}, took {time.time() - startTime:.2f}s')
            epochTimes.append(time.time() - startTime)
            
            if valLoss < bestValLoss:
                bestValLoss = valLoss
                if checkpoint_path is not None:
                    torch.save(self.state_dict(), checkpoint_path)

        return trainErrors, valErrors, trainedEpochs, epochTimes
    
    def evaluate(self, loader, criterion=nn.MSELoss()):
        self.eval()
        totalLoss = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                outputs, _ = self(inputs)
                loss = criterion(outputs, targets)
                totalLoss += loss.item()
        avgLoss = totalLoss / len(loader)
        return avgLoss
    
    def predict(self, test_loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for inputs, *_ in test_loader:
                inputs = inputs.unsqueeze(1).to(self.device)
                outputs, _ = self(inputs)
                predictions.append(outputs)
        return torch.cat(predictions, dim=0)

    def load_weights(self, path):
        print(f'Loading model from {path}')
        if self.device == 'cpu':
            self.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.load_state_dict(torch.load(path))
            self.to(self.device)
        print('Model loaded successfully')

class LSTMSequenceToSequence(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, output_dim=1, device='cpu'):
        super().__init__()
        self.device = torch.device(device)

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

        self.to(self.device)

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, 256)
        Returns:
            output: (B, 256) or (B, 256, output_dim) depending on output_dim
        """
        x = x.to(self.device)
        B, H, L = x.shape
        assert H == 1, f"Expected input shape (B, 1, L), got {x.shape}"

        x = x.view(B, L).unsqueeze(-1)       # (B, L, 1)
        x = self.embedding(x)                # (B, L, hidden_dim)
        x, _ = self.lstm(x)                  # (B, L, hidden_dim)
        x = self.head(x)                     # (B, L, output_dim)

        if x.shape[-1] == 1:
            x = x.squeeze(-1)                # (B, L)

        return x  # (B, 256) or (B, 256, output_dim)

    def fit(self, train_loader, val_loader, checkpoint_path, nEpochs=100, learningRate=0.001, schedulerStep=15, stopperPatience=20, stopperTol=0.0001):
        self.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learningRate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=schedulerStep)
        # earlyStop = earlyStopper(patience=stopperPatience, tol=stopperTol)
        trainErrors = []
        valErrors = []
        epochTimes = []
        bestValLoss = float('inf')
        
        if checkpoint_path != None and checkpoint_path[-4:] != '.pth':
            raise ValueError("Checkpoint path must end with .pth")
        
        print('Training Model')
        print('Initial learning rate:', scheduler.get_last_lr())
        trainedEpochs = 0

        for epoch in range(nEpochs):
            self.train()
            startTime = time.time()
            runningLoss = 0.0
            for inputs, targets in tqdm(train_loader, file=sys.stdout, desc=f'Epoch {epoch + 1}', unit='batch'):
                inputs = inputs.unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                runningLoss += loss.item()
            trainLoss = runningLoss / len(train_loader)
            trainErrors.append(trainLoss)
            trainedEpochs += 1
            valLoss = self.evaluate(val_loader, criterion)
            valErrors.append(valLoss)
            
            # if earlyStop.check(valLoss):
            #     print(f'Early stopping at epoch {epoch + 1}')
            #     break
            
            lastLR = scheduler.get_last_lr()
            scheduler.step(valLoss)
            if lastLR != scheduler.get_last_lr():
                print(f'Learning rate changed to {scheduler.get_last_lr()}')
                
            print(f'Epoch [{epoch + 1}/{nEpochs}], Train Loss: {trainLoss:.4f}, Validation Loss: {valLoss:.4f}, took {time.time() - startTime:.2f}s')
            epochTimes.append(time.time() - startTime)
            
            if valLoss < bestValLoss:
                bestValLoss = valLoss
                if checkpoint_path is not None:
                    torch.save(self.state_dict(), checkpoint_path)

        return trainErrors, valErrors, trainedEpochs, epochTimes
    
    def evaluate(self, loader, criterion=nn.MSELoss()):
        self.eval()
        totalLoss = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                totalLoss += loss.item()
        avgLoss = totalLoss / len(loader)
        return avgLoss
    
    def predict(self, test_loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for inputs, *_ in test_loader:
                inputs = inputs.unsqueeze(1).to(self.device)
                outputs = self(inputs)
                predictions.append(outputs)
        return torch.cat(predictions, dim=0)

    def load_weights(self, path):
        print(f'Loading model from {path}')
        if self.device == 'cpu':
            self.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.load_state_dict(torch.load(path))
            self.to(self.device)
        print('Model loaded successfully')


class TransformerWithAttentionAggregation(nn.Module):
    def __init__(self, input_dim=1, seq_len=256, d_model=128, nhead=4, num_layers=4, output_dim=4, device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.seq_len = seq_len

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model, device=self.device))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(d_model, output_dim)

        # Attention weights: from d_model to scalar weight per token
        self.attention_fc = nn.Linear(d_model, 1)

        self.to(self.device)

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, 256)
        """
        x = x.to(self.device)
        B, H, W = x.shape
        assert H == 1 and W == self.seq_len, f"Expected input shape (B, 1, {self.seq_len}), got {x.shape}"

        x = x.view(B, W).unsqueeze(-1)  # (B, 256, 1)
        x = self.embedding(x) + self.pos_embedding  # (B, 256, d_model)

        x = self.transformer(x)  # (B, 256, d_model)
        tokenwise_outputs = self.head(x)  # (B, 256, 4)

        attn_weights = self.attention_fc(x)  # (B, 256, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (B, 256, 1)

        weighted_output = (tokenwise_outputs * attn_weights).sum(dim=1)  # (B, 4)

        return weighted_output, tokenwise_outputs, attn_weights  # (B, 4), (B, 256, 4), (B, 256, 1)
    
    def fit(self, train_loader, val_loader, checkpoint_path, nEpochs=100, learningRate=0.001, schedulerStep=15, stopperPatience=20, stopperTol=0.0001):
        self.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learningRate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=schedulerStep)
        # earlyStop = earlyStopper(patience=stopperPatience, tol=stopperTol)
        trainErrors = []
        valErrors = []
        epochTimes = []
        bestValLoss = float('inf')
        
        if checkpoint_path != None and checkpoint_path[-4:] != '.pth':
            raise ValueError("Checkpoint path must end with .pth")
        
        print('Training Model')
        print('Initial learning rate:', scheduler.get_last_lr())
        trainedEpochs = 0

        for epoch in range(nEpochs):
            self.train()
            startTime = time.time()
            runningLoss = 0.0
            for inputs, targets in tqdm(train_loader, file=sys.stdout, desc=f'Epoch {epoch + 1}', unit='batch'):
                inputs = inputs.unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs, _, _ = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                runningLoss += loss.item()
            trainLoss = runningLoss / len(train_loader)
            trainErrors.append(trainLoss)
            trainedEpochs += 1
            valLoss = self.evaluate(val_loader, criterion)
            valErrors.append(valLoss)
            
            # if earlyStop.check(valLoss):
            #     print(f'Early stopping at epoch {epoch + 1}')
            #     break
            
            lastLR = scheduler.get_last_lr()
            scheduler.step(valLoss)
            if lastLR != scheduler.get_last_lr():
                print(f'Learning rate changed to {scheduler.get_last_lr()}')
                
            print(f'Epoch [{epoch + 1}/{nEpochs}], Train Loss: {trainLoss:.4f}, Validation Loss: {valLoss:.4f}, took {time.time() - startTime:.2f}s')
            epochTimes.append(time.time() - startTime)
            
            if valLoss < bestValLoss:
                bestValLoss = valLoss
                if checkpoint_path is not None:
                    torch.save(self.state_dict(), checkpoint_path)

        return trainErrors, valErrors, trainedEpochs, epochTimes
    
    def evaluate(self, loader, criterion=nn.MSELoss()):
        self.eval()
        totalLoss = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                outputs, _, _ = self(inputs)
                loss = criterion(outputs, targets)
                totalLoss += loss.item()
        avgLoss = totalLoss / len(loader)
        return avgLoss
    
    def predict(self, test_loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for inputs, *_ in test_loader:
                inputs = inputs.unsqueeze(1).to(self.device)
                outputs, _, _ = self(inputs)
                predictions.append(outputs)
        return torch.cat(predictions, dim=0)
    
    def load_weights(self, path):
        print(f'Loading model from {path}')
        if self.device == 'cpu':
            self.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.load_state_dict(torch.load(path))
            self.to(self.device)
        print('Model loaded successfully')

class SimpleCNN(nn.Module):
    def __init__(self, cnn_blocks, device='cpu'):
        super().__init__()
        self.device = device
        self.cnn_blocks = cnn_blocks
        self.conv_layers = nn.ModuleList()
        
        if cnn_blocks > 5:
            raise ValueError("Maximum number of CNN blocks is 5.")
                
        Hout, Wout = 1, 256  # Initial input size

        # Take TPCNet's approach to number of kernels and kernel widths
        for i in range(cnn_blocks):
            if i == 0:
                in_channels = 1
                mid_channels = 80 # is 72 in TPCNet, but we use 80 to allow for up to 5 CNN blocks
                out_channels = 72 # is 64 in TPCNet, but we use 72 to allow for up to 5 CNN blocks
            else:
                in_channels = out_channels
                mid_channels = out_channels - 8
                out_channels = out_channels - 16
                
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=(1, 7),
                    stride=1,
                    padding=0,
                    bias=True,
                    padding_mode='zeros'
                )
            )
            self.conv_layers.append(nn.BatchNorm2d(mid_channels))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 33),
                    stride=1,
                    padding=0,
                    bias=True,
                    padding_mode='zeros'
                )
            )
            self.conv_layers.append(nn.BatchNorm2d(out_channels))
            self.conv_layers.append(nn.ReLU())
            
            # Calculate output size after this block
            Hout, Wout = self.get_output_size(Hout, Wout, k=[1, 7], s=[1, 1], p=[0, 0], d=[1, 1])
            Hout, Wout = self.get_output_size(Hout, Wout, k=[1, 33], s=[1, 1], p=[0, 0], d=[1, 1])
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(out_channels * Wout, 4)

        # init parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
            elif isinstance(layer, nn.BatchNorm2d):
                x = layer(x)
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x)
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")
        
        x = self.flatten(x)
        x = self.linear(x)
        
        return x

    def get_output_size(self, Hin, Win, k, s=[1, 1], p=[0, 0], d=[1, 1]):
        Hout = int((Hin + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1)
        Wout = int((Win + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1)
        return Hout, Wout
    
    def fit(self, train_loader, val_loader, checkpoint_path, nEpochs=100, learningRate=0.001, schedulerStep=15, stopperPatience=20, stopperTol=0.0001):
        self.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learningRate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=schedulerStep)
        trainErrors = []
        valErrors = []
        epochTimes = []
        bestValLoss = float('inf')
        
        if checkpoint_path != None and checkpoint_path[-4:] != '.pth':
            raise ValueError("Checkpoint path must end with .pth")
        
        print('Training Model')
        print('Initial learning rate:', scheduler.get_last_lr())
        trainedEpochs = 0

        for epoch in range(nEpochs):
            self.train()
            startTime = time.time()
            runningLoss = 0.0
            # for inputs, targets in tqdm(train_loader, file=sys.stdout, desc=f'Epoch {epoch + 1}', unit='batch'):
            for inputs, targets in train_loader:
                inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                runningLoss += loss.item()
            trainLoss = runningLoss / len(train_loader)
            trainErrors.append(trainLoss)
            trainedEpochs += 1
            valLoss = self.evaluate(val_loader)
            valErrors.append(valLoss)
    
            lastLR = scheduler.get_last_lr()
            scheduler.step(valLoss)
            if lastLR != scheduler.get_last_lr():
                print(f'Learning rate changed to {scheduler.get_last_lr()}')

            print(f'Epoch [{epoch + 1}/{nEpochs}], Train Loss: {trainLoss:.4f}, Validation Loss: {valLoss:.4f}, took {time.time() - startTime:.2f}s')
            epochTimes.append(time.time() - startTime)

            if valLoss < bestValLoss:
                bestValLoss = valLoss
                if checkpoint_path is not None:
                    torch.save(self.state_dict(), checkpoint_path)

        return trainErrors, valErrors, trainedEpochs, epochTimes
    
    def evaluate(self, loader, criterion=None):
        criterion = nn.MSELoss() if criterion is None else criterion
        self.eval()
        totalLoss = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                totalLoss += loss.item()
        avgLoss = totalLoss / len(loader)
        return avgLoss
    
    def predict(self, test_loader, numPredictions):
        self.eval()
        allPredictions = []
        # for _ in tqdm(range(numPredictions), file=sys.stdout, desc='Predicting'):
        for _ in range(numPredictions):
            predictions = []
            with torch.no_grad():
                for inputs, *_ in test_loader:
                    inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                    outputs = self(inputs)
                    predictions.append(outputs)
            allPredictions.append(torch.cat(predictions, dim=0))
        return torch.stack(allPredictions, dim=0)
    
    def load_weights(self, path):
        print(f'Loading model from {path}')
        if self.device == 'cpu':
            self.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.load_state_dict(torch.load(path))
            self.to(self.device)
        print('Model loaded successfully')

class SimpleBNN(nn.Module):
    def __init__(self, cnn_blocks, prior_mu, prior_sigma, kl_weight = 0.01, device='cpu'):
        super().__init__()
        self.device = device
        self.cnn_blocks = cnn_blocks
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.kl_weight = kl_weight
        self.conv_layers = nn.ModuleList()
        
        if cnn_blocks > 5:
            raise ValueError("Maximum number of CNN blocks is 5.")
                
        Hout, Wout = 1, 256  # Initial input size

        # Take TPCNet's approach to number of kernels and kernel widths
        for i in range(cnn_blocks):
            if i == 0:
                in_channels = 1
                mid_channels = 80 # is 72 in TPCNet, but we use 80 to allow for up to 5 CNN blocks
                out_channels = 72 # is 64 in TPCNet, but we use 72 to allow for up to 5 CNN blocks
            else:
                in_channels = out_channels
                mid_channels = out_channels - 8
                out_channels = out_channels - 16
                
            self.conv_layers.append(
                bnn.BayesConv2d(
                    prior_mu=prior_mu,
                    prior_sigma=prior_sigma,
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=(1, 7),
                    stride=1,
                    padding=0,
                    bias=True,
                    padding_mode='zeros'
                )
            )
            self.conv_layers.append(bnn.BayesBatchNorm2d(prior_mu, prior_sigma, mid_channels))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(
                bnn.BayesConv2d(
                    prior_mu=prior_mu,
                    prior_sigma=prior_sigma,
                    in_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 33),
                    stride=1,
                    padding=0,
                    bias=True,
                    padding_mode='zeros'
                )
            )
            self.conv_layers.append(bnn.BayesBatchNorm2d(prior_mu, prior_sigma, out_channels))
            self.conv_layers.append(nn.ReLU())
            
            # Calculate output size after this block
            Hout, Wout = self.get_output_size(Hout, Wout, k=[1, 7], s=[1, 1], p=[0, 0], d=[1, 1])
            Hout, Wout = self.get_output_size(Hout, Wout, k=[1, 33], s=[1, 1], p=[0, 0], d=[1, 1])
        
        self.flatten = nn.Flatten()
        self.linear = bnn.BayesLinear(prior_mu, prior_sigma, out_channels * Wout, 4)

        # TorchBNN uses the same initialisation as Adv-BNN which seems fine
        # # init parameter
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0]*m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2./n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

    def forward(self, x):
        for layer in self.conv_layers:
            if isinstance(layer, bnn.BayesConv2d):
                x = layer(x)
            elif isinstance(layer, bnn.BayesBatchNorm2d):
                x = layer(x)
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x)
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")
        
        x = self.flatten(x)
        x = self.linear(x)
        
        return x

    def get_output_size(self, Hin, Win, k, s=[1, 1], p=[0, 0], d=[1, 1]):
        Hout = int((Hin + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1)
        Wout = int((Win + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1)
        return Hout, Wout
    
    def lossFunction(self, outputs, targets, KLweight):
        MSE = nn.MSELoss()
        BKLoss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        return MSE(outputs, targets) + KLweight * BKLoss(self)
    
    def fit(self, train_loader, val_loader, checkpoint_path, nEpochs=100, learningRate=0.001, schedulerStep=15, stopperPatience=20, stopperTol=0.0001):
        self.to(self.device)
        criterion = self.lossFunction
        optimizer = optim.Adam(self.parameters(), lr=learningRate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=schedulerStep)
        trainErrors = []
        valErrors = []
        epochTimes = []
        bestValLoss = float('inf')
        
        if checkpoint_path != None and checkpoint_path[-4:] != '.pth':
            raise ValueError("Checkpoint path must end with .pth")
        
        print('Training Model')
        print('Initial learning rate:', scheduler.get_last_lr())
        trainedEpochs = 0

        for epoch in range(nEpochs):
            self.train()
            startTime = time.time()
            runningLoss = 0.0
            # for inputs, targets in tqdm(train_loader, file=sys.stdout, desc=f'Epoch {epoch + 1}', unit='batch'):
            for inputs, targets in train_loader:
                inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets, self.kl_weight)
                loss.backward()
                optimizer.step()
                runningLoss += loss.item()
            trainLoss = runningLoss / len(train_loader)
            trainErrors.append(trainLoss)
            trainedEpochs += 1
            valLoss = self.evaluate(val_loader)
            valErrors.append(valLoss)
    
            lastLR = scheduler.get_last_lr()
            scheduler.step(valLoss)
            if lastLR != scheduler.get_last_lr():
                print(f'Learning rate changed to {scheduler.get_last_lr()}')

            print(f'Epoch [{epoch + 1}/{nEpochs}], Train Loss: {trainLoss:.4f}, Validation Loss: {valLoss:.4f}, took {time.time() - startTime:.2f}s')
            epochTimes.append(time.time() - startTime)

            if valLoss < bestValLoss:
                bestValLoss = valLoss
                if checkpoint_path is not None:
                    torch.save(self.state_dict(), checkpoint_path)

        return trainErrors, valErrors, trainedEpochs, epochTimes
    
    def evaluate(self, loader, criterion=None):
        criterion = nn.MSELoss() if criterion is None else criterion
        self.eval()
        totalLoss = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                totalLoss += loss.item()
        avgLoss = totalLoss / len(loader)
        return avgLoss
    
    def predict(self, test_loader, numPredictions):
        self.eval()
        allPredictions = []
        # for _ in tqdm(range(numPredictions), file=sys.stdout, desc='Predicting'):
        for _ in range(numPredictions):
            predictions = []
            with torch.no_grad():
                for inputs, *_ in test_loader:
                    inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                    outputs = self(inputs)
                    predictions.append(outputs)
            allPredictions.append(torch.cat(predictions, dim=0))
        return torch.stack(allPredictions, dim=0)
    
    def load_weights(self, path):
        print(f'Loading model from {path}')
        if self.device == 'cpu':
            self.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.load_state_dict(torch.load(path))
            self.to(self.device)
        print('Model loaded successfully')
