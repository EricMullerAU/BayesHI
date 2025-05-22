import math
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchbnn as bnn
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    """ Additive sinusoidal positional encoding. """
    def __init__(self, d_model, max_len=256):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

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
    def __init__(self, cnnBlocks, kernelNumber, kernelWidth, MHANumber, transformerNumber, priorMu, priorSigma, posEncType, device):
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
            nn.TransformerEncoderLayer(d_model=kernelNumber, nhead=MHANumber, batch_first=True),
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
        x = torch.cat((F.softmax(x[:, :3], dim=1), torch.relu(x[:, 3]).unsqueeze(1)), dim=1)
        return x

    def get_output_size(self, Hin, Win, k, s=[1, 1], p=[0, 0], d=[1, 1]):
        Hout = int((Hin + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1)
        Wout = int((Win + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1)
        return Hout, Wout

    def lossFunction(self, outputs, targets, KLweight):
        MSE = nn.MSELoss()
        BKLoss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        return MSE(outputs, targets) + KLweight * BKLoss(self)

    def fit(self, trainLoader, valLoader, trainingProcessPath, prefix, nEpochs = 50, learningRate = 0.0005, schedulerStep = 15, stopperPatience = 5, stopperTol = 1e-4, maxKLweight = 0.01, maxKLepoch = 50):
        self.to(self.device)
        criterion = self.lossFunction
        optimizer = optim.Adam(self.parameters(), lr=learningRate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=schedulerStep)
        earlyStop = earlyStopper(patience=stopperPatience, tol=stopperTol)
        trainErrors = []
        valErrors = []
        epochTimes = []
        bestValLoss = float('inf')
        bestModelPath = trainingProcessPath / f'{prefix}_best_model.pth'

        print('Training Model')
        print('Initial learning rate:', scheduler.get_last_lr())
        trainedEpochs = 0

        for epoch in range(nEpochs):
            KLweight = maxKLweight * min(1, epoch / maxKLepoch)
            startTime = time.time()
            self.train()
            runningLoss = 0.0
            for inputs, targets in trainLoader:
                inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets, KLweight)
                loss.backward()
                optimizer.step()
                runningLoss += loss.item()

            trainLoss = runningLoss / len(trainLoader)
            trainErrors.append(trainLoss)
            trainedEpochs += 1
            valLoss = self.evaluate(valLoader, nn.MSELoss())
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
                torch.save(self.state_dict(), bestModelPath)

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

    def predict(self, testLoader, numPredictions):
        self.eval()
        allPredictions = []
        for _ in tqdm(range(numPredictions), desc='Predicting', file=sys.stdout):
            predictions = []
            with torch.no_grad():
                for inputs, _ in testLoader:
                    inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                    outputs = self(inputs)
                    predictions.append(outputs)
            allPredictions.append(torch.cat(predictions, dim=0))
        return torch.stack(allPredictions, dim=0)

    def load_weights(self, path = '/scratch/fd08/em8117/training_process/grid_search/peoff_c1_k12_w51_mha4_t1_e50_lr0.0005_s15_p5_tol1e-4_kl0.01/paperModelv2NoScalingKLannealing_best_model.pth'):
        print(f'Loading model from {path}')
        try:
            if self.device == 'cpu':
                self.load_state_dict(torch.load(path, map_location=self.device))
            else:
                self.load_state_dict(torch.load(path))
                self.to(self.device)
            print('Model loaded successfully')
        except:
            print('Error loading model')

class TPCNetPositionalEncoding(nn.Module):
    def __init__(self, num_features, sequence_len=6, d_model=9):
        super(TPCNetPositionalEncoding, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        pe     = torch.zeros((1, sequence_len, d_model), dtype=torch.float32).to(self.device)
        factor = -math.log(10000.0) / d_model  # outs loop
        
        for index in range(0, sequence_len):  # position of word in seq
            for i in range(0, d_model, 2):
                #print("i==",i)
                div_term = math.exp(i * factor)
                pe[0, index, i] = math.sin(index * div_term)
                if(i+1 < d_model):
                    pe[0, index, i+1] = math.cos(index * div_term)
                
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x has shape [seq_len, bat_size, embed_dim]
        x = x + self.pe[:x.size(0), :]
        return x

class tpcnet_all_phases(nn.Module):
    def __init__(self, num_output=2, in_channels=1, input_row=2, input_column=256, drop_out_rate=0., lpe=False, device='cpu'):
        super(tpcnet_all_phases, self).__init__()

        p = [0, 0] # padding
        d = [1, 1] # dilation
        k = [1, 6] # kernael_size # 7 here
        s = [1, 1] # stride
        
        self.device = device
        
        self.num_features = 54
        self.input_row    = input_row
        self.in_channels  = in_channels
        self.input_column = input_column

        kernel_wid = 33# 40 if input_column == 256 else 10
        
        self.drop_rate   = drop_out_rate
        self.pos_encoder = TPCNetPositionalEncoding(num_features=self.num_features, sequence_len=6, d_model=9)
        self.lpe = lpe
        self.pos_embedding = nn.Parameter(torch.randn(self.in_channels,self.input_row, self.input_column))

        # num_layer*8 + 8
        
        # CNN layers (outchannels = outchannels-8)
        kernelsize = (1,7) if (input_row < 2) else (2,3) 
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


        self.conv9 = nn.Conv2d(in_channels=8, out_channels=4,  kernel_size=kernelsize, stride=1, padding=0, bias=True, padding_mode='zeros')
        self.bn9 = nn.BatchNorm2d(4)
        Hout, Wout = self.get_output_size(4, Wout, k = kernelsize, s=s, p=p, d=d)
        # print('>>> Conv2: ', Hout, Wout)

        self.conv10 = nn.Conv2d(in_channels=4, out_channels=2,  kernel_size=(1,kernel_wid), stride=1, padding=0, bias=True, padding_mode='zeros')
        self.bn10 = nn.BatchNorm2d(2)
        Hout, Wout = self.get_output_size(2, Wout, k = (1,kernel_wid), s=s, p=p, d=d)
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
                dropout=self.drop_rate,
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
        # print(1, x.size())
        if self.lpe:
            # x =  x + self.pos_embedding
            x = self.pos_encoder(x)
            # print(1, 'lpe', x.size())
        
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

        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)
        # print(x.size())

        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x)
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
        
        # # Add sinusoidal positional encoding
        # if not self.lpe:
        #     x = self.pos_encoder(x)
        
        # Transformer MODEL
        x = self.transformer(x)
        # print('x after trans: ', x.size())
        x = self.flatten(x)
        # print('x after flatten: ', x.size())
    
        x = self.decoder(x)
        # print('x output: ', x.size())
        return x

    def get_output_size(self, Hin, Win, k, s=[1, 1], p=[0, 0], d=[1, 1]):
        Hout = int((Hin + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1)
        Wout = int((Win + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1)
        return Hout, Wout

    def lossFunction(self, outputs, targets):
        MSE = nn.MSELoss()
        return MSE(outputs, targets)

    def fit(self, trainLoader, valLoader, trainingProcessPath, prefix, nEpochs = 100, learningRate = 0.0001, schedulerStep = 15, stopperPatience = 20, stopperTol = 0):
        self.to(self.device)
        criterion = self.lossFunction
        optimizer = optim.Adam(self.parameters(), lr=learningRate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=schedulerStep)
        earlyStop = earlyStopper(patience=stopperPatience, tol=stopperTol)
        trainErrors = []
        valErrors = []
        epochTimes = []
        bestValLoss = float('inf')
        #TODO: save with epoch number in name for interruped training, add handling to train if not at provided epochs
        bestModelPath = trainingProcessPath / f'{prefix}_best_model.pth'

        print('Training Model')
        print('Initial learning rate:', scheduler.get_last_lr())
        trainedEpochs = 0

        for epoch in range(nEpochs):
            startTime = time.time()
            self.train()
            runningLoss = 0.0
            for inputs, targets in tqdm(trainLoader, file=sys.stdout, desc='Training'):
                inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                runningLoss += loss.item()

            trainLoss = runningLoss / len(trainLoader)
            trainErrors.append(trainLoss)
            trainedEpochs += 1
            valLoss = self.evaluate(valLoader, nn.MSELoss())
            valErrors.append(valLoss)
            
            if trainLoss == np.NaN or valLoss == np.NaN:
                print('NaN loss detected, stopping training')
                break
            
            if earlyStop.check(valLoss):
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
                torch.save(self.state_dict(), bestModelPath)

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

    def predict(self, testLoader, numPredictions):
        if numPredictions  > 1:
            print('Warning: numPredictions > 1, but this model is not Bayesian, so all predictions will be the same. Re-run with numPredictions = 1 for faster inference.')
        self.eval()
        allPredictions = []
        for _ in tqdm(range(numPredictions), file=sys.stdout, desc='Predicting'):
            predictions = []
            with torch.no_grad():
                for inputs, *_ in testLoader:
                    inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                    outputs = self(inputs)
                    predictions.append(outputs)
            allPredictions.append(torch.cat(predictions, dim=0))
        return torch.stack(allPredictions, dim=0)

    def load_weights(self, path = '/scratch/fd08/em8117/training_process/TPCNet/TPCNet_best_model.pth'):
        print(f'Loading model from {path}')
        try:
            if self.device == 'cpu':
                self.load_state_dict(torch.load(path, map_location=self.device))
            else:
                self.load_state_dict(torch.load(path))
                self.to(self.device)
            print('Model loaded successfully')
        except:
            print('Error loading model')
    

class bayeshi_model(nn.Module):
    def __init__(self, cnnBlocks, kernelNumber, kernelWidth1, kernelWidth2, kernelMult, pooling, MHANumber, transformerNumber, priorMu, priorSigma, posEncType, device):
        super(bayeshi_model, self).__init__()
        self.device = device
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
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
            if pooling == 'max':
                self.conv_layers.append(nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
            elif pooling == 'avg':
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
            if pooling == 'max':
                self.conv_layers.append(nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
            elif pooling == 'avg':
                self.conv_layers.append(nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))
            
            in_channels = out_channels
            Hout, Wout = self.get_output_size(Hout, Wout, k=[1, kernelWidth1], s=[1, 1], p=[0, 0], d=[1, 1])
            if pooling != 'off':
                Wout = Wout // 2
            Hout, Wout = self.get_output_size(Hout, Wout, k=[1, kernelWidth2], s=[1, 1], p=[0, 0], d=[1, 1])
            if pooling != 'off':
                Wout = Wout // 2
            
        if posEncType == 'sinusoidal':
            self.positional_encoding = PositionalEncoding(d_model=kernelNumber)
        elif posEncType == 'stochastic':
            self.positional_encoding = StochasticPositionalEncoding(d_model=kernelNumber)
        else:
            self.positional_encoding = None
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=kernelNumber, nhead=MHANumber, batch_first=True),
            num_layers=transformerNumber
        )
        self.flatten = nn.Flatten()
        self.decoder = bnn.BayesLinear(prior_mu=priorMu, prior_sigma=priorSigma,
                                        in_features=kernelNumber * Wout, out_features=4)

    def forward(self, x):
        for layer in self.conv_layers:
            if isinstance(layer, bnn.BayesConv2d):
                # Apply convolutional layer
                x = layer(x)
                x = F.relu(x)
            elif isinstance(layer, nn.MaxPool2d) or isinstance(conv, nn.AvgPool2d):
                # Apply pooling layer and then ReLU
                x = layer(x)
        # Once finished with the convolutions pass to the transformer
        x = x.squeeze(2).permute(0, 2, 1)  # (batch, width, channels)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.flatten(x)
        x = self.decoder(x)
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

    def fit(self, trainLoader, valLoader, checkpoint_path, nEpochs = 100, learningRate = 0.0001, schedulerStep = 15, stopperPatience = 20, stopperTol = 0.0001, maxKLweight = 0.01, maxKLepoch = 100):
        
        self.to(self.device)
        criterion = self.lossFunction
        optimizer = optim.Adam(self.parameters(), lr=learningRate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=schedulerStep)
        earlyStop = earlyStopper(patience=stopperPatience, tol=stopperTol)
        trainErrors = []
        valErrors = []
        epochTimes = []
        bestValLoss = float('inf')
        #TODO: save with epoch number in name for interruped training, add handling to train if not at provided epochs
        # Check if the provided path ends with .pth
        if checkpoint_path[-4:] != '.pth':
            raise ValueError("Checkpoint path must end with .pth")
        
        print('Training Model')
        print('Initial learning rate:', scheduler.get_last_lr())
        trainedEpochs = 0

        for epoch in range(nEpochs):
            KLweight = maxKLweight * min(1, epoch / maxKLepoch)
            startTime = time.time()
            self.train()
            runningLoss = 0.0
            for inputs, targets in trainLoader:
                inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                # print('Outputs:', outputs[0])
                # print('Targets:', targets[0])
                loss = criterion(outputs, targets, KLweight)
                # print('Loss:', loss.item())
                loss.backward()
                optimizer.step()
                runningLoss += loss.item()

            trainLoss = runningLoss / len(trainLoader)
            trainErrors.append(trainLoss)
            trainedEpochs += 1
            valLoss = self.evaluate(valLoader, nn.MSELoss())
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

            # Lmao this saves every epoch instead of every run at the moment. Oops.
            if valLoss < bestValLoss:
                bestValLoss = valLoss
                torch.save(self.state_dict(), checkpoint_path)

        return trainErrors, valErrors, trainedEpochs, epochTimes

    def evaluate(self, loader, criterion=nn.MSELoss()):
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

    def predict(self, testLoader, numPredictions):
        self.eval()
        allPredictions = []
        for _ in tqdm(range(numPredictions), file=sys.stdout, desc='Predicting'):
            predictions = []
            with torch.no_grad():
                for inputs, *_ in testLoader:
                    inputs = inputs.unsqueeze(1).unsqueeze(1).to(self.device)
                    outputs = self(inputs)
                    predictions.append(outputs)
            allPredictions.append(torch.cat(predictions, dim=0))
        return torch.stack(allPredictions, dim=0)

    def load_weights(self, path = '/scratch/fd08/em8117/training_process/TIGRESS_grid_search/TIGRESS_pesinusoidal_c1_k8_wone31_wtwo3_km2_ptoff_mha4_t1_e100_lr0.0001_s15_p20_tol1e-4_kl0.01/run1/TIGRESS_model_best_model_full.pth'):
        print(f'Loading model from {path}')
        try:
            if self.device == 'cpu':
                self.load_state_dict(torch.load(path, map_location=self.device))
            else:
                self.load_state_dict(torch.load(path))
                self.to(self.device)
            print('Model loaded successfully')
        except:
            print('Error loading model')