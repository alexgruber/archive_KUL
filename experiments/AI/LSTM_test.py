
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from itertools import repeat
from netCDF4 import Dataset, num2date

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
torch.manual_seed(1)

from torch.utils.data import Dataset, DataLoader

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

def get_data_sample():

    lut = pd.read_csv('/Users/u0116961/Documents/work/machine_learning/test_data/MERRA2/grid_lut.csv', index_col=0)

    i_lat = 500
    i_lon = 500

    ds = Dataset('/Users/u0116961/Documents/work/machine_learning/test_data/COPERNICUS_DMP/DMP_COPERNICUS_timeseries.nc')
    dmp_ts = pd.DataFrame({'DMP': ds['DMP'][:,i_lat,i_lon]}, index=pd.DatetimeIndex(num2date(ds['time'][:], ds['time'].units)))
    dmp_ts['dekad'] = (dmp_ts.index.day.values / 10.).round().astype('int') + (dmp_ts.index.month.values.astype('int')-1)*3
    dmp_ts['seas'] = np.convolve(dmp_ts['DMP'], np.ones(5) / 5, mode='same')
    dmp_ts['clim'] = np.convolve(dmp_ts['DMP'].groupby(dmp_ts['dekad']).mean(), np.ones(3) / 3., 'same')[dmp_ts['dekad']-1]
    dmp_ts['anom_st'] = dmp_ts['DMP'] - dmp_ts['seas']
    dmp_ts['anom_lt'] = dmp_ts['DMP'] - dmp_ts['clim']

    # lat = ds['lat'][i_lat]
    # lon = ds['lon'][i_lon]
    #
    # idx = ((lut['merra2_lat']-lat)**2 + (lut['merra2_lon']-lon)**2).idxmin()
    #
    # ssm_ts = pd.read_csv('/Users/u0116961/Documents/work/machine_learning/test_data/MERRA2/timeseries/%i.csv' % idx, index_col=0, header=None, names=['ssm'], parse_dates=True)
    # ssm_ts = ssm_ts.resample('1D').mean()
    # ssm_ts['seas'] = np.convolve(ssm_ts['ssm'], np.ones(31)/31, 'same')
    # ssm_ts['anom_st'] = ssm_ts['ssm'] - ssm_ts['seas']

    return dmp_ts


def get_neuron_number(n_input=3, n_output=1, n_samples=200, alpha=7):
    """Rule of thumb for estimating the 'optimal' number of hidden neurons"""

    return round(n_samples / (alpha * (n_input + n_output)))



# def create_inout_sequences(data, window):
#     """
#     Creates training data as a list of moving blocks of <window_size> with the
#     <block + 1>th element to be predicted.
#     """
#
#     N = len(data) - window
#     indices = range(N)
#     fkt = lambda idx, data, window: (data[idx:idx+window,:], data[idx+window:idx+window+1,0])
#
#     return list(map(fkt, indices, repeat(data, N), repeat(window,N)))

def create_inout_sequences(data, window):
    """
    Creates training data as a list of moving blocks of <window_size> with the
    <block + 1>th element to be predicted.
    """

    N = len(data) - window
    indices = range(N)

    x = [data[idx:idx+window,:] for idx in indices]
    y = [data[idx+window:idx+window+1,0] for idx in indices]

    return x, y


class LSTM(nn.Module):

    def __init__(self, input_size=1, output_size=1, n_layers=1, hidden_size=50, batch_size=1):
        super().__init__()

        self.n_layers = n_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        # Initialize LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)

        # Initialize output layer
        self.linear = nn.Linear(hidden_size, output_size)

        self.hidden_cell = (torch.randn(n_layers, batch_size, hidden_size),
                            torch.randn(n_layers, batch_size, hidden_size))


    def forward(self, input_seq):
        # lstm_out, self.hidden_cell = self.lstm(input_seq.view(self.batch_size, len(input_seq), -1), self.hidden_cell)
        lstm_out, self.hidden_cell = self.lstm(input_seq.float(), self.hidden_cell)
        predictions = self.linear(lstm_out[:,-1,:])
        return predictions


    def reset_hidden(self):
        batch_size = self.batch_size if self.training else 1
        self.hidden_cell = (torch.randn(self.n_layers, batch_size, self.hidden_size),
                            torch.randn(self.n_layers, batch_size, self.hidden_size))


class Data(Dataset):
    def __init__(self, x, y):
        super(Data, self).__init__()
        assert len(x) == len(y) # assuming shape[0] = dataset size
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def test_lstm():

    # ------------------------------------------------------------------------------------------------
    # --- Initializations ---

    # xx months of test data
    test_data_size = 36

    # predictions for month i are based on the <train_window> previous months.
    train_window = 12

    # LSTM parameters
    input_dim = 2
    output_dim = 1
    n_layers = 1
    hidden_size = 150
    batch_size = 8

    # # training epochs
    epochs = 200
    learning_rate = 0.01


    # ------------------------------------------------------------------------------------------------
    # --- Data preparation ---

    # Initialize data set (12 years of monthly data = 144 data points)
    passengers = sns.load_dataset("flights")['passengers'].values.astype('float')

    x = np.arange(len(passengers))
    [k, d] = np.polyfit(x, passengers, deg=1)
    trend = k*x + d

    all_data = np.vstack((passengers, trend)).T

    # Split into training and test data
    train_data = all_data[:-test_data_size,:]
    test_data = all_data[-test_data_size:,:]

    # Normalize data between [-1, 1] and convert to tensor
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_data[:,0].reshape(-1, 1))
    train_data_normalized = scaler.transform(train_data)

    # Create the training data set
    # train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
    x, y = create_inout_sequences(train_data_normalized, train_window)
    trainloader  = DataLoader(Data(x, y), batch_size=batch_size, shuffle=True, num_workers=8)

    # ------------------------------------------------------------------------------------------------
    # --- LSTM setup ---

    model = LSTM(input_size=input_dim,
                 output_size=output_dim,
                 n_layers=n_layers,
                 hidden_size=hidden_size,
                 batch_size=batch_size)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # ------------------------------------------------------------------------------------------------
    # --- Model training ---

    losses_all = []
    for i in range(epochs):
        for seq, labels in trainloader:

            optimizer.zero_grad()                                       # Reset gradients for this epoch
            model.reset_hidden()                                        # Reset hidden cell state
            y_pred = model(seq)                                         # Forward propagation
            single_loss = loss_function(y_pred, labels.float())         # Calculate loss
            single_loss.backward()                                      # Back-propagate gradients
            optimizer.step()                                            # Perform parameter updates
        losses_all.append(single_loss.item())

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    plt.figure(figsize=(10, 10))
    plt.plot(range(epochs), losses_all)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.tight_layout()
    plt.show()


    # ------------------------------------------------------------------------------------------------
    # --- Model evaluation ---

    model.eval()        # switch from training to evaluation mode

    test_inputs = train_data_normalized[-train_window:]     # Last part of training data
    test_trend = scaler.transform(test_data)[:,1]           # Trend as additional input

    for i in range(test_data_size):
        with torch.no_grad():

            model.reset_hidden()

            prediction = model(torch.Tensor(test_inputs[-train_window:]).view(1,train_window,-1))             # Make prediction
            test_inputs = np.concatenate((test_inputs, np.array([[prediction.data, test_trend[i]]])))         # Append do input sequence

            # test_inputs = torch.cat((test_inputs, prediction.data))     # Append to input sequence

    actual_predictions = scaler.inverse_transform((test_inputs[-test_data_size:,0]).reshape(-1, 1)).flatten()


    # ------------------------------------------------------------------------------------------------
    # --- Vizualize output ---

    plt.figure(figsize=(15,5))
    plt.title('Month vs Passenger')
    plt.ylabel('Total Passengers')
    plt.xlabel('Months')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)

    plt.plot(all_data)

    x = np.arange(len(all_data)-test_data_size, len(all_data))
    plt.plot(x, actual_predictions)

    plt.show()


if __name__=='__main__':

    test_lstm()

    # get_data_sample().plot()
    # plt.show()

