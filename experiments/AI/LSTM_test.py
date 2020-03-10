
import numpy as np
import pandas as pd

from netCDF4 import Dataset

import torch
import torch.nn as nn


def get_data_sample():

    lut = pd.read_csv('/Users/u0116961/Documents/work/machine_learning/test_data/MERRA2/grid_lut.csv', index_col=0)

    i_lat = 500
    i_lon = 500

    ds = Dataset('/Users/u0116961/Documents/work/machine_learning/test_data/COPERNICUS_DMP/DMP_COPERNICUS_timeseries.nc')
    dmp_ts = pd.Series(ds['DMP'][:,i_lat,i_lon])

    lat = ds['lat'][i_lat]
    lon = ds['lon'][i_lon]

    idx = ((lut['merra2_lat']-lat)**2 + (lut['merra2_lon']-lon)**2).idxmin()

    ssm_ts = pd.read_csv('/Users/u0116961/Documents/work/machine_learning/test_data/MERRA2/timeseries/%i.csv' % idx, index_col=0, header=None, names=['ssm'], parse_dates=True)
    ssm_ts = ssm_ts.resample('1D').mean()




def get_neuron_number(n_input=3, n_output=1, n_samples=200, alpha=7):
    return round(n_samples / (alpha * (n_input + n_output)))


#Set Parameters for a small LSTM network
input_dim  = 5          # dimension of input layer
hidden_dim  = 10        # dimension of hidden layer / cell state
n_layers = 1            # number of LSTM cells

seq_len = 1             # size of predictive sequence
batch_size = 1          # size of training batches

output_size = 1         # size of the output



lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

h0 = torch.randn(n_layers, batch_size, hidden_dim)  # initial hidden state
c0 = torch.randn(n_layers, batch_size, hidden_dim)  # initial cell state
initial = (h0, c0)

inp = torch.randn(batch_size, seq_len, input_dim)

out, (hn, cn) = lstm(inp, (h0, c0))

print("Output shape: ", out.shape)
print("Hidden: ", hidden)






'''
Input dimension: size of input at each time step [x1, x2, ... , xn]
Hidden dimension: shape of hidden state (short-term memory) and cell state (long-term memory) at each time step
Number of layers: # of LSTM layers/cells stacked on top of each other

'''