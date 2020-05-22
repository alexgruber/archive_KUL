
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from netCDF4 import Dataset, num2date

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
torch.manual_seed(1)

from torch.utils.data import Dataset as torchDs
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

def read_data(i_lat=0, i_lon=0):

    # [10 days]
    seas_window = 5
    clim_window = 5

    with Dataset('/Users/u0116961/Documents/work/machine_learning/test_data/COPERNICUS_DMP/DMP_COPERNICUS_timeseries.nc') as ds:
        dmp_ts = pd.DataFrame({'DMP': ds['DMP'][:,i_lat,i_lon]}, index=pd.DatetimeIndex(num2date(ds['time'][:], ds['time'].units)))

        lat = ds['lat'][i_lat].data
        lon = ds['lon'][i_lon].data

    with Dataset('/Users/u0116961/data_sets/MERRA2/MERRA2_timeseries.nc4') as ds:

        ind_lat = abs(ds['lat'][:]-lat).argmin()
        ind_lon = abs(ds['lon'][:]-lon).argmin()

        merra_ts = pd.DataFrame(index=pd.DatetimeIndex(num2date(ds['time'][:], ds['time'].units)))
        # variables = ['SFMC', 'RZMC', 'PRMC', 'TSOIL1', 'TSURF', 'SWLAND', 'LWLAND']
        variables = ['SFMC', 'TSOIL1', 'SWLAND', 'LWLAND']
        for var in variables:
            merra_ts[var] = np.convolve(ds[var][:,ind_lat,ind_lon], np.ones(9) / 9, mode='same')
            if (var == 'TSOIL1') | (var == 'TSURF'):
                merra_ts[var] -= 273.15
            if (var == 'SFMC') | (var == 'RZMC') | (var == 'PRMC'):
                merra_ts[var] *= 100
            # if var == 'LWLAND':
            #     merra_ts[var] *= -1
            if var == 'SWLAND':
                merra_ts[var] /= 3
        merra_ts.index = merra_ts.index.shift(4, freq='D')
        merra_ts = merra_ts.reindex(dmp_ts.index)

        df = pd.concat((dmp_ts,merra_ts), axis='columns')
        df['dekad'] = (df.index.day.values / 10.).round().astype('int') + (df.index.month.values.astype('int')-1)*3

        for col in df:
            if col != 'dekad':
                df[col + '_seas'] = np.convolve(df[col], np.ones(seas_window) / seas_window, mode='same')
                df[col + '_anom_st'] = df[col] - df[col + '_seas']

                df[col + '_clim'] = np.convolve(df[col].groupby(df['dekad']).mean(), np.ones(clim_window) / clim_window, 'same')[df['dekad']-1]
                df[col + '_anom_lt'] = df[col] - df[col + '_clim']

        return df

def crosscorr(x, y, lags=None):

    assert len(x) == len(y)
    if not lags:
        lags = len(x)-1

    return [np.corrcoef(x,y)[0,1]] + [np.corrcoef(x[lag:], y[:-lag])[0,1] for lag in range(1, lags)]

def create_crosscorr_plot(i_lat=0, i_lon=0):

    fout = '/Users/u0116961/Documents/work/machine_learning/DMP_prediction/plots/acf_%i_%i.png' % (i_lat, i_lon)

    df = read_data(i_lat=i_lat, i_lon=i_lon)

    lags = 13

    xticks = [0, 4, 8, 12]
    xlabels = [tick*10 for tick in xticks]

    variables = ['DMP','SFMC', 'TSOIL1', 'SWLAND', 'LWLAND']
    # variables = ['DMP','SFMC', 'RZMC', 'PRMC', 'TSOIL1', 'TSURF', 'SWLAND', 'LWLAND']
    modes = ['', '_clim', '_seas', '_anom_lt', '_anom_st']

    fontsize = 12
    f = plt.figure(figsize=(16, 10), dpi=300)

    for i, dmp_mode in enumerate(modes):
        for j, var in enumerate(variables):
            plt.subplot(len(modes), len(variables), j + i * len(variables)+1)
            plt.xticks(xticks,xlabels, fontsize=fontsize-2)
            plt.yticks(fontsize=fontsize-2)

            if i == 0:
                plt.title(var, fontsize=fontsize)
            if i < len(modes)-1:
                plt.gca().set_xticks([])
            else:
                if j == 2:
                    plt.xlabel('time lag [days]', fontsize=fontsize)
            if j == 0:
                plt.ylabel('DMP' + dmp_mode, fontsize=fontsize)
            else:
                if (j > 1)|(i<=2):
                    plt.gca().set_yticks([])
            if 'anom' in dmp_mode:
                if j == 0:
                    plt.ylim(-0.9,0.9)
                else:
                    plt.ylim(-0.4,0.4)
            else:
                plt.ylim(-1,1)
            plt.xlim(0,lags)
            plt.axhline(color='black', linestyle='--', linewidth=0.75)
            for pos in xticks:
                plt.axvline(pos, color='black', linestyle='--', linewidth=0.25)

            corrs = pd.DataFrame(index=np.arange(lags))
            for mode in modes:
                corrs[mode[1::]] = crosscorr(df['DMP' + dmp_mode], df[var+mode], lags=lags)
            corrs.columns.values[0] = 'raw'
            corrs.plot(ax=plt.gca(), legend=True if ((i==0) & (j==0)) else False)

    f.savefig(fout, bbox_inches='tight')

def calc_correlation_matrix():

    df = read_data()

    R_file = '/Users/u0116961/Documents/work/machine_learning/DMP_prediction/R_matrix.xlsx'
    corr = df.corr().to_excel(R_file)


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
        batch_size = self.batch_size if self.training else 1
        lstm_out, self.hidden_cell = self.lstm(input_seq.float(), self.hidden_cell)
        predictions = self.linear(lstm_out[:,-1,:]).view(batch_size,1,-1)
        return predictions

    def reset_hidden(self):
        batch_size = self.batch_size if self.training else 1
        self.hidden_cell = (torch.randn(self.n_layers, batch_size, self.hidden_size),
                            torch.randn(self.n_layers, batch_size, self.hidden_size))


class Data(torchDs):
    def __init__(self, x, y):
        super(Data, self).__init__()
        assert len(x) == len(y) # assuming shape[0] = dataset size
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def read_data_sequences(ind_lat=350, ind_lon=0, in_vars=[], out_vars=[], window=1):
    """
    Creates training data as a list of moving blocks of <window_size> with the
    <block + 1>th element to be predicted.
    """

    df = read_data(ind_lat, ind_lon)[in_vars]
    assert len(np.where(np.isnan(df))[0]) == 0

    df.plot()
    plt.tight_layout()
    plt.show()

    scalers = []
    for i, var in enumerate(in_vars):
        scalers += [StandardScaler()]
        df[var] = scalers[i].fit_transform(df[var].values.reshape(-1,1))
    scalers = scalers[0:len(out_vars)]

    in_data = df[in_vars].values
    out_data = df[out_vars].values

    N = len(in_data) - window
    indices = range(N)

    in_seq = [in_data[idx:idx+window,:] for idx in indices]
    out_seq = [out_data[idx+window:idx+window+1,:] for idx in indices]

    return in_seq, out_seq, scalers


def test_lstm():

    # ------------------------------------------------------------------------------------------------
    # --- Initializations ---

    ind_lat = 350
    ind_lon = 0

    out_vars = ['DMP', 'DMP_anom_st', 'DMP_anom_lt']
    in_vars = out_vars + ['DMP_clim',
                          'SFMC', 'SFMC_clim', 'SFMC_seas', 'SFMC_anom_lt', 'SFMC_anom_st',
                          'TSOIL1', 'TSOIL1_clim', 'TSOIL1_seas', 'TSOIL1_anom_lt', 'TSOIL1_anom_st',
                          'SWLAND', 'SWLAND_clim', 'SWLAND_seas', 'SWLAND_anom_lt', 'SWLAND_anom_st',
                          'LWLAND', 'LWLAND_clim', 'LWLAND_seas', 'LWLAND_anom_lt', 'LWLAND_anom_st'
                          ]

    train_window = 18       # [dekads]

    test_data_frac = 0.25

    # LSTM parameters
    input_dim = len(in_vars)
    output_dim = len(out_vars)
    n_layers = 1
    hidden_size = 100
    batch_size = 4

    # training epochs
    epochs = 200
    learning_rate = 0.01


    # ------------------------------------------------------------------------------------------------
    # --- Data preparation ---

    # reading input data
    in_seq, out_seq, scalers = read_data_sequences(ind_lat, ind_lon, in_vars, out_vars, train_window)

    test_data_size = round(len(in_seq) * test_data_frac)

    # Split into training and test data
    train_data_in = in_seq[:-test_data_size]
    train_data_out = out_seq[:-test_data_size]

    test_data_in = in_seq[-test_data_size:]
    test_data_out = out_seq[-test_data_size:]

    # Create the training data set
    trainloader = DataLoader(Data(train_data_in, train_data_out), batch_size=batch_size, shuffle=True,
                              num_workers=8, drop_last=True)
    testloader = DataLoader(Data(test_data_in, test_data_out), batch_size=1, shuffle=False,
                             num_workers=8, drop_last=True)


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
            optimizer.step()                                            # Perform weight updates
        losses_all.append(single_loss.item())

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    # plt.figure(figsize=(10, 10))
    # plt.plot(range(epochs), losses_all)
    # plt.ylabel('Loss')
    # plt.xlabel('epoch')
    # plt.tight_layout()
    # plt.show()


    # ------------------------------------------------------------------------------------------------
    # --- Model evaluation ---

    model.eval()        # switch from training to evaluation mode

    preds = np.full((len(testloader),6),np.nan)

    for i,(seq, labels) in enumerate(testloader):

        # deactivates automatic gradient computation -> reduce memory and speed up calculations
        with torch.no_grad():
            model.reset_hidden()

            preds[i,:3] = model(seq)
            preds[i,3:] = labels

    for i, scaler in enumerate(scalers):

        preds[:,i] = scaler.inverse_transform((preds[:,i]).reshape(-1, 1)).flatten()
        preds[:,i+3] = scaler.inverse_transform((preds[:,i+3]).reshape(-1, 1)).flatten()


    cols = [var + '_pred' for var in out_vars] + [var + '_true' for var in out_vars]
    df_out = pd.DataFrame(preds, columns=cols)

    corr = df_out.corr()
    corrs = (corr['DMP_pred']['DMP_true'],
             corr['DMP_anom_lt_pred']['DMP_anom_lt_true'],
             corr['DMP_anom_st_pred']['DMP_anom_st_true'])
    title = 'R (DMP): %.3f     R (DMP_anom_lt): %.3f     R(DMP_anom_st): %.3f' % corrs


    df_out.plot()
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------------------------------
    # --- Vizualize output ---

    # plt.figure(figsize=(15,5))
    # plt.title('Month vs Passenger')
    # plt.ylabel('Total Passengers')
    # plt.xlabel('Months')
    # plt.grid(True)
    # plt.autoscale(axis='x', tight=True)
    #
    # plt.plot(all_data)
    #
    # x = np.arange(len(all_data)-test_data_size, len(all_data))
    # plt.plot(x, actual_predictions)
    #
    # plt.show()


if __name__=='__main__':

    # test_lstm()
    #
    i_lat = 350
    i_lon = 350

    df = read_data(i_lat=i_lat, i_lon=i_lon)

    variables = ['DMP','SFMC', 'TSOIL1', 'SWLAND', 'LWLAND']

    df[variables].plot()
    plt.tight_layout()
    plt.show()

    # for i_lat in [350, 700, 1050]:
    #     for i_lon in [0, 350, 700, 1050]:
    #         create_crosscorr_plot(i_lat=i_lat, i_lon=i_lon)

            # fout = '/Users/u0116961/Documents/work/machine_learning/DMP_prediction/ts_%i_%i.png' % (i_lat, i_lon)
            # f = plt.figure(figsize=(10,5), dpi=300)
            # read_data(i_lat=i_lat, i_lon=i_lon)[['DMP','SFMC', 'TSOIL1', 'SWLAND', 'LWLAND']].plot(ax=plt.gca())
            # plt.ylim(-120,180)
            # f.savefig(fout, bbox_inches='tight')
            # plt.close()

    # x, y = read_data_sequences()

# 350 / 1050
# 350 / 350