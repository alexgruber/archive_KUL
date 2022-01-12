
import numpy as np
import pandas as pd

from pathlib import Path
from netCDF4 import Dataset, num2date


class io(object):
    '''
    Recommended filters:

    AGB:
        AGB_err > 100

    SMOS_IC:
        Flags > 0
        RMSE > 8
        VOD_StdErr > 0.1
        Yearly avg VOD > 1.2
        Outliers (> 3xStdDev from 80d moving-average)

    LAI:
        Outliers (> 3xStdDev from 80d moving-average)

    MERRA2:
        TSOIL1 < 277.15
        SNOMAS > 0

    '''


    def __init__(self, name):

        root = Path('/Users/u0116961/data_sets/')

        self.lut = pd.read_csv(root / 'LUT_EASE25_MERRA2_SIF_South_America.csv', index_col=0)

        self.name = name
        if name == 'AGB':
            self.ds = Dataset(root / 'AGB' / 'resampled' / 'AGB_25km.nc')
        elif name == 'TCL':
            self.ds = Dataset(root / 'tree_cover_loss' / 'resampled' / 'TCL_25km.nc')
        elif name == 'LAI':
            self.ds = Dataset(root / 'COPERNICUS_LAI' / 'COPERNICUS_LAI_timeseries.nc')
            self.ds_img = Dataset(root / 'COPERNICUS_LAI' / 'COPERNICUS_LAI_images.nc')
        elif name == 'SIF':
            self.ds = Dataset(root / 'SIF' / 'SIF_SouthAm_20180201_20210930_8day_0.25deg.nc')
            self.ds_img = self.ds
        elif name == 'SMOS_IC':
            self.ds = Dataset(root / 'SMOS_IC' / 'south_america_2010_2020' / 'SMOS_IC_timeseries.nc')
            self.ds_img = Dataset(root / 'SMOS_IC' / 'south_america_2010_2020' / 'SMOS_IC_images.nc')
        elif name == 'MERRA2':
            self.ds = Dataset(root / 'MERRA2' / 'south_america_2010_2020' / 'MERRA2_timeseries.nc')
            self.ds_img = Dataset(root / 'MERRA2' / 'south_america_2010_2020' / 'MERRA2_images.nc')
        else:
            valid = ['AGB', 'TCL', 'LAI', 'SIF', 'SMOS_IC', 'MERRA2']
            print(f'Unknown data set. Allowed: {", ".join(valid)}')

        self.lon, self.lat = np.meshgrid(self.ds['lon'][:].data, self.ds['lat'][:].data)

        if 'time' in self.ds.variables.keys():
            self.dates = pd.DatetimeIndex(num2date(self.ds['time'][:], self.ds['time'].units,
                                          only_use_python_datetimes=True, only_use_cftime_datetimes=False))

    # Not used because it's way too slow and provides barely any benefit
    @staticmethod
    def remove_outliers(data, window_size=90):

        dt = pd.Timedelta(window_size/2., 'D')
        for t, val in data.iteritems():
            mean = data[t - dt:t + dt].dropna().mean()
            std = data[t - dt:t + dt].dropna().std()
            if abs(mean-val) > 3*std:
                data.loc[t] = np.nan


    def read_img(self, var, n=None, date_from=None, date_to=None, return_ind=False):

        if not hasattr(self, 'ds_img'):
            data = self.ds[var][:, :].data
        else:
            if n:
                if self.name == 'SIF':
                    data = self.ds_img[var][:, :, n].data
                else:
                    data = self.ds_img[var][n, :, :].data
            elif date_from and date_to:
                ind = np.where((self.dates>=date_from)&(self.dates<=date_to))[0]
                if self.name == 'SIF':
                    data = np.moveaxis(self.ds_img[var][:, :, ind].data, 2, 0)
                else:
                    data = self.ds_img[var][ind, :, :].data
            else:
                if self.name == 'SIF':
                    data = np.moveaxis(self.ds_img[var][:, :, :].data, 2, 0)
                else:
                    data = self.ds_img[var][:, :, :].data

        if self.name == 'SIF':
            data[data == -999] = np.nan
        else:
            data[data == -9999] = np.nan

        if var == 'T2M':
            data -= 273.15

        if return_ind:
            return data, ind
        else:
            return data


    def read(self, var, *args, latlon=False, date_from=None, date_to=None):

        if len(args) == 1:
            if self.name == 'MERRA2':
                row = self.lut.loc[args[0], 'row_merra']
                col = self.lut.loc[args[0], 'col_merra']
            if self.name == 'SIF':
                row = self.lut.loc[args[0], 'row_sif']
                col = self.lut.loc[args[0], 'col_sif']
            else:
                row = self.lut.loc[args[0], 'row_ease']
                col = self.lut.loc[args[0], 'col_ease']
        else:
            if latlon is False:
                row, col = args
            else:
                lat, lon = args
                idx = np.argmin((self.lut.lat-lat)**2 + (self.lut.lon-lon)**2)
                if self.name == 'MERRA2':
                    row = self.lut.loc[idx, 'row_merra']
                    col = self.lut.loc[idx, 'col_merra']
                if self.name == 'SIF':
                    row = self.lut.loc[idx, 'row_sif']
                    col = self.lut.loc[idx, 'col_sif']
                else:
                    row = self.lut.loc[idx, 'row_ease']
                    col = self.lut.loc[idx, 'col_ease']

        if len(self.ds[var].shape) == 2:
            data = self.ds[var][row, col].data
        else:
            if self.name == 'SIF':
                data = self.ds[var][row, col, :].data
                data[data == -999] = np.nan
            else:
                data = self.ds[var][:, row, col].data
                data[data == -9999] = np.nan
            data = pd.Series(data, self.dates)
            if var == 'T2M':
                data -= 273.15

            if date_from and date_to:
                data = data.loc[date_from:date_to]

        return data

    def close(self):
        self.ds.close()
        if hasattr(self, 'ds_img'):
            self.ds_img.close()

