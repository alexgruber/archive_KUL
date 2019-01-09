
import os
import numpy as np
import pandas as pd

from pytesmo.temporal_matching import df_match

class reader():

    def __init__(self):

        self.root = r'D:\data_sets'
        self.sensors = ['ASCAT','AMSR2','SMOS','SMAP','MERRA2']

    def read(self, gpi, sensors=None, return_anomaly=False, longterm=False, resample=False, match=False, dt=0):

        if sensors is None:
            sensors = self.sensors
        Ser_list = list()
        for sensor in sensors:
            fname = os.path.join(self.root,sensor,'timeseries', '%i.csv' % gpi)
            if os.path.exists(fname):
                Ser = pd.read_csv(fname, index_col=0, header=None, names=[sensor])
                Ser.index = pd.DatetimeIndex(Ser.index)
                if resample is True:
                    Ser = Ser.resample('1D').last()
                if sensor == 'ASCAT':
                    Ser /= 100.
            else:
                Ser = pd.Series(name=sensor)
            Ser_list.append(Ser)

        df = pd.concat(Ser_list,axis='columns')

        if return_anomaly is True:
            calc_anomaly(df, longterm=longterm)

        if match is True:
            df = collocate(df, dt=dt)
        return df

def calc_anomaly(in_df, longterm=False):

    df = in_df.copy()

    for col in df:
        df[col] = calc_anom(df[col], longterm=longterm)

    return df

def calc_anom(Ser, longterm=False):

    xSer = Ser.dropna().copy()
    if len(xSer) == 0:
        return xSer

    doys = xSer.index.dayofyear.values
    doys[xSer.index.is_leap_year & (doys > 59)] -= 1
    climSer = pd.Series(index=xSer.index,name=xSer.name)

    if longterm is True:
        climSer[:] = self.calc_clim(xSer)
    else:
        years = xSer.index.year
        for yr in np.unique(years):
            clim = calc_clim(xSer[years == yr])
            climSer[years == yr] = clim[doys[years == yr]].values

    return xSer - climSer

def calc_clim(Ser, window_size=35, n_min=17):

    xSer = Ser.dropna().copy()
    doys = xSer.index.dayofyear.values

    # in leap years, subtract 1 for all days after Feb 28
    doys[xSer.index.is_leap_year & (doys > 59)] -= 1

    clim_doys = np.arange(365) + 1
    clim = pd.Series(index=clim_doys)
    n_data = pd.Series(index=clim_doys)

    for doy in clim_doys:
        # Avoid artifacts at start/end of year
        tmp_doys = doys.copy()
        if doy < window_size / 2.:
            tmp_doys[tmp_doys > 365 - (np.ceil(window_size / 2.) - doy)] -= 365
        if doy > 365 - (window_size / 2. - 1):
            tmp_doys[tmp_doys < np.ceil(window_size / 2.) - (365 - doy)] += 365

        n_data[doy] = len(xSer[(tmp_doys >= doy - np.floor(window_size / 2.)) & \
                               (tmp_doys <= doy + np.floor(window_size / 2.))])

        if n_data[doy] >= n_min:
            clim[doy] = xSer[(tmp_doys >= doy - np.floor(window_size / 2.)) & \
                             (tmp_doys <= doy + np.floor(window_size / 2.))].values.mean()

    return clim

def collocate(df, dt=0):

    ref_df = pd.DataFrame(
        index=pd.date_range(df.index.min().date(), df.index.max().date()) + pd.Timedelta(dt, 'h'))
    args = [df[col].dropna() for col in df]
    matched = df_match(ref_df, *args, window=0.5)
    for i, col in enumerate(df):
        ref_df[col] = matched[i][col]

    return ref_df

if __name__=='__main__':

    gpi = 72512

    io = reader()

    df = io.read(72512,sensors=['ASCAT','AMSR2'])


    # dts = np.arange(0,24,0.5)
    # n_matches = np.full(len(dts),0)
    #
    # for i,dt in enumerate(dts):
    #
    #     df = io.read(gpi,sensors=['ASCAT','AMSR2','SMOS','SMAP','MERRA2'], match=True, dt=dt)
    #     n_matches[i] = len(df.dropna())
    #
    # # print n_matches
    # print df.index.min(), df.index.max()
    #
    # print n_matches.min(), n_matches.max(), dts[n_matches == n_matches.max()]