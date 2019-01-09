
import os

import numpy as np
import pandas as pd
import xarray as xr

from netCDF4 import Dataset

from pytesmo.time_series.filtering import moving_average

'''
activate dev
python
import sys
sys.path.append(r'I:\python')
from validation_good_practice.data_readers.amsr2 import reshuffle_amsr2
reshuffle_amsr2()
'''

def mask_vod():

    dir_in = r'D:\data_sets\AMSR2\timeseries_w_vod'
    dir_out = r'D:\data_sets\AMSR2\timeseries'

    files = os.listdir(dir_in)

    for fname in files:
        df = pd.read_csv(os.path.join(dir_in,fname),index_col=0)

        df.index = pd.to_datetime(df['vod'].index)
        df['vod_ma'] = moving_average(df['vod'], window_size=35)
        Ser = df[df['vod_ma']<=0.6]['sm']

        if len(Ser) > 10:
            Ser.to_csv(os.path.join(dir_out,fname))

def reshuffle_amsr2():

    # Load RFI maps
    rfi2015 = np.load(r"D:\data_sets\AMSR2\RFI_MAPS\ALL_AMSR2_D2015.npy")
    rfi2016 = np.load(r"D:\data_sets\AMSR2\RFI_MAPS\ALL_AMSR2_D2016.npy")
    rfi2017 = np.load(r"D:\data_sets\AMSR2\RFI_MAPS\ALL_AMSR2_D2017.npy")
    rfi2018 = np.load(r"D:\data_sets\AMSR2\RFI_MAPS\ALL_AMSR2_D2018.npy")

    # Collect all nc files
    nc_files = []
    for root, dirs, files in os.walk(r'D:\data_sets\AMSR2\S3_VEGC'):
        for f in files:
            if f.find('.nc') != -1:
                nc_files.append(os.path.join(root, f))
    dates = pd.to_datetime([f[-11:-3] for f in nc_files])

    # get a list of all CONUS gpis
    gpi_lut = pd.read_csv(r"D:\data_sets\EASE2_grid\grid_lut.csv", index_col=0)
    ease_gpis = gpi_lut.index.values

    # Array with ALL possible dates and ALL CONUS gpis
    sm_arr = np.full((len(dates),len(ease_gpis)), np.nan)
    vod_arr = np.full((len(dates),len(ease_gpis)), np.nan)
    sec_arr = np.full((len(dates),len(ease_gpis)), np.nan)

    # Fill in result array from orbit files
    for i, f in enumerate(nc_files):
        print "%i / %i" % (i, len(nc_files))

        yr = dates[i].year

        ds = Dataset(f)
        if i == 0:
            lats = ds.variables['LAT'][:]
            lons = ds.variables['LON'][:]

        for res_ind, ease_gpi in enumerate(ease_gpis):
            ind_lat = np.where(lats == gpi_lut.loc[ease_gpi,'amsr2_lat'])[0][0]
            ind_lon = np.where(lons == gpi_lut.loc[ease_gpi,'amsr2_lon'])[0][0]

            if yr == 2015:
                band = rfi2015[ind_lat,ind_lon]
            elif yr == 2016:
                band = rfi2016[ind_lat,ind_lon]
            elif yr == 2017:
                band = rfi2017[ind_lat,ind_lon]
            else:
                band = rfi2018[ind_lat,ind_lon]

            if np.isnan(band) | (band == 0):
                continue

            tsurf = ds.variables['TSURF'][ind_lat, ind_lon]
            if band == 1:
                sm = ds.variables['SM_069'][ind_lat, ind_lon]
                vod = ds.variables['VOD_069'][ind_lat, ind_lon]
            elif band == 2:
                sm = ds.variables['SM_073'][ind_lat, ind_lon]
                vod = ds.variables['VOD_073'][ind_lat, ind_lon]
            else:
                sm = ds.variables['SM_107'][ind_lat, ind_lon]
                vod = ds.variables['VOD_107'][ind_lat, ind_lon]

            if (sm <= 0.)|(sm >= 1.)|(tsurf <=275.15):
                continue

            sm_arr[i, res_ind] = sm
            vod_arr[i, res_ind] = vod
            sec_arr[i, res_ind] = ds.variables['SCANTIME'][ind_lat, ind_lon]

        ds.close()

    sec_arr[np.isnan(sec_arr)] = 0.

    # Write out valid time series of all CONIS GPIS into separate .csv files
    dir_out = r'D:\data_sets\AMSR2\timeseries_w_vod'
    for i, gpi in enumerate(ease_gpis):
        idx = dates + pd.to_timedelta(sec_arr[:,i],unit='s').round('min')
        df = pd.DataFrame({'sm': sm_arr[:,i], 'vod':vod_arr[:,i]},index=idx).dropna()
        if len(df) > 0:
            fname = os.path.join(dir_out, '%i.csv' % gpi)
            df.to_csv(fname,float_format='%.4f')

if __name__=='__main__':
    mask_vod()