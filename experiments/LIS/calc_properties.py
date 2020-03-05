

import numpy as np
import pandas as pd
import rasterio as rio

from netCDF4 import Dataset, num2date



def calc_avg_timeseries():

    with rio.open('/data_sets/LIS/NoahMP_belgium/mask_demer.tif') as ds:
        mask_demer = np.flipud(ds.read()[0, :, :])
    with rio.open('/data_sets/LIS/NoahMP_belgium/mask_ourthe.tif') as ds:
        mask_ourthe = np.flipud(ds.read()[0, :, :])

    with  Dataset('/data_sets/LIS/ASCAT/timeseries.nc') as ascat:
        dates = pd.to_datetime(num2date(ascat['time'][:], units=ascat['time'].units))

        avg = pd.DataFrame(index=dates)

        res_asc_dem = avg.copy()
        res_asc_our = avg.copy()
        res_noah_dem = avg.copy()
        res_noah_our = avg.copy()

        ind = np.where(mask_demer == 1)
        for i, (row,col) in enumerate(zip(ind[0],ind[1])):
            print('%i / %i' % (i+1, len(ind[0])))
            res_asc_dem.loc[:,i] = ascat.variables['SoilMoisture'][:,row,col]
        res_asc_dem.replace(-9999, np.nan, inplace=True)
        avg.loc[:,'ascat_demer'] = res_asc_dem.mean(axis=1)
        del res_asc_dem

        ind = np.where(mask_ourthe == 1)
        for i, (row,col) in enumerate(zip(ind[0],ind[1])):
            print('%i / %i' % (i+1, len(ind[0])))
            res_asc_our.loc[:,i] = ascat.variables['SoilMoisture'][:,row,col]
        res_asc_our.replace(-9999, np.nan, inplace=True)
        avg.loc[:, 'ascat_ourthe'] = res_asc_our.mean(axis=1)
        del res_asc_our

    with Dataset('/data_sets/LIS/NoahMP_belgium/timeseries.nc') as noah:

        ind = np.where(mask_demer == 1)
        for i, (row, col) in enumerate(zip(ind[0], ind[1])):
            print('%i / %i' % (i + 1, len(ind[0])))
            res_noah_dem.loc[:, i] = noah.variables['SoilMoisture'][:, row, col]
        res_noah_dem.replace(-9999, np.nan, inplace=True)
        avg.loc[:, 'noah_demer'] = res_noah_dem.mean(axis=1)
        del res_noah_dem

        ind = np.where(mask_ourthe == 1)
        for i, (row, col) in enumerate(zip(ind[0], ind[1])):
            print('%i / %i' % (i + 1, len(ind[0])))
            res_noah_our.loc[:, i] = noah.variables['SoilMoisture'][:, row, col]
        res_noah_our.replace(-9999, np.nan, inplace=True)
        avg.loc[:, 'noah_ourthe'] = res_noah_our.mean(axis=1)
        del res_noah_our

    avg.to_csv('/data_sets/LIS/catchment_averages.csv')


if __name__ == '__main__':

    calc_avg_timeseries()

