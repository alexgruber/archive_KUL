

import os
import logging

import numpy as np
import pandas as pd

from pathlib import Path

from netCDF4 import Dataset
from collections import OrderedDict

from scipy.stats import pearsonr
from pytesmo.temporal_matching import df_match
# from pytesmo.metrics import tcol_snr as TCA
from validation_good_practice.ancillary.metrics import TCA_calc

from pyldas.interface import LDAS_io
from myprojects.readers.ascat import HSAF_io
from myprojects.readers.insitu import ISMN_io

from myprojects.timeseries import calc_anomaly

# from netCDF4 import Dataset

def insitu_evaluation(root, iteration):

    result_file = root / 'insitu.csv'

    noDA = LDAS_io('xhourly',           'US_M36_SMOS40_noDA_cal_scaled')
    DA_const_err = LDAS_io('xhourly',   'US_M36_SMOS40_DA_cal_scaled')
    DA_MadKF = LDAS_io('xhourly',       'US_M36_SMOS40_TB_MadKF_DA_it%i' % iteration)

    t_ana = pd.DatetimeIndex(LDAS_io('ObsFcstAna', 'US_M36_SMOS40_DA_cal_scaled').timeseries.time.values).sort_values()

    ismn = ISMN_io()

    runs = ['noDA', 'DA_const_err','DA_madkf']
    tss = [noDA.timeseries, DA_const_err.timeseries, DA_MadKF.timeseries.transpose('lat','lon','time')]

    variables = ['sm_surface','sm_rootzone','sm_profile']
    modes = ['absolute','longterm','shortterm']

    # ismn.list = ismn.list.iloc[101::]

    i = 0
    for meta, ts_insitu in ismn.iter_stations(surface_only=False):
        i += 1
        logging.info('%i/%i' % (i, len(ismn.list)))

        if ts_insitu is None:
            continue

        res = pd.DataFrame(meta.copy()).transpose()
        col = meta.ease_col
        row = meta.ease_row

        for var in variables:
            for mode in modes:

                if mode == 'absolute':
                    ts_ref = ts_insitu[var].dropna()
                elif mode == 'mean':
                    ts_ref = calc_anomaly(ts_insitu[var], mode).dropna()
                else:
                    ts_ref = calc_anomaly(ts_insitu[var], method='moving_average', longterm=(mode=='longterm')).dropna()

                if len(ts_ref) < 10:
                    continue

                ts_ref.index = ts_ref.index.tz_convert(None)

                for run,ts_model in zip(runs,tss):

                    ind = (ts_model['snow_mass'][row, col].values == 0)&(ts_model['soil_temp_layer1'][row, col].values > 277.15)
                    ts_mod = ts_model[var][row, col].to_series().loc[ind]
                    if len(ts_mod) < 10:
                        continue

                    ts_mod.index += pd.to_timedelta('2 hours')
                    # TODO: Make sure that time of netcdf file is correct!!

                    if mode == 'absolute':
                        ts_mod = ts_mod.dropna()
                    else:
                        ts_mod = calc_anomaly(ts_mod, method='moving_average', longterm=mode=='longterm').dropna()

                    tmp = pd.DataFrame({1: ts_ref, 2: ts_mod}).loc[t_ana,:].dropna()
                    res['len_' + mode + '_' + var] = len(tmp)

                    if len(tmp) < 10:
                        continue

                    r,p = pearsonr(tmp[1],tmp[2])

                    res['corr_' + run +'_' + mode + '_' + var] = r if (r > 0) & (p < 0.01) else np.nan
                    res['rmsd_' + run +'_' + mode + '_' + var] = np.sqrt(((tmp[1]-tmp[2])**2).mean())
                    res['ubrmsd_' + run +'_' + mode + '_' + var] = np.sqrt((((tmp[1]-tmp[1].mean())-(tmp[2]-tmp[2].mean()))**2).mean())


        if (os.path.isfile(result_file) == False):
            res.to_csv(result_file, float_format='%0.4f')
        else:
            res.to_csv(result_file, float_format='%0.4f', mode='a', header=False)


def lonlat2gpi(lon,lat,gpi_list):

    rdiff = np.sqrt((gpi_list.lon - lon)**2 + (gpi_list.lat - lat)**2)
    return gpi_list.iloc[np.where((rdiff - rdiff.min()) < 0.0001)[0][0],0]

def TCA_insitu_evaluation(root, iteration):

    result_file = root / 'insitu_TCA.csv'

    noDA = LDAS_io('xhourly', 'US_M36_SMOS40_noDA_cal_scaled')
    DA_const_err = LDAS_io('xhourly', 'US_M36_SMOS40_DA_cal_scaled')
    DA_MadKF = LDAS_io('xhourly', 'US_M36_SMOS40_TB_MadKF_DA_it%i' % iteration)

    t_ana = pd.DatetimeIndex(LDAS_io('ObsFcstAna', 'US_M36_SMOS40_DA_cal_scaled').timeseries.time.values).sort_values()

    ascat = HSAF_io()
    gpi_list = pd.read_csv('/data_sets/ASCAT/warp5_grid/pointlist_warp_conus.csv',index_col=0)

    ismn = ISMN_io()

    runs = ['noDA', 'DA_const_err','DA_madkf']
    tss = [noDA.timeseries, DA_const_err.timeseries, DA_MadKF.timeseries.transpose('lat','lon','time')]

    variables = ['sm_surface','sm_rootzone','sm_profile']
    modes = ['absolute','longterm','shortterm']

    for i, (meta, ts_insitu) in enumerate(ismn.iter_stations(surface_only=False)):
        logging.info('%i/%i' % (i, len(ismn.list)))

        try:

            res = pd.DataFrame(meta.copy()).transpose()
            col = meta.ease_col
            row = meta.ease_row

            gpi = lonlat2gpi(meta.lon, meta.lat, gpi_list)

            ts_ascat = ascat.read(gpi, resample_time=False)
            if ts_ascat is None:
                continue

            for var in variables:
                for mode in modes:

                    if mode == 'absolute':
                        ts_asc = ts_ascat.dropna()
                    else:
                        ts_asc = calc_anomaly(ts_ascat, method='moving_average', longterm=(mode == 'longterm')).dropna()
                    ts_asc.name = 'ascat'
                    ts_asc = pd.DataFrame(ts_asc)

                    if mode == 'absolute':
                        ts_ins = ts_insitu[var].dropna()
                    else:
                        ts_ins = calc_anomaly(ts_insitu[var], method='moving_average',
                                              longterm=(mode == 'longterm')).dropna()
                    ts_ins.name = 'insitu'
                    ts_ins = pd.DataFrame(ts_ins)
                    ts_ins.index = ts_ins.index.tz_convert(None)

                    for run, ts_model in zip(runs, tss):

                        ind = (ts_model['snow_mass'][row, col].values == 0) & (
                                    ts_model['soil_temp_layer1'][row, col].values > 277.15)

                        ts_model = ts_model[var][row, col].to_series().loc[ind]
                        ts_model.index += pd.to_timedelta('2 hours')

                        if mode == 'absolute':
                            ts_mod = ts_model.loc[t_ana].dropna()
                        else:
                            ts_mod = calc_anomaly(ts_model.loc[t_ana], method='moving_average',
                                                  longterm=(mode == 'longterm')).dropna()
                        ts_mod.name = 'model'
                        ts_mod = pd.DataFrame(ts_mod)

                        matched = df_match(ts_mod, ts_asc, ts_ins, window=0.5)
                        data = ts_mod.join(matched[0][['ascat', ]]).join(matched[1][['insitu', ]]).dropna()

                        tc_res = TCA_calc(data, ref_ind=0)

                        res['R2_model_' + run + '_' + mode + '_' + var] = tc_res[0][0]
                        res['R2_ascat_' + run + '_' + mode + '_' + var] = tc_res[0][1]
                        res['R2_insitu_' + run + '_' + mode + '_' + var] = tc_res[0][2]

                        res['ubRMSE_model_' + run + '_' + mode + '_' + var] = tc_res[1][0]
                        res['ubRMSE_ascat_' + run + '_' + mode + '_' + var] = tc_res[1][1]
                        res['ubRMSE_insitu_' + run + '_' + mode + '_' + var] = tc_res[1][2]

                        res['beta_ascat_' + run + '_' + mode + '_' + var] = tc_res[2][1]
                        res['beta_insitu_' + run + '_' + mode + '_' + var] = tc_res[2][2]

                        res['len_' + mode + '_' + var] = len(data)

            if (os.path.isfile(result_file) == False):
                res.to_csv(result_file, float_format='%0.4f')
            else:
                res.to_csv(result_file, float_format='%0.4f', mode='a', header=False)

        except:
            continue


def ncfile_init(fname, lats, lons, runs, species, tags):

    ds = Dataset(fname, 'w', 'NETCDF4')

    dims = ['lat','lon','run','species']
    dimvals = [lats, lons, runs, species]
    chunksizes = [len(lats), len(lons), 1, 1]
    dtypes = ['float32','float32','uint8', 'uint8']

    for dim, dimval, chunksize, dtype in zip(dims, dimvals, chunksizes, dtypes):
        ds.createDimension(dim, len(dimval))
        ds.createVariable(dim, dtype, dimensions=[dim], chunksizes=[chunksize], zlib=True)
        ds.variables[dim][:] = dimval

    for tag in tags:
        if tag.find('innov') != -1:
            ds.createVariable(tag, 'float32', dimensions=dims, chunksizes=chunksizes, fill_value=-9999., zlib=True)
        else:
            ds.createVariable(tag, 'float32', dimensions=dims[0:-1], chunksizes=chunksizes[0:-1], fill_value=-9999., zlib=True)

    return ds

def filter_diagnostics_evaluation(root, iteration):

    result_file = root / 'filter_diagnostics.nc'

    iter1 = LDAS_io('ObsFcstAna','US_M36_SMOS40_DA_cal_scaled')
    iter2 = LDAS_io('ObsFcstAna','US_M36_SMOS40_TB_MadKF_DA_it%i' % 4)
    iter3 = LDAS_io('ObsFcstAna','US_M36_SMOS40_TB_MadKF_DA_it%i' % 5)
    # iter2 = LDAS_io('ObsFcstAna','US_M36_SMOS40_TB_MadKF_DA_it%i' % (iteration - 2))
    # iter3 = LDAS_io('ObsFcstAna','US_M36_SMOS40_TB_MadKF_DA_it%i' % (iteration - 1))
    iter4 = LDAS_io('ObsFcstAna','US_M36_SMOS40_TB_MadKF_DA_it%i' % iteration)

    runs = OrderedDict([(1,iter1.timeseries),
                        (2,iter2.timeseries),
                        (3,iter3.timeseries),
                        (4,iter4.timeseries)])

    tags = ['norm_innov_mean','norm_innov_var']

    lons = np.unique(iter1.grid.tilecoord['com_lon'].values)
    lats = np.unique(iter1.grid.tilecoord['com_lat'].values)[::-1]

    species = iter1.timeseries['species'].values

    with ncfile_init(result_file, lats, lons, [1,2,3,4], species, tags) as ds:

        for i_run,run in enumerate(runs):
            for i_spc,spc in enumerate(species):

                logging.info('run %i, species %i' % (i_run,i_spc))

                ds['norm_innov_mean'][:,:,i_run,i_spc] = ((runs[run]['obs_obs'].isel(species=i_spc)- runs[run]['obs_fcst'].isel(species=i_spc)) /
                                                          np.sqrt(runs[run]['obs_obsvar'].isel(species=i_spc) + runs[run]['obs_fcstvar'].isel(species=i_spc))).mean(dim='time').values
                ds['norm_innov_var'][:,:,i_run,i_spc] = ((runs[run]['obs_obs'].isel(species=i_spc) - runs[run]['obs_fcst'].isel(species=i_spc)) /
                                                         np.sqrt(runs[run]['obs_obsvar'].isel(species=i_spc) + runs[run]['obs_fcstvar'].isel(species=i_spc))).var(dim='time').values


if __name__ == '__main__':

    iteration = 51

    root = Path('/work/MadKF/CLSM/iter_%i/validation' % iteration)

    if not (root).exists():
        Path.mkdir(root, parents=True)

    # insitu_evaluation(root, iteration)
    # TCA_insitu_evaluation(root, iteration)
    filter_diagnostics_evaluation(root, iteration)



