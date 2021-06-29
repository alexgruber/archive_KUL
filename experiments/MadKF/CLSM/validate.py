import warnings
warnings.filterwarnings("ignore")

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

from sklearn.linear_model import LinearRegression

from pyldas.interface import LDAS_io
from myprojects.readers.ascat import HSAF_io
from myprojects.readers.insitu import ISMN_io

from myprojects.timeseries import calc_anom

# from netCDF4 import Dataset

def insitu_evaluation(root):

    result_file = root / 'insitu.csv'

    ts_ana = LDAS_io('ObsFcstAna', 'US_M36_SMAP_TB_DA_scaled_4K_obserr').timeseries['obs_obs']
    t_ana = pd.DatetimeIndex(ts_ana.time.values).sort_values()

    ismn = ISMN_io()

    rmsd_root = 'US_M36_SMAP_TB_DA_scl_'
    rmsd_exps = [x.name.split(rmsd_root)[1] for x in Path('/Users/u0116961/data_sets/LDASsa_runs').glob('*RMSD*')]
    names = ['open_loop', 'DA_4K_obserr'] + \
            [f'SMAP_it{i}{j}' for i in  range(1,5) for j in  range(1,4)] + \
            rmsd_exps
            # [f'SMOS40_it61{i}' for i in range(3,6)] + \
    runs = ['US_M36_SMAP_TB_OL_scaled_4K_obserr', 'US_M36_SMAP_TB_DA_scaled_4K_obserr'] + \
           [f'US_M36_SMAP_TB_MadKF_DA_it{i}{j}' for i in  range(1,5) for j in  range(1,4)] + \
           [rmsd_root + exp for exp in rmsd_exps]
           # [f'US_M36_SMOS40_TB_MadKF_DA_it61{i}' for i in range(3,6)] + \

    tss = [LDAS_io('xhourly', run).timeseries for run in runs]

    variables = ['sm_surface','sm_rootzone','sm_profile']
    modes = ['absolute','longterm','shortterm']

    # ismn.list = ismn.list.iloc[301::]

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
                else:
                    ts_ref = calc_anom(ts_insitu[var], longterm=(mode=='longterm')).dropna()

                for run,ts_model in zip(names,tss):

                    res['len_' + mode + '_' + var] = 0
                    res['corr_' + run + '_' + mode + '_' + var] = np.nan
                    res['rmsd_' + run + '_' + mode + '_' + var] = np.nan
                    res['ubrmsd_' + run + '_' + mode + '_' + var] = np.nan

                    if len(ts_ref) < 10:
                        continue

                    ind = (ts_model['snow_mass'].isel(lat=row, lon=col).values == 0)&\
                          (ts_model['soil_temp_layer1'].isel(lat=row, lon=col).values > 277.15)
                    ts_mod = ts_model[var].isel(lat=row, lon=col).to_series().loc[ind]
                    if len(ts_mod) < 10:
                        continue

                    ts_mod.index += pd.to_timedelta('2 hours')
                    # TODO: Make sure that time of netcdf file is correct!!

                    ind_obs = np.bitwise_or.reduce(~np.isnan(ts_ana[:, :, row, col].values), 1)

                    if mode == 'absolute':
                        ts_mod = ts_mod.dropna()
                    else:
                        ts_mod = calc_anom(ts_mod, longterm=mode=='longterm').dropna()

                    tmp = pd.DataFrame({1: ts_ref, 2: ts_mod}).reindex(t_ana[ind_obs]).dropna()
                    res['len_' + mode + '_' + var] = len(tmp)

                    if len(tmp) < 10:
                        continue

                    r,p = pearsonr(tmp[1],tmp[2])
                    res['corr_' + run +'_' + mode + '_' + var] = r if (r > 0) & (p < 0.05) else np.nan
                    res['rmsd_' + run +'_' + mode + '_' + var] = np.sqrt(((tmp[1]-tmp[2])**2).mean())
                    res['ubrmsd_' + run +'_' + mode + '_' + var] = np.sqrt((((tmp[1]-tmp[1].mean())-(tmp[2]-tmp[2].mean()))**2).mean())

        if not result_file.exists():
            res.to_csv(result_file, float_format='%0.4f')
        else:
            res.to_csv(result_file, float_format='%0.4f', mode='a', header=False)


def lonlat2gpi(lon,lat,gpi_list):

    rdiff = np.sqrt((gpi_list.lon - lon)**2 + (gpi_list.lat - lat)**2)
    return gpi_list.iloc[np.where((rdiff - rdiff.min()) < 0.0001)[0][0]].name

def TCA_insitu_evaluation(root):

    result_file = root / 'insitu_TCA.csv'

    rmsd_root = 'US_M36_SMAP_TB_DA_scl_'
    rmsd_exps = [x.name.split(rmsd_root)[1] for x in Path('/Users/u0116961/data_sets/LDASsa_runs').glob('*RMSD*')]
    names = ['open_loop', 'DA_4K_obserr'] + \
            [f'SMAP_it{i}{j}' for i in  range(1,5) for j in  range(1,4)] + \
            rmsd_exps
            # [f'SMOS40_it61{i}' for i in range(3,6)] + \
    runs = ['US_M36_SMAP_TB_OL_scaled_4K_obserr', 'US_M36_SMAP_TB_DA_scaled_4K_obserr'] + \
           [f'US_M36_SMAP_TB_MadKF_DA_it{i}{j}' for i in  range(1,5) for j in  range(1,4)] + \
           [rmsd_root + exp for exp in rmsd_exps]
           # [f'US_M36_SMOS40_TB_MadKF_DA_it61{i}' for i in range(3,6)] + \

    tss = [LDAS_io('xhourly', run).timeseries for run in runs]
    # for run in runs:
    #     try:
    #         LDAS_io('xhourly', run).timeseries
    #     except:
    #         print(run)

    ts_ana = LDAS_io('ObsFcstAna', 'US_M36_SMAP_TB_DA_scaled_4K_obserr').timeseries['obs_obs']
    t_ana = pd.DatetimeIndex(ts_ana.time.values).sort_values()

    ascat = HSAF_io()
    gpi_list = pd.read_csv(ascat.root / 'warp5_grid' / 'pointlist_warp_conus.csv',index_col=0)

    ismn = ISMN_io()

    variables = ['sm_surface','sm_rootzone','sm_profile']
    modes = ['absolute','longterm','shortterm']

    if result_file.exists():
        tmp_res = pd.read_csv(result_file, index_col=0)
        start = np.where((ismn.list.network == tmp_res.network.iloc[-1])&(ismn.list.station == tmp_res.station.iloc[-1]))[0][0]+1
        ismn.list = ismn.list.iloc[start:,:]

    for i, (meta, ts_insitu) in enumerate(ismn.iter_stations(surface_only=False)):
        if 'tmp_res' in locals():
            if (meta.network in tmp_res) & (meta.station in tmp_res):
                print(f'Skipping {i}')
                continue

        logging.info('%i/%i' % (i, len(ismn.list)))

        try:
            res = pd.DataFrame(meta.copy()).transpose()
            col = meta.ease_col
            row = meta.ease_row

            gpi = lonlat2gpi(meta.lon, meta.lat, gpi_list)

            ts_ascat = ascat.read(gpi)
            if ts_ascat is None:
                continue

            for var in variables:
                for mode in modes:

                    if mode == 'absolute':
                        ts_asc = ts_ascat.dropna()
                    else:
                        ts_asc = calc_anom(ts_ascat, longterm=(mode == 'longterm')).dropna()
                    ts_asc.name = 'ascat'
                    ts_asc = pd.DataFrame(ts_asc)

                    if mode == 'absolute':
                        ts_ins = ts_insitu[var].dropna()
                    else:
                        ts_ins = calc_anom(ts_insitu[var], longterm=(mode == 'longterm')).dropna()
                    ts_ins.name = 'insitu'
                    ts_ins = pd.DataFrame(ts_ins)

                    for run, ts_model in zip(names, tss):

                        ind = (ts_model['snow_mass'].isel(lat=row, lon=col).values == 0)&\
                              (ts_model['soil_temp_layer1'].isel(lat=row, lon=col).values > 277.15)

                        ts_model = ts_model[var].isel(lat=row, lon=col).to_series().loc[ind]
                        ts_model.index += pd.to_timedelta('2 hours')

                        ind_obs = np.bitwise_or.reduce(~np.isnan(ts_ana[:, :, row, col].values), 1)

                        if mode == 'absolute':
                            ts_mod = ts_model.reindex(t_ana[ind_obs]).dropna()
                        else:
                            ts_mod = calc_anom(ts_model.reindex(t_ana[ind_obs]), longterm=(mode == 'longterm')).dropna()
                        ts_mod.name = 'model'
                        ts_mod = pd.DataFrame(ts_mod)

                        matched = df_match(ts_mod, ts_asc, ts_ins, window=0.5)
                        data = ts_mod.join(matched[0]['ascat']).join(matched[1]['insitu']).dropna()

                        tc_res = TCA_calc(data, ref_ind=0)
                        corr = data.corr()

                        res['R2_model_ascat' + run + '_' + mode + '_' + var] = corr['model']['ascat']
                        res['R2_model_insitu' + run + '_' + mode + '_' + var] = corr['model']['insitu']
                        res['R2_ascat_insitu' + run + '_' + mode + '_' + var] = corr['ascat']['insitu']


                        res['R2_model_' + run + '_' + mode + '_' + var] = tc_res[0][0]
                        res['R2_ascat_' + run + '_' + mode + '_' + var] = tc_res[0][1]
                        res['R2_insitu_' + run + '_' + mode + '_' + var] = tc_res[0][2]

                        res['ubRMSE_model_' + run + '_' + mode + '_' + var] = tc_res[1][0]
                        res['ubRMSE_ascat_' + run + '_' + mode + '_' + var] = tc_res[1][1]
                        res['ubRMSE_insitu_' + run + '_' + mode + '_' + var] = tc_res[1][2]

                        res['beta_ascat_' + run + '_' + mode + '_' + var] = tc_res[2][1]
                        res['beta_insitu_' + run + '_' + mode + '_' + var] = tc_res[2][2]

                        res['len_' + mode + '_' + var] = len(data)

            if not result_file.exists():
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

def filter_diagnostics_evaluation(root):

    result_file = root / 'filter_diagnostics.nc'

    rmsd_root = 'US_M36_SMAP_TB_DA_scl_'
    rmsd_exps = [x.name.split(rmsd_root)[1] for x in Path('/Users/u0116961/data_sets/LDASsa_runs').glob('*RMSD*')]
    names = ['open_loop', 'DA_4K_obserr'] + \
            [f'SMAP_it{i}{j}' for i in  range(1,5) for j in  range(1,4)] + \
            rmsd_exps
            # [f'SMOS40_it61{i}' for i in range(3,6)] + \
    runs = ['US_M36_SMAP_TB_OL_scaled_4K_obserr', 'US_M36_SMAP_TB_DA_scaled_4K_obserr'] + \
           [f'US_M36_SMAP_TB_MadKF_DA_it{i}{j}' for i in  range(1,5) for j in  range(1,4)] + \
           [rmsd_root + exp for exp in rmsd_exps]
           # [f'US_M36_SMOS40_TB_MadKF_DA_it61{i}' for i in range(3,6)] + \

    tss = [LDAS_io('ObsFcstAna',run) for run in runs]

    # runs = OrderedDict([(1,iter1.timeseries),
    #                     (2,iter2.timeseries),
    #                     (3,iter3.timeseries),
    #                     (4,iter4.timeseries)])

    tags = ['innov_autocorr','norm_innov_mean','norm_innov_var','norm_innov_mean_corr','norm_innov_var_corr']

    lons = np.unique(tss[0].grid.tilecoord['com_lon'].values)
    lats = np.unique(tss[0].grid.tilecoord['com_lat'].values)[::-1]

    species = tss[0].timeseries['species'].values

    # ---- Correction for obs_pert scaling ----
    tg = LDAS_io().grid.tilegrids
    tc = LDAS_io().grid.tilecoord
    ind_col = tc.i_indg.values - tg.loc['domain', 'i_offg']
    ind_row = tc.j_indg.values - tg.loc['domain', 'j_offg']
    root_smap = Path('/Users/u0116961/Documents/work/MadKF/CLSM/SMAP')
    root_smos = Path('/Users/u0116961/Documents/work/MadKF/CLSM/SMOS40')

    # obs_err.loc[:, 'obs_var_spc%i' % spc] = ensvar['fcst_var_spc%i' % spc] * mse['mse_obs_spc%i' % spc] / mse['mse_fcst_spc%i' % spc]
    # obs_mse = obs_var * fcst_mse / fcst_var

    with ncfile_init(result_file, lats, lons, list(range(len(runs))), species, tags) as ds:

        for i_run,run in enumerate(tss):
            for i_spc,spc in enumerate(species):

                logging.info('run %i, species %i' % (i_run,i_spc))

                scl_arr = np.full((83, 173), np.nan)
                if 'SMAP' in names[i_run]:
                    it, mode = int(names[i_run][-2]), int(names[i_run][-1])
                    if it == 1:
                        tmproot = root_smap / f'iter_{it}{mode}_init'
                        ensvar = pd.read_csv(tmproot / 'result_files' / 'ens_var.csv', index_col=0)
                        mse = pd.read_csv(tmproot / 'result_files' / 'mse.csv', index_col=0)
                    else:
                        tmproot = root_smap / f'iter_{it-1}{mode}'
                        ensvar = pd.read_csv(tmproot / 'result_files' / 'ens_var.csv', index_col=0)
                        mse = pd.read_csv(tmproot / 'result_files' / 'mse_corrected.csv', index_col=0)

                    scl_arr[ind_row, ind_col] = (mse[f'mse_fcst_spc{spc}'] / ensvar[f'fcst_var_spc{spc}']).values
                scl_arr[np.isnan(scl_arr)] = 1

                innov = run.timeseries['obs_obs'].isel(species=i_spc) - run.timeseries['obs_fcst'].isel(species=i_spc)

                ds['innov_autocorr'][:,:,i_run,i_spc] = calc_iac_spc(innov)

                ds['norm_innov_mean'][:,:,i_run,i_spc] = \
                    (innov / np.sqrt(run.timeseries['obs_obsvar'].isel(species=i_spc) + run.timeseries['obs_fcstvar'].isel(species=i_spc))).mean(dim='time').values
                ds['norm_innov_var'][:,:,i_run,i_spc] = \
                    (innov / np.sqrt(run.timeseries['obs_obsvar'].isel(species=i_spc) + run.timeseries['obs_fcstvar'].isel(species=i_spc))).var(dim='time').values

                if 'SMAP' in names[i_run]:
                    ds['norm_innov_mean_corr'][:,:,i_run,i_spc] = \
                        (innov / np.sqrt(run.timeseries['obs_obsvar'].isel(species=i_spc) * scl_arr + run.timeseries['obs_fcstvar'].isel(species=i_spc))).mean(dim='time').values
                    ds['norm_innov_var_corr'][:,:,i_run,i_spc] = \
                        (innov / np.sqrt(run.timeseries['obs_obsvar'].isel(species=i_spc) * scl_arr + run.timeseries['obs_fcstvar'].isel(species=i_spc))).var(dim='time').values


def calc_iac_spc(innov):

    iac = np.full(innov.shape[1::], np.nan)
    nlats, nlons = iac.shape

    for i_lat in range(nlats):
        for i_lon in range(nlons):

            ts = innov.isel(lat=i_lat, lon=i_lon).to_pandas().dropna()
            if len(ts) == 0:
                continue
            ts_l = ts.copy()

            lags = (ts.index[1::] - ts.index[0:-1]).value_counts().sort_index().index
            lags = lags[lags <= '5 days']

            ac = []
            for lag in lags:
                ts_l.index = ts.index + lag
                ac.append(ts.corr(ts_l))

            try:
                f = LinearRegression().fit(lags.total_seconds().values.reshape(-1, 1), np.array(ac).reshape(-1, 1))
                iac[i_lat,i_lon] = f.predict([[48 * 3600]])[0, 0]
            except:
                continue
    return iac

def validate_all():

    # iteration = 4

    # root = Path(f'/Users/u0116961/Documents/work/MadKF/CLSM/validation/iter_{iteration}')
    root = Path(f'/Users/u0116961/Documents/work/MadKF/CLSM/validation')

    if not root.exists():
        Path.mkdir(root, parents=True)

    insitu_evaluation(root)
    TCA_insitu_evaluation(root)
    # filter_diagnostics_evaluation(root)


if __name__ == '__main__':
    validate_all()




