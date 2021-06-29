
import platform

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from math import floor
from pathlib import Path
from itertools import repeat, combinations
from multiprocessing import Pool

from scipy.stats import pearsonr

import seaborn as sns
sns.set_context('talk', font_scale=0.8)

import matplotlib.pyplot as plt

from pyldas.interface import LDAS_io

from myprojects.readers.insitu import ISMN_io
from myprojects.readers.ascat import HSAF_io

from myprojects.timeseries import calc_anom
from myprojects.functions import merge_files

from pytesmo.time_series.anomaly import calc_climatology

from validation_good_practice.ancillary.paths import Paths
from validation_good_practice.plots import plot_ease_img

def run_ascat_eval(iteration, n_procs=1):

    res_path = Path(f'~/Documents/work/MadKF/CLSM/SM_err_ratio/validation').expanduser()
    if not res_path.exists():
        Path.mkdir(res_path, parents=True)

    it = repeat(iteration, n_procs)
    part = np.arange(n_procs) + 1
    parts = repeat(n_procs, n_procs)

    p = Pool(n_procs)
    p.starmap(run_ascat_eval_part, zip(it, part, parts))

    merge_files(res_path, pattern='ascat_eval_part*.csv', fname='ascat_eval.csv', delete=True)

def run_ascat_eval_part(iteration, part, parts, ref='ascat'):

    if platform.system() == 'Linux':
        stg = '/staging/leuven/stg_00024/OUTPUT/alexg'
        smap_path = Path('/staging/leuven/stg_00024/OUTPUT/alexg/data_sets/SMAP/timeseries')
    else:
        stg = None
        smap_path = Path('/Users/u0116961/data_sets/SMAP/timeseries')

    res_path = Path(f'~/Documents/work/MadKF/CLSM/SM_err_ratio/validation').expanduser()
    if not res_path.exists():
        Path.mkdir(res_path, parents=True)

    result_file = res_path / ('ascat_eval_part%i.csv' % part)

    tc_res = pd.read_csv('/Users/u0116961/Documents/work/validation_good_practice/CI80/ASCAT_SMAP_CLSM/result.csv', index_col=0)

    lut = pd.read_csv(Paths().lut, index_col=0)

    # Split grid cell list for parallelization
    subs = (np.arange(parts + 1) * len(lut) / parts).astype('int')
    subs[-1] = len(lut)
    start = subs[part - 1]
    end = subs[part]

    # Look-up table that contains the grid cells to iterate over
    lut = lut.iloc[start:end, :]

    rmsd_root = 'US_M36_SMAP_TB_DA_SM_ERR_'
    rmsd_exps = list(np.sort([x.name.split(rmsd_root)[1] for x in Path('/Users/u0116961/data_sets/LDASsa_runs').glob('*SM_ERR_scl_seas*')]))

    names = ['open_loop', 'OL_no_pcorr', 'DA_4K_obserr_seas'] + \
            rmsd_exps
    runs = ['US_M36_SMAP_TB_OL_scaled_4K_obserr', 'US_M36_SMAP_TB_OL_scaled_noPcorr', 'US_M36_SMAP_TB_DA_scl_seas_4K_obserr'] + \
           [rmsd_root + exp for exp in rmsd_exps]

    roots = [None for i in range(len(runs))]

    dss = [LDAS_io('xhourly', run, root=root).timeseries for run, root in zip(runs, roots)]
    grid = LDAS_io('ObsFcstAna', runs[0], root=stg).grid

    # t_ana = pd.DatetimeIndex(LDAS_io('ObsFcstAna', runs[0]).timeseries.time.values).sort_values()
    ds_obs_smap = (LDAS_io('ObsFcstAna', 'US_M36_SMAP_TB_DA_scl_seas_4K_obserr', root=stg).timeseries['obs_ana'])
    ds_obs_smos1 = (LDAS_io('ObsFcstAna', 'US_M36_SMOS40_TB_MadKF_DA_it613', root=stg).timeseries['obs_ana'])
    ds_obs_smos2 = (LDAS_io('ObsFcstAna', 'US_M36_SMOS40_TB_MadKF_DA_it615', root=stg).timeseries['obs_ana'])

    modes = ['absolute', 'longterm', 'shortterm']

    ascat = HSAF_io()

    for cnt, (gpi, data) in enumerate(lut.iterrows()):
        print('%i / %i' % (cnt, len(lut)))

        col = int(data.ease2_col - grid.tilegrids.loc['domain', 'i_offg'])
        row = int(data.ease2_row - grid.tilegrids.loc['domain', 'j_offg'])

        res = pd.DataFrame(index=(gpi,))
        res['col'] = int(data.ease2_col)
        res['row'] = int(data.ease2_row)
        res['lcol'] = col
        res['lrow'] = row

        try:
            ts_ascat = ascat.read(data['ascat_gpi']).resample('1d').mean().dropna()
            ts_ascat = ts_ascat[~ts_ascat.index.duplicated(keep='first')]
            ts_ascat.name = 'ASCAT'
        except:
            continue

        # if ref == 'ascat':
        # else:
        #     try:
        #         ts_smap = pd.read_csv(smap_path / f'{gpi}.csv',
        #                               index_col=0, parse_dates=True, names=('smap',))['smap'].resample('1d').mean().dropna()
        #     except:
        #         continue
        # t_df_smos1 = ds_obs_smos1.sel(species=[1, 2]).isel(lat=row, lon=col).to_pandas()
        # t_df_smos2 = ds_obs_smos2.sel(species=[1, 2]).isel(lat=row, lon=col).to_pandas()
        # t_ana_smos1 = t_df_smos1[~np.isnan(t_df_smos1[1]) | ~np.isnan(t_df_smos1[2])].resample('1d').mean().index
        # t_ana_smos2 = t_df_smos2[~np.isnan(t_df_smos2[1]) | ~np.isnan(t_df_smos2[2])].resample('1d').mean().index
        # t_ana_smos_u = t_ana_smos1.union(t_ana_smos2)
        # t_ana_smos_i = t_ana_smos1.intersection(t_ana_smos2)

        t_df_smap = ds_obs_smap.sel(species=[1, 2]).isel(lat=row, lon=col).to_pandas()
        t_ana_smap_u = t_df_smap[~np.isnan(t_df_smap[1]) | ~np.isnan(t_df_smap[2])].resample('1d').mean().index
        # t_ana_smap_i = t_ana_smap_u[(t_ana_smap_u >= t_ana_smos_i.min()) & (t_ana_smap_u <= t_ana_smos_i.max())]
        t_ana_smap_i = t_ana_smap_u

        var = 'sm_surface'
        for mode in modes:

            try:
                if mode == 'longterm':
                    r_asc = np.sqrt(tc_res.loc[gpi, 'r2_grid_anom_lt_m_ASCAT_tc_ASCAT_SMAP_CLSM'])
                    r_mod = np.sqrt(tc_res.loc[gpi, 'r2_grid_anom_lt_m_CLSM_tc_ASCAT_SMAP_CLSM'])
                elif mode == 'longshortterm':
                    r_asc = np.sqrt(tc_res.loc[gpi, 'r2_grid_anom_lst_m_ASCAT_tc_ASCAT_SMAP_CLSM'])
                    r_mod = np.sqrt(tc_res.loc[gpi, 'r2_grid_anom_lst_m_CLSM_tc_ASCAT_SMAP_CLSM'])
                elif mode == 'shortterm':
                    r_asc = np.sqrt(tc_res.loc[gpi, 'r2_grid_anom_st_m_ASCAT_tc_ASCAT_SMAP_CLSM'])
                    r_mod = np.sqrt(tc_res.loc[gpi, 'r2_grid_anom_st_m_CLSM_tc_ASCAT_SMAP_CLSM'])
                else:
                    r_asc = np.sqrt(tc_res.loc[gpi, 'r2_grid_abs_m_ASCAT_tc_ASCAT_SMAP_CLSM'])
                    r_mod = np.sqrt(tc_res.loc[gpi, 'r2_grid_abs_m_CLSM_tc_ASCAT_SMAP_CLSM'])
            except:
                r_asc = np.nan
                r_mod = np.nan

            if mode == 'longshortterm':
                ts_ref = calc_anom(ts_ascat.copy(), longterm=True).dropna()
            elif mode == 'shortterm':
                ts_ref = calc_anom(ts_ascat.copy(), longterm=False).dropna()
            elif mode == 'longterm':
                ts_ref = (calc_anom(ts_ascat.copy(), longterm=True) - calc_anom(ts_ascat.copy(), longterm=False)).dropna()
            else:
                ts_ref = ts_ascat.dropna()

            # if ref == 'ascat':
            # else:
            #     if mode == 'absolute':
            #         ts_ref = ts_smap.copy()
            #     else:
            #         ts_ref = calc_anom(ts_smap.copy(), longterm=(mode == 'longterm')).dropna()


            for run, ts_model in zip(names, dss):

                if 'SMOS40' in run:
                    t_ana_u, t_ana_i = t_ana_smos_u, t_ana_smos_i
                else:
                    t_ana_u, t_ana_i = t_ana_smap_u, t_ana_smap_i

                ind = (ts_model['snow_mass'][:, row, col].values == 0) & (
                        ts_model['soil_temp_layer1'][:, row, col].values > 277.15)
                ts_mod = ts_model[var][:, row, col].to_series().loc[ind]
                ts_mod.index += pd.to_timedelta('2 hours')

                if mode == 'longshortterm':
                    ts_mod = calc_anom(ts_mod.copy(), longterm=True).dropna()
                elif mode == 'shortterm':
                    ts_mod = calc_anom(ts_mod.copy(), longterm=False).dropna()
                elif mode == 'longterm':
                    ts_mod = (calc_anom(ts_mod.copy(), longterm=True) - calc_anom(ts_mod.copy(), longterm=False)).dropna()
                else:
                    ts_mod = ts_mod.dropna()
                ts_mod = ts_mod.resample('1d').mean()

                if run == 'open_loop':
                    res[f'r_tca_{run}_{mode}'] = r_mod

                tmp = pd.DataFrame({1: ts_ref, 2: ts_mod}).dropna()
                res[f'len_{run}_{mode}'] = len(tmp)
                r, p = pearsonr(tmp[1], tmp[2]) if len(tmp) > 10 else (np.nan, np.nan)
                res[f'r_{run}_{mode}'] = r
                res[f'r_corr_{run}_{mode}'] = min(r / r_asc, 1)

                tmp = pd.DataFrame({1: ts_ref, 2: ts_mod}).reindex(t_ana_u).dropna()
                res[f'ana_u_len_{run}_{mode}'] = len(tmp)
                r, p = pearsonr(tmp[1], tmp[2]) if len(tmp) > 10 else (np.nan, np.nan)
                res[f'ana_u_r_{run}_{mode}'] = r
                res[f'ana_u_r_corr_{run}_{mode}'] = min(r / r_asc, 1)

                tmp = pd.DataFrame({1: ts_ref, 2: ts_mod}).reindex(t_ana_i).dropna()
                res[f'ana_i_len_{run}_{mode}'] = len(tmp)
                r, p = pearsonr(tmp[1], tmp[2]) if len(tmp) > 10 else (np.nan, np.nan)
                res[f'ana_i_r_{run}_{mode}'] = r
                res[f'ana_i_r_corr_{run}_{mode}'] = min(r / r_asc, 1)

        if not result_file.exists():
            res.to_csv(result_file, float_format='%0.3f')
        else:
            res.to_csv(result_file, float_format='%0.3f', mode='a', header=False)


def run_ascat_eval_smos(n_procs=1):

    res_path = Path(f'~/Documents/work/MadKF/CLSM/SMOS40/validation/multiperiod/ascat').expanduser()
    if not res_path.exists():
        Path.mkdir(res_path, parents=True)

    part = np.arange(n_procs) + 1
    parts = repeat(n_procs, n_procs)

    p = Pool(n_procs)
    p.starmap(run_ascat_eval_smos_part, zip(part, parts))

    merge_files(res_path, pattern='ascat_eval_smos_part*.csv', fname='ascat_eval_smos.csv', delete=True)

def run_ascat_eval_smos_part(part, parts, ref='ascat'):

    periods = [['2010-04-01', '2020-04-01'],
               ['2010-04-01', '2015-04-01'],
               ['2015-04-01', '2020-04-01'],
               ['2010-04-01', '2012-10-01'],
               ['2012-10-01', '2015-04-01'],
               ['2015-04-01', '2017-10-01'],
               ['2017-10-01', '2020-04-01'],]

    res_path = Path(f'~/Documents/work/MadKF/CLSM/SMOS40/validation/multiperiod/ascat').expanduser()
    if not res_path.exists():
        Path.mkdir(res_path, parents=True)

    result_file = res_path / f'ascat_eval_smos_part{part}.csv'

    lut = pd.read_csv(Paths().lut, index_col=0)

    # Split grid cell list for parallelization
    subs = (np.arange(parts + 1) * len(lut) / parts).astype('int')
    subs[-1] = len(lut)
    start = subs[part - 1]
    end = subs[part]

    # Look-up table that contains the grid cells to iterate over
    lut = lut.iloc[start:end, :]

    names = ['open_loop'] + [f'SMOS40_it62{i}' for i in range(1,5)]
    runs = ['US_M36_SMOS40_TB_OL_noScl'] + [f'US_M36_SMOS40_TB_MadKF_DA_it62{i}' for i in range(1,5)]

    grid = LDAS_io('ObsFcstAna', runs[0]).grid
    dss_xhourly = [LDAS_io('xhourly', run).timeseries for run in runs]
    dss_obs_ana = [LDAS_io('ObsFcstAna', run).timeseries['obs_ana'] for run in runs]

    modes = ['absolute', 'longterm', 'shortterm']

    ascat = HSAF_io()

    for cnt, (gpi, data) in enumerate(lut.iterrows()):
        print('%i / %i' % (cnt, len(lut)))

        col = int(data.ease2_col - grid.tilegrids.loc['domain', 'i_offg'])
        row = int(data.ease2_row - grid.tilegrids.loc['domain', 'j_offg'])

        res = pd.DataFrame(index=(gpi,))
        res['col'] = int(data.ease2_col)
        res['row'] = int(data.ease2_row)
        res['lcol'] = col
        res['lrow'] = row

        try:
            ts_ascat = ascat.read(data['ascat_gpi'], resample_time=False).resample('1d').mean().dropna()
            ts_ascat = ts_ascat[~ts_ascat.index.duplicated(keep='first')]
            ts_ascat.name = 'ASCAT'
        except:
            continue


        dfs = [ds.sel(species=[1, 2]).isel(lat=row, lon=col).to_pandas().resample('1d').mean() for ds in dss_obs_ana]
        idx = [df[np.any(~np.isnan(df),axis=1)].index for df in dfs]


        t_ana = idx[0].intersection(idx[1]).intersection(idx[2]).intersection(idx[3])

        var = 'sm_surface'
        for mode in modes:

            if mode == 'absolute':
                ts_ref = ts_ascat.copy()
            else:
                ts_ref = calc_anom(ts_ascat.copy(), longterm=(mode == 'longterm')).dropna()

            for run, ts_model in zip(names, dss_xhourly):

                ind = (ts_model['snow_mass'][:, row, col].values == 0) & (
                        ts_model['soil_temp_layer1'][:, row, col].values > 277.15)
                ts_mod = ts_model[var][:, row, col].to_series().loc[ind]
                ts_mod.index += pd.to_timedelta('2 hours')

                if mode == 'absolute':
                    ts_mod = ts_mod.dropna()
                else:
                    ts_mod = calc_anom(ts_mod, longterm=mode == 'longterm').dropna()
                ts_mod = ts_mod.reindex(t_ana).dropna()

                for i, p in enumerate(periods):
                    tmp = pd.DataFrame({1: ts_ref, 2: ts_mod})[p[0]:p[1]].dropna()
                    res[f'p{i}_len_{run}_{mode}'] = len(tmp)
                    r, p = pearsonr(tmp[1], tmp[2]) if len(tmp) > 10 else (np.nan, np.nan)
                    res[f'p{i}_r_{run}_{mode}'] = r

        if not result_file.exists():
            res.to_csv(result_file, float_format='%0.3f')
        else:
            res.to_csv(result_file, float_format='%0.3f', mode='a', header=False)

def run_insitu_eval_smos(n_procs=1):

    res_path = Path(f'~/Documents/work/MadKF/CLSM/SMOS40/validation/multiperiod/insitu').expanduser()
    if not res_path.exists():
        Path.mkdir(res_path, parents=True)

    part = np.arange(n_procs) + 1
    parts = repeat(n_procs, n_procs)

    p = Pool(n_procs)
    p.starmap(run_insitu_eval_smos_part, zip(part, parts))

    merge_files(res_path, pattern='insitu_eval_smos_part*.csv', fname='insitu_eval_smos.csv', delete=True)


def run_insitu_eval_smos_part(part, parts, ref='ascat'):

    periods = [['2010-04-01', '2020-04-01'],
               ['2010-04-01', '2015-04-01'],
               ['2015-04-01', '2020-04-01'],
               ['2010-04-01', '2012-10-01'],
               ['2012-10-01', '2015-04-01'],
               ['2015-04-01', '2017-10-01'],
               ['2017-10-01', '2020-04-01'],]

    res_path = Path(f'~/Documents/work/MadKF/CLSM/SMOS40/validation/multiperiod/insitu').expanduser()
    if not res_path.exists():
        Path.mkdir(res_path, parents=True)

    result_file = res_path / f'insitu_eval_smos_part{part}.csv'

    ismn = ISMN_io()

    # Split grid cell list for parallelization
    subs = (np.arange(parts + 1) * len(ismn.list) / parts).astype('int')
    subs[-1] = len(ismn.list)
    start = subs[part - 1]
    end = subs[part]

    # Look-up table that contains the grid cells to iterate over
    ismn.list = ismn.list.iloc[start:end]

    names = [f'SMOS40_it62{i}' for i in range(1,5)]
    runs = [f'US_M36_SMOS40_TB_MadKF_DA_it62{i}' for i in range(1,5)]

    grid = LDAS_io('ObsFcstAna', runs[0]).grid
    dss_xhourly = [LDAS_io('xhourly', run).timeseries for run in runs]
    dss_obs_ana = [LDAS_io('ObsFcstAna', run).timeseries['obs_ana'] for run in runs]

    modes = ['absolute', 'longterm', 'shortterm']
    variables = ['sm_surface', 'sm_rootzone', 'sm_profile']

    # ismn.list = ismn.list.iloc[18::]

    for i, (meta, ts_insitu) in enumerate(ismn.iter_stations(surface_only=False)):
        print(f'{i} / {len(ismn.list)}')

        if ts_insitu is None:
            continue

        res = pd.DataFrame(meta.copy()).transpose()
        col = meta.ease_col
        row = meta.ease_row

        dfs = [ds.sel(species=[1, 2]).isel(lat=row, lon=col).to_pandas().resample('1d').mean() for ds in dss_obs_ana]
        idx = [df[np.any(~np.isnan(df),axis=1)].index for df in dfs]
        t_ana = idx[0].intersection(idx[1]).intersection(idx[2]).intersection(idx[3])

        for var in variables:
            for mode in modes:

                if mode == 'absolute':
                    ts_ref = ts_insitu[var].dropna()
                else:
                    ts_ref = calc_anom(ts_insitu[var], longterm=(mode=='longterm')).dropna()
                if len(ts_ref) > 0:
                    ts_ref = ts_ref.resample('1d').mean()
                else:
                    ts_ref.index = pd.DatetimeIndex(ts_ref)

                for run, ts_model in zip(names, dss_xhourly):

                    ind = (ts_model['snow_mass'][:, row, col].values == 0) & (
                            ts_model['soil_temp_layer1'][:, row, col].values > 277.15)
                    ts_mod = ts_model[var][:, row, col].to_series().loc[ind]
                    ts_mod.index += pd.to_timedelta('2 hours')

                    if mode == 'absolute':
                        ts_mod = ts_mod.dropna()
                    else:
                        ts_mod = calc_anom(ts_mod, longterm=(mode=='longterm')).dropna()
                    ts_mod = ts_mod.reindex(t_ana).dropna()

                    for j, p in enumerate(periods):
                        tmp = pd.DataFrame({1: ts_ref, 2: ts_mod})[p[0]:p[1]].dropna()
                        res[f'p{j}_len_{run}_{mode}_{var}'] = len(tmp)
                        r, p = pearsonr(tmp[1], tmp[2]) if len(tmp) > 10 else (np.nan, np.nan)
                        res[f'p{j}_r_{run}_{mode}_{var}'] = r

        if not result_file.exists():
            res.to_csv(result_file, float_format='%0.3f')
        else:
            res.to_csv(result_file, float_format='%0.3f', mode='a', header=False)


def plot_smos_eval():

    res = pd.read_csv(Path(f'~/Documents/work/MadKF/CLSM/SMOS40/validation/multiperiod/ascat_eval_smos.csv').expanduser(), index_col=0)
    fbase = Path(f'~/Documents/work/MadKF/CLSM/SMOS40/plots/multiperiod/per_period').expanduser()
    if not fbase.exists():
        Path.mkdir(fbase, parents=True)

    cbr_r = [0,0.8]
    # cbr_r = [-0.2,0.2]
    # cbr_len = [2e2, 2e3]
    cbr_len = [0, 250]

    n_subits = 4
    exps = ['SMOS40_it621', 'SMOS40_it622', 'SMOS40_it623', 'SMOS40_it624']
    modes = ['absolute', 'shortterm', 'longterm']

    # ref_exp = 'SMOS40_it622'

    params = ['r', 'len']
    cbranges = [cbr_r, cbr_len]

    # params = ['r',]
    # cbranges = [cbr_r,]

    fontsize = 12

    for p, cbr in zip(params, cbranges):

        cmap = 'seismic_r' if ('r' in p) else 'YlGn'

        for per in np.arange(7):

            f = plt.figure(figsize=(25,10))

            for i, m in enumerate(modes):
                for j, e in enumerate(exps):

                    plt.subplot(len(modes),len(exps), len(exps)*i + j + 1)

                    col = f'p{per}_{p}_{e}_{m}'

                    im = plot_ease_img(res, f'{col}' , title=f'{e} / {m}', cmap=cmap, cbrange=cbr, fontsize=fontsize, print_median=True, log_scale=False)

            plot_centered_cbar(f, im, len(exps), fontsize=fontsize)

            f.savefig(fbase / f'{p}_per{per}.png', dpi=300, bbox_inches='tight')
            plt.close()

        # plt.show()

def plot_smos_eval_insitu():

    res = pd.read_csv(Path(f'~/Documents/work/MadKF/CLSM/SMOS40/validation/multiperiod/insitu/insitu_eval_smos.csv').expanduser(), index_col=0)
    fbase = Path(f'~/Documents/work/MadKF/CLSM/SMOS40/plots/multiperiod/insitu/per_period').expanduser()
    if not fbase.exists():
        Path.mkdir(fbase, parents=True)

    cbr_r = [-0.2,1]
    cbr_len = [0, 1000]

    n_subits = 4
    exps = ['SMOS40_it621', 'SMOS40_it622', 'SMOS40_it623', 'SMOS40_it624']
    modes = ['absolute', 'shortterm', 'longterm']
    variables = ['sm_surface', 'sm_rootzone', 'sm_profile']

    params = ['r', 'len']
    cbranges = [cbr_r, cbr_len]

    fontsize = 12

    var = variables[0]

    for p, cbr in zip(params, cbranges):

        cmap = 'seismic_r' if ('r' in p) else 'YlGn'

        for per in np.arange(7):

            f = plt.figure(figsize=(25,10))

            for i, m in enumerate(modes):
                for j, e in enumerate(exps):

                    ax = plt.subplot(len(modes),len(exps), len(exps)*i + j + 1)

                    col = f'p{per}_{p}_{e}_{m}_{var}'

                    sns.histplot(res[col], bins=20, kde=False, ax=ax)
                    ax.set_xlim(cbr)
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    ax.set_title(col)

            f.savefig(fbase / f'{p}_per{per}.png', dpi=300, bbox_inches='tight')
            plt.close()

def plot_smos_eval_diffs():

    difference = True

    sub = 'skillgain' if difference else ''

    res = pd.read_csv(Path(f'~/Documents/work/MadKF/CLSM/SMOS40/validation/multiperiod/ascat/ascat_eval_smos.csv').expanduser(), index_col=0)
    fbase = Path(f'~/Documents/work/MadKF/CLSM/SMOS40/plots/multiperiod/ascat').expanduser() / sub

    n_subits = 4
    exps = ['SMOS40_it621', 'SMOS40_it622', 'SMOS40_it623', 'SMOS40_it624']
    modes = ['absolute', 'shortterm', 'longterm']

    per = ['2010-2020',
           '2010-2015',
           '2015-2020',
           '2010-2012',
           '2012-2015',
           '2015-2017',
           '2017-2020']

    p = 'r'
    cbr = [-0.2,0.2]

    fontsize = 10

    cmap = 'seismic_r' if ('r' in p) else 'YlGn'

    combs = list(combinations(np.arange(7), 2))
    combs = [combs[i] for i in [0, 1, 6, 7, 8, 13, 14, 15, 18, 20]]

    for j, e in enumerate(exps):

        froot = fbase / e
        if not froot.exists():
            Path.mkdir(froot, parents=True)

        for i, m in enumerate(modes):

            f = plt.figure(figsize=(26,6))

            for k, comb in enumerate(combs):

                plt.subplot(2, 5, k+1)

                if difference:
                    res[f'diff_{k}'] = (res[f'p{comb[0]}_{p}_{e}_{m}'] - res[f'p{comb[0]}_{p}_open_loop_{m}']) \
                                       - (res[f'p{comb[1]}_{p}_{e}_{m}'] - res[f'p{comb[1]}_{p}_open_loop_{m}'])
                else:
                    res[f'diff_{k}'] = res[f'p{comb[0]}_{p}_{e}_{m}'] - res[f'p{comb[1]}_{p}_{e}_{m}']

                title = f'{per[comb[0]]} minus {per[comb[1]]}'

                im = plot_ease_img(res, f'diff_{k}' , title=title, cmap=cmap, cbrange=cbr, fontsize=fontsize, print_median=True)

            plot_centered_cbar(f, im, 5, fontsize=fontsize, col_offs=0)

            f.savefig(froot / f'{m}.png', dpi=300, bbox_inches='tight')
            plt.close()

def plot_smos_eval_diffs_insitu():

    res = pd.read_csv(Path(f'~/Documents/work/MadKF/CLSM/SMOS40/validation/multiperiod/insitu/insitu_eval_smos.csv').expanduser(), index_col=0)
    fbase = Path(f'~/Documents/work/MadKF/CLSM/SMOS40/plots/multiperiod/insitu').expanduser()

    n_subits = 4
    exps = ['SMOS40_it621', 'SMOS40_it622', 'SMOS40_it623', 'SMOS40_it624']
    modes = ['absolute', 'shortterm', 'longterm']
    variables = ['sm_surface', 'sm_rootzone', 'sm_profile']

    per = ['2010-2020',
           '2010-2015',
           '2015-2020',
           '2010-2012',
           '2012-2015',
           '2015-2017',
           '2017-2020']

    p = 'r'
    cbr = [-0.6,0.6]

    fontsize = 10

    cmap = 'seismic_r' if ('r' in p) else 'YlGn'

    combs = list(combinations(np.arange(7), 2))
    combs = [combs[i] for i in [0, 1, 6, 7, 8, 13, 14, 15, 18, 20]]

    for v in variables:
        for e in exps:

            froot = fbase / v / e
            if not froot.exists():
                Path.mkdir(froot, parents=True)

            for m in modes:

                f = plt.figure(figsize=(24,8))

                for k, comb in enumerate(combs):

                    ax = plt.subplot(2, 5, k+1)

                    res[f'diff_{k}'] = res[f'p{comb[0]}_{p}_{e}_{m}_{v}'] - res[f'p{comb[1]}_{p}_{e}_{m}_{v}']
                    title = f'{per[comb[0]]} minus {per[comb[1]]}'

                    sns.histplot(res[f'diff_{k}'], bins=20, kde=False, ax=ax)
                    ax.set_xlim(cbr)
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    ax.set_title(title, fontsize=fontsize)


                f.savefig(froot / f'{m}.png', dpi=300, bbox_inches='tight')
                plt.close()


def scatterplot_abs_vs_anom():

    res = pd.read_csv(Path(f'~/Documents/work/MadKF/CLSM/SM_err_ratio/validation/ascat_eval.csv').expanduser(), index_col=0)
    fbase = Path(f'~/Documents/work/MadKF/CLSM/SM_err_ratio/plots/st_vs_lt_gain').expanduser()
    if not fbase.exists():
        Path.mkdir(fbase, parents=True)

    ref='open_loop'
    # ref='DA_4K_obserr'

    fontsize = 12
    cmap = 'seismic_r'

    if ref=='open_loop':
        xlim = [-0.45,0.15]
        ylim = [-0.3,0.4]
        vmax = 50
    else:
        xlim = [-0.15,0.15]
        ylim = [-0.15,0.15]
        vmax = 90
    exps = ['abs_1', 'anom_lt_1', 'anom_st_3']
    titles = ['P/R absolute', ' P/R long-term', ' P/R short-term ']

    f = plt.figure(figsize=(17, 6))

    for i, (exp, tit) in enumerate(zip(exps,titles)):

        xcol = f'ana_u_r_{exp}_shortterm'
        xrefcol = f'ana_u_r_{ref}_shortterm'

        ycol = f'ana_u_r_{exp}_longterm'
        yrefcol = f'ana_u_r_{ref}_longterm'

        ax = plt.subplot(1, 3, i+1)
        ax.hexbin(res[xcol] - res[xrefcol], res[ycol] - res[yrefcol],
                  gridsize=45, extent=xlim+ylim, vmin=1, vmax=vmax, bins='log',
                  cmap='jet', mincnt=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.axhline(linewidth=1,linestyle='--',color='k')
        plt.axvline(linewidth=1,linestyle='--',color='k')
        # plt.plot([-1,1],[-1,1],'k--.', linewidth=1)
        ax.set_title(tit)
        if i==1:
            ax.set_xlabel('$\Delta$R (short-term)')
        if i==0:
            ax.set_ylabel('$\Delta$R (long-term)')

    f.savefig(fbase / f'ref_{ref}.png', dpi=300, bbox_inches='tight')
    plt.close()

def scatterplot_gain_vs_skill():

    res = pd.read_csv(Path(f'~/Documents/work/MadKF/CLSM/SM_err_ratio/validation/ascat_eval.csv').expanduser(), index_col=0)
    fbase = Path(f'~/Documents/work/MadKF/CLSM/SM_err_ratio/plots/gain_vs_skill').expanduser()
    if not fbase.exists():
        Path.mkdir(fbase, parents=True)

    sensors = ['ASCAT', 'SMAP', 'CLSM']
    tc_res = pd.read_csv('/Users/u0116961/Documents/work/validation_good_practice/CI80/ASCAT_SMAP_CLSM/result.csv', index_col=0)
    for sens in sensors:
        res[f'r2_{sens}_absolute'] = tc_res.reindex(res.index)[f'r2_grid_abs_m_{sens}_tc_ASCAT_SMAP_CLSM']
        res[f'r2_{sens}_longterm'] = tc_res.reindex(res.index)[f'r2_grid_anom_lt_m_{sens}_tc_ASCAT_SMAP_CLSM']
        res[f'r2_{sens}_shortterm'] = tc_res.reindex(res.index)[f'r2_grid_anom_st_m_{sens}_tc_ASCAT_SMAP_CLSM']

    fontsize = 12
    cmap = 'seismic_r'

    xlim = [0,1]
    ylim = [-0.3,0.4]
    vmax = 50

    ref = 'open_loop'
    modes = ['absolute', 'longterm', 'shortterm']
    exps = ['abs_1', 'anom_lt_1', 'anom_st_3']
    titles = ['P/R absolute', ' P/R longterm', ' P/R shortterm']

    for sens in sensors:

        f = plt.figure(figsize=(16, 12))
        for i, mode in enumerate(modes):
            for j, (exp, tit) in enumerate(zip(exps,titles)):

                xcol = f'r2_{sens}_{mode}'
                # xcol = f'r2_{sens}_shortterm'

                ycol = f'ana_u_r_{exp}_{mode}'
                yrefcol = f'ana_u_r_{ref}_{mode}'

                ax = plt.subplot(3, 3, i*3 + j + 1)
                ax.hexbin(res[xcol], res[ycol] - res[yrefcol],
                          gridsize=45, extent=xlim+ylim, vmin=1, vmax=vmax, bins='log',
                          cmap='jet', mincnt=1)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                plt.axhline(linewidth=1,linestyle='--',color='k')
                plt.axvline(linewidth=1,linestyle='--',color='k')
                # plt.plot([-1,1],[-1,1],'k--.', linewidth=1)
                if i==0:
                    ax.set_title(tit)
                if (i==2) & (j==1):
                    ax.set_xlabel(f'R$^2$ {sens}')
                if j==0:
                    ax.set_ylabel(f'$\Delta$R ({mode})')

        # plt.tight_layout()
        # plt.show()

        f.savefig(fbase / f'{sens}.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_eval(iteration):

    analysis_only = True
    relative = True
    # ref_exp = 'open_loop'
    ref_exp = 'OL_no_pcorr'

    scaling = 'seas'

    sub = 'ana_' if analysis_only else ''
    fext = '_rel' if relative else '_abs'
    res = pd.read_csv(Path(f'~/Documents/work/MadKF/CLSM/SM_err_ratio/validation/ascat_eval.csv').expanduser(), index_col=0)
    fbase = Path(f'~/Documents/work/MadKF/CLSM/SM_err_ratio/plots/ascat_eval/{scaling}/{ref_exp}').expanduser()
    if not fbase.exists():
        Path.mkdir(fbase, parents=True)

    # tc_res = pd.read_csv('/Users/u0116961/Documents/work/validation_good_practice/CI80/ASCAT_SMAP_CLSM/result.csv', index_col=0)
    # res['r2_ascat_absolute'] = tc_res.reindex(res.index)['r2_grid_abs_m_ASCAT_tc_ASCAT_SMAP_CLSM']
    # res['r2_ascat_longterm'] = tc_res.reindex(res.index)['r2_grid_anom_lt_m_ASCAT_tc_ASCAT_SMAP_CLSM']
    # res['r2_ascat_shortterm'] = tc_res.reindex(res.index)['r2_grid_anom_st_m_ASCAT_tc_ASCAT_SMAP_CLSM']
    # topo = pd.read_csv('/Users/u0116961/data_sets/ASCAT/static_layer/topographic_complexity_conus.csv', index_col=0, squeeze=True, header=None)

    cbr_r = [0.6, 1.4] if relative else [0.2,0.7]
    cbr_r = [-0.3, 0.3] if relative else [0.2,0.7]
    cbr_len = [2e2, 2e3]

    rmsd_root = f'US_M36_SMAP_TB_DA_SM_ERR_scl_{scaling}_'
    rmsd_exps = list(np.sort([x.name.split(rmsd_root)[1] for x in Path('/Users/u0116961/data_sets/LDASsa_runs').glob(f'*SM_ERR_scl_{scaling}*')]))
    exps = ['open_loop', 'OL_no_pcorr', 'DA_4K_obserr_seas'] + \
            rmsd_exps
    # exps = ['open_loop', 'DA_4K_obserr'] + \
    #         rmsd_exps

    # rmsd_root = 'US_M36_SMAP_TB_DA_scl_'
    # rmsd_exps = [x.name.split(rmsd_root)[1] for x in Path('/Users/u0116961/data_sets/LDASsa_runs').glob('*RMSD*')]
    # exps = ['open_loop', 'DA_4K_obserr'] + \
    #         [f'SMAP_it{i}{j}' for i in range(1, 5) for j in range(1, 4)] + \
    #         rmsd_exps
    #
    # ind = [0,1,2,3,4,18,14,5,6,7,15,16,8,9,10,17,19,11,12,13]
    # exps = np.array(exps)[ind]

    modes = ['absolute', 'shortterm', 'longterm']

    # params = ['ana_len']
    # cbranges = [cbr_len, cbr_len, cbr_r]
    #
    params = [f'u_r']
    cbranges = [cbr_r]

    fontsize = 12

    for p, cbr in zip(params, cbranges):

        cmap = 'seismic_r' if ('r' in p) & relative else 'YlGn'

        for i, m in enumerate(modes):

            f = plt.figure(figsize=(25,17))

            cnt = 0
            for e in exps:
                cnt += 1
                if cnt==6:
                    cnt += 1

                # plt.subplot(4, 5, j+1)
                plt.subplot(4, 3, cnt)
                # plt.subplot(len(modes),len(exps), len(exps)*i + j + 1)

                scl = '' if cnt < 4 else f'scl_{scaling}_'

                col = f'{sub}{p}_{scl}{e}_{m}'
                if relative and (p != 'len'):
                    ref_col = f'{sub}{p}_{ref_exp}_{m}'
                    res[f'{col}_diff'] = (res[col] - res[ref_col])

                    # res.loc[topo.reindex(res.index) > 15, f'{col}_diff'] = np.nan

                    # res[f'{col}_diff'] = (res[col] - res[ref_col]) / (1 - res[ref_col])

                    ext = '_diff'
                else:
                    ext = ''

                log_scale = True if 'len' in p else False
                im = plot_ease_img(res, f'{col}{ext}' , title=f'{e}', cmap=cmap, cbrange=cbr, fontsize=fontsize, print_median=True, log_scale=log_scale)
                # im = plot_ease_img(res, f'{col}{ext}' , title=f'{e} / {m}', cmap=cmap, cbrange=cbr, fontsize=fontsize, print_median=True, log_scale=log_scale)
                # if (i == 2) & (j == 1):
                #     ax = im.axes

            plot_centered_cbar(f, im, 5, fontsize=fontsize)

            f.savefig(fbase / f'ascat_eval_{m}_{sub}{p}{fext}.png', dpi=300, bbox_inches='tight')
            plt.close()

        # plt.show()



def plot_centered_cbar(f, im, n_cols, wspace=0.04, hspace=0.025, bottom=0.06, fontsize=12, col_offs=0):

    f.subplots_adjust(wspace=wspace, hspace=hspace, bottom=bottom)

    ctr = n_cols/2 * -1
    if ctr % 1 == 0:
        pos1 = f.axes[int(ctr) - 1 - col_offs].get_position()
        pos2 = f.axes[int(ctr) - col_offs].get_position()
        x1 = (pos1.x0 + pos1.x1) / 2
        x2 = (pos2.x0 + pos2.x1) / 2
    else:
        pos = f.axes[floor(ctr)].get_position()
        x1 = pos.x0
        x2 = pos.x1

    cbar_ax = f.add_axes([x1, 0.03, x2 - x1, 0.03])
    cbar = f.colorbar(im, orientation='horizontal', cax=cbar_ax)
    for t in cbar.ax.get_xticklabels():
        t.set_fontsize(fontsize)


def plot_ascat_smap_eval(iteration):

    analysis_only = True
    relative = True
    ref_exp = 'open_loop'

    sub = 'ana_' if analysis_only else ''
    fext = '_rel' if relative else '_abs'

    res_ascat = pd.read_csv(f'/Users/u0116961/Documents/work/MadKF/CLSM/SMAP/validation/iter_{iteration}/ascat_eval.csv', index_col=0)
    res_smap = pd.read_csv(f'/Users/u0116961/Documents/work/MadKF/CLSM/SMAP/validation/iter_{iteration}/smap_eval.csv', index_col=0)
    fbase = Path(f'/Users/u0116961/Documents/work/MadKF/CLSM/SMAP/plots/iter_{iteration}')
    if not fbase.exists():
        Path.mkdir(fbase, parents=True)

    cbr_r = [0.6, 1.4] if relative else [0.2,0.7]
    cbr_r = [-0.4, 0.4] if relative else [0.2,0.7]
    cbr_len = [500, 1500]


    exps = ['SMOSSMAP_short', 'SMAP_it11']
    modes = ['absolute', 'longterm', 'shortterm']

    # params = ['len', 'r']
    # cbranges = [cbr_len, cbr_r]

    params = ['r']
    cbranges = [cbr_r]

    fontsize = 12

    for p, cbr in zip(params, cbranges):

        cmap = 'seismic_r' if (p == 'r') & relative else 'YlGn'

        f = plt.figure(figsize=(22,8))

        for j, e in enumerate(exps):
            for i, m in enumerate(modes):

                plt.subplot(len(exps),len(modes), len(modes)*j + i + 1)

                col = f'{sub}{p}_{e}_{m}'
                if relative and (p != 'len'):
                    ref_col = f'{sub}{p}_{ref_exp}_{m}'
                    res_ascat[f'{col}_diff'] = (res_ascat[col] - res_ascat[ref_col])  / (1 - res_ascat[ref_col])
                    res_smap[f'{col}_diff'] = (res_smap[col] - res_smap[ref_col])  / (1 - res_smap[ref_col])
                    ext = '_diff'
                else:
                    ext = ''

                if j == 0:
                    im = plot_ease_img(res_ascat, f'{col}{ext}' , title=f'Skill gain {m} (ref: ASCAT)', cmap=cmap, cbrange=cbr, fontsize=fontsize, print_median=True)
                else:
                    im = plot_ease_img(res_smap, f'{col}{ext}' , title=f'Skill gain {m} (ref: SMAP)', cmap=cmap, cbrange=cbr, fontsize=fontsize, print_median=True)

        f.subplots_adjust(wspace=0.04, hspace=0.025, bottom=0.06)
        # pos1 = f.axes[-3].get_position()
        # pos2 = f.axes[-4].get_position()
        # x1 = (pos1.x0 + pos1.x1) / 2
        # x2 = (pos2.x0 + pos2.x1) / 2
        pos = f.axes[-2].get_position()
        x1 = pos.x0
        x2 = pos.x1
        cbar_ax = f.add_axes([x1, 0.03, x2-x1, 0.03])
        cbar = f.colorbar(im, orientation='horizontal', cax=cbar_ax)
        for t in cbar.ax.get_xticklabels():
            t.set_fontsize(fontsize)

        f.savefig(fbase / f'ascat_smap_eval.png', dpi=300, bbox_inches='tight')
        plt.close()

        # plt.show()


if __name__=='__main__':

    iteration = 4

    # run_ascat_eval_smos(n_procs=14)
    # run_ascat_eval_smos_part(2, 2)

    # run_insitu_eval_smos(n_procs=14)
    # run_insitu_eval_smos_part(3, 14)

    plot_eval(iteration)
    # scatterplot_abs_vs_anom()
    # scatterplot_gain_vs_skill()

    # plot_smos_eval()
    # plot_smos_eval_insitu()

    # plot_smos_eval_diffs()
    # plot_smos_eval_diffs_insitu()

    # plot_ascat_smap_eval(iteration)

    # run_ascat_eval_part(iteration, 1, 1)

'''
from myprojects.experiments.MadKF.CLSM.validate_madkf_smap import run_ascat_eval, plot_eval
iteration = 4
run_ascat_eval(iteration, n_procs=14)

# plot_eval(iteration)

'''
