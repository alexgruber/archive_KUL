
import platform

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
from itertools import repeat
from multiprocessing import Pool

from scipy.stats import pearsonr

import seaborn as sns
import matplotlib.pyplot as plt

from pyldas.interface import LDAS_io

from myprojects.readers.insitu import ISMN_io
from myprojects.readers.ascat import HSAF_io

from myprojects.timeseries import calc_anom
from myprojects.functions import merge_files

from validation_good_practice.ancillary.paths import Paths
from validation_good_practice.plots import plot_ease_img

def run_ascat_eval(iteration, n_procs=1):

    it = repeat(iteration, n_procs)
    part = np.arange(n_procs) + 1
    parts = repeat(n_procs, n_procs)

    p = Pool(n_procs)
    p.starmap(run_ascat_eval_part, zip(it, part, parts))

    res_path = Path(f'~/Documents/work/MadKF/CLSM/SMAP/validation/iter_{iteration}').expanduser()
    merge_files(res_path, pattern='ascat_eval_part*.csv', fname='ascat_eval.csv', delete=True)

def run_ascat_eval_part(iteration, part, parts, ref='ascat'):

    if platform.system() == 'Linux':
        stg = '/staging/leuven/stg_00024/OUTPUT/alexg'
        smap_path = Path('/staging/leuven/stg_00024/OUTPUT/alexg/data_sets/SMAP/timeseries')
    else:
        stg = None
        smap_path = Path('/Users/u0116961/data_sets/SMAP/timeseries')

    res_path = Path(f'~/Documents/work/MadKF/CLSM/SMAP/validation/iter_{iteration}').expanduser()
    if not res_path.exists():
        Path.mkdir(res_path, parents=True)

    result_file = res_path / ('ascat_eval_part%i.csv' % part)

    lut = pd.read_csv(Paths().lut, index_col=0)

    # Split grid cell list for parallelization
    subs = (np.arange(parts + 1) * len(lut) / parts).astype('int')
    subs[-1] = len(lut)
    start = subs[part - 1]
    end = subs[part]

    # Look-up table that contains the grid cells to iterate over
    lut = lut.iloc[start:end, :]

    names = ['open_loop', 'SMOSSMAP_short', 'SMOS40_it631'] + [f'SMAP_it{iteration}{i}' for i in range(1,4)]
    runs = ['US_M36_SMAP_TB_OL_noScl', 'US_M36_SMAP_TB_DA_scl_SMOSSMAP_short', 'US_M36_SMOS40_TB_MadKF_DA_it613'] + [f'US_M36_SMAP_TB_MadKF_DA_it{iteration}{i}' for i in range(1,4)]
    roots = [stg, stg, stg, None, None, None]

    dss = [LDAS_io('xhourly', run, root=root).timeseries for run, root in zip(runs, roots)]
    grid = LDAS_io('xhourly', runs[0], root=stg).grid

    # t_ana = pd.DatetimeIndex(LDAS_io('ObsFcstAna', runs[0]).timeseries.time.values).sort_values()
    ds_obs_smap = (LDAS_io('ObsFcstAna', 'US_M36_SMAP_TB_OL_noScl', root=stg).timeseries['obs_ana'])
    ds_obs_smos = (LDAS_io('ObsFcstAna', 'US_M36_SMOS40_TB_MadKF_DA_it613', root=stg).timeseries['obs_ana'])

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

        if ref == 'ascat':
            try:
                ts_ascat = ascat.read(data['ascat_gpi'], resample_time=False).resample('1d').mean().dropna()
                ts_ascat = ts_ascat[~ts_ascat.index.duplicated(keep='first')]
                ts_ascat.name = 'ASCAT'
            except:
                continue
        else:
            try:
                ts_smap = pd.read_csv(smap_path / f'{gpi}.csv',
                                      index_col=0, parse_dates=True, names=('smap',))['smap'].resample('1d').mean().dropna()
            except:
                continue

        t_df_smap = ds_obs_smap.sel(species=[1, 2]).isel(lat=row, lon=col).to_pandas()
        t_df_smos = ds_obs_smos.sel(species=[1, 2]).isel(lat=row, lon=col).to_pandas()
        t_ana_smap = t_df_smap[~np.isnan(t_df_smap[1]) | ~np.isnan(t_df_smap[2])].resample('1d').mean().index
        t_ana_smos = t_df_smos[~np.isnan(t_df_smos[1]) | ~np.isnan(t_df_smos[2])].resample('1d').mean().index

        var = 'sm_surface'
        for mode in modes:

            if ref == 'ascat':
                if mode == 'absolute':
                    ts_ref = ts_ascat.copy()
                else:
                    ts_ref = calc_anom(ts_ascat.copy(), longterm=(mode == 'longterm')).dropna()
            else:
                if mode == 'absolute':
                    ts_ref = ts_smap.copy()
                else:
                    ts_ref = calc_anom(ts_smap.copy(), longterm=(mode == 'longterm')).dropna()

            for run, ts_model in zip(names, dss):

                t_ana = t_ana_smos if run == 'SMOS40_it631' else t_ana_smap

                ind = (ts_model['snow_mass'][:, row, col].values == 0) & (
                        ts_model['soil_temp_layer1'][:, row, col].values > 277.15)
                ts_mod = ts_model[var][:, row, col].to_series().loc[ind]
                ts_mod.index += pd.to_timedelta('2 hours')

                if mode == 'absolute':
                    ts_mod = ts_mod.dropna()
                else:
                    ts_mod = calc_anom(ts_mod, longterm=mode == 'longterm').dropna()
                ts_mod = ts_mod.resample('1d').mean()

                tmp = pd.DataFrame({1: ts_ref, 2: ts_mod}).dropna()
                res['len_' + run + '_' + mode] = len(tmp)
                r, p = pearsonr(tmp[1], tmp[2]) if len(tmp) > 10 else (np.nan, np.nan)
                res['r_' + run + '_' + mode] = r

                tmp = pd.DataFrame({1: ts_ref, 2: ts_mod}).reindex(t_ana).dropna()
                res['ana_len_' + run + '_' + mode] = len(tmp)
                r, p = pearsonr(tmp[1], tmp[2]) if len(tmp) > 10 else (np.nan, np.nan)
                res['ana_r_' + run + '_' + mode] = r

        if not result_file.exists():
            res.to_csv(result_file, float_format='%0.3f')
        else:
            res.to_csv(result_file, float_format='%0.3f', mode='a', header=False)

def plot_eval(iteration):

    analysis_only = True
    relative = True
    ref_exp = 'SMOS40_it631'

    sub = 'ana_' if analysis_only else ''
    fext = '_rel' if relative else '_abs'

    res = pd.read_csv(Path(f'~/Documents/work/MadKF/CLSM/SMAP/validation/iter_{iteration}/ascat_eval.csv').expanduser(), index_col=0)
    fbase = Path(f'~/Documents/work/MadKF/CLSM/SMAP/plots/iter_{iteration}').expanduser()
    if not fbase.exists():
        Path.mkdir(fbase, parents=True)

    cbr_r = [0.6, 1.4] if relative else [0.2,0.7]
    cbr_r = [-0.3, 0.3] if relative else [0.2,0.7]
    cbr_len = [500, 1500]

    exps = ['SMOS40_it631', 'SMOSSMAP_short'] + [f'SMAP_it{iteration}{i}' for i in range(1, 4)]
    modes = ['absolute', 'longterm', 'shortterm']

    # params = ['len', 'r']
    # cbranges = [cbr_len, cbr_r]

    params = ['r']
    cbranges = [cbr_r]

    fontsize = 12

    for p, cbr in zip(params, cbranges):

        cmap = 'seismic_r' if (p == 'r') & relative else 'YlGn'

        f = plt.figure(figsize=(25,8))

        for i, m in enumerate(modes):
            for j, e in enumerate(exps):

                plt.subplot(len(modes),len(exps), len(exps)*i + j + 1)

                col = f'{sub}{p}_{e}_{m}'
                if relative and (p != 'len'):
                    ref_col = f'{sub}{p}_{ref_exp}_{m}'
                    res[f'{col}_diff'] = (res[col] - res[ref_col])  / (1 - res[ref_col])
                    ext = '_diff'
                else:
                    ext = ''

                im = plot_ease_img(res, f'{col}{ext}' , title=f'{e} / {m}', cmap=cmap, cbrange=cbr, fontsize=fontsize, print_median=True)
                # if (i == 2) & (j == 1):
                #     ax = im.axes

        f.subplots_adjust(wspace=0.04, hspace=0.025, bottom=0.06)
        # pos1 = f.axes[-3].get_position()
        # pos2 = f.axes[-4].get_position()
        # x1 = (pos1.x0 + pos1.x1) / 2
        # x2 = (pos2.x0 + pos2.x1) / 2
        pos = f.axes[-3].get_position()
        x1 = pos.x0
        x2 = pos.x1
        cbar_ax = f.add_axes([x1, 0.03, x2-x1, 0.03])
        cbar = f.colorbar(im, orientation='horizontal', cax=cbar_ax)
        for t in cbar.ax.get_xticklabels():
            t.set_fontsize(fontsize)

        f.savefig(fbase / f'ascat_eval_{sub}{p}{fext}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # plt.show()

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

    iteration = 2

    run_ascat_eval(iteration, n_procs=9)
    # run_ascat_eval_part(iteration, 2, 2)

    plot_eval(iteration)
    # plot_ascat_smap_eval(iteration)

'''
from myprojects.experiments.MadKF.CLSM.validate_madkf_smap import run_ascat_eval, plot_eval
iteration = 2
run_ascat_eval(iteration, n_procs=9)
plot_eval(iteration)

'''
