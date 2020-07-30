
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

from scipy.stats import pearsonr

from itertools import repeat
from multiprocessing import Pool

# from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import seaborn as sns

from pyldas.interface import LDAS_io
from pyldas.templates import template_error_Tb40

from myprojects.readers.insitu import ISMN_io
from myprojects.readers.ascat import HSAF_io

from pytesmo.temporal_matching import matching, df_match

from myprojects.timeseries import calc_anom
from myprojects.functions import merge_files

from validation_good_practice.data_readers.interface import reader
from validation_good_practice.ancillary.paths import Paths
from validation_good_practice.ancillary.grid import EASE2
from validation_good_practice.plots import plot_ease_img

def colrow2easegpi(col, row, glob=False):

    # convert local to global indexing
    grid = LDAS_io().grid
    if not glob:
        col += grid.tilegrids.loc['domain', 'i_offg']
        row += grid.tilegrids.loc['domain', 'j_offg']

    grid = EASE2()
    lons, lats = np.meshgrid(grid.ease_lons, grid.ease_lats)
    cols, rows = np.meshgrid(np.arange(len(grid.ease_lons)), np.arange(len(grid.ease_lats)))
    lut = pd.Series(np.arange(cols.size), index=([cols.flatten(), rows.flatten()]))

    return lut.reindex(zip(col, row)).values


def run_ismn_eval():

    experiments = [['SMOSSMAP', 'short']]

    names = ['open_loop'] + ['MadKF_SMOS40'] + ['_'.join(exp) for exp in experiments]
    runs = ['US_M36_SMAP_TB_OL_noScl'] + ['US_M36_SMOS40_TB_MadKF_DA_it613'] + [f'US_M36_SMAP_TB_DA_scl_{name}' for name in names[2::]]

    dss = [LDAS_io('xhourly', run).timeseries for run in runs]

    result_file = Path('/Users/u0116961/Documents/work/LDAS/2020-03_scaling/validation/ismn_eval.csv')
    t_ana = pd.DatetimeIndex(LDAS_io('ObsFcstAna', runs[0]).timeseries.time.values).sort_values()

    variables = ['sm_surface','sm_rootzone','sm_profile']
    modes = ['absolute','longterm','shortterm']

    ismn = ISMN_io()
    ismn.list = ismn.list.iloc[70::]

    i = 0
    for meta, ts_insitu in ismn.iter_stations(surface_only=False):
        i += 1
        logging.info('%i/%i' % (i, len(ismn.list)))

        if len(ts_insitu := ts_insitu['2015-04-01':'2020-04-01']) < 50:
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

                for run, ts_model in zip(names, dss):

                    ind = (ts_model['snow_mass'][:, row, col].values == 0)&(ts_model['soil_temp_layer1'][:, row, col].values > 277.15)
                    ts_mod = ts_model[var][:, row, col].to_series().loc[ind]
                    ts_mod.index += pd.to_timedelta('2 hours')

                    if mode == 'absolute':
                        ts_mod = ts_mod.dropna()
                    else:
                        ts_mod = calc_anom(ts_mod, longterm=mode=='longterm').dropna()

                    tmp = pd.DataFrame({1: ts_ref, 2: ts_mod}).dropna()
                    res['len_' + mode + '_' + var] = len(tmp)
                    r, p = pearsonr(tmp[1],tmp[2]) if len(tmp) > 10 else (np.nan, np.nan)
                    res['r_' + run +'_' + mode + '_' + var] = r
                    # res['p_' + run +'_' + mode + '_' + var] = p
                    # res['rmsd_' + run +'_' + mode + '_' + var] = np.sqrt(((tmp[1]-tmp[2])**2).mean())
                    res['ubrmsd_' + run +'_' + mode + '_' + var] = np.sqrt((((tmp[1]-tmp[1].mean())-(tmp[2]-tmp[2].mean()))**2).mean())

                    tmp = pd.DataFrame({1: ts_ref, 2: ts_mod}).reindex(t_ana).dropna()
                    res['ana_len_' + mode + '_' + var] = len(tmp)
                    r, p = pearsonr(tmp[1], tmp[2]) if len(tmp) > 10 else (np.nan, np.nan)
                    res['ana_r_' + run + '_' + mode + '_' + var] = r
                    # res['ana_p_' + run + '_' + mode + '_' + var] = p
                    # res['ana_rmsd_' + run +'_' + mode + '_' + var] = np.sqrt(((tmp[1]-tmp[2])**2).mean())
                    res['ana_ubrmsd_' + run +'_' + mode + '_' + var] = np.sqrt((((tmp[1]-tmp[1].mean())-(tmp[2]-tmp[2].mean()))**2).mean())


        if not result_file.exists():
            res.to_csv(result_file, float_format='%0.4f')
        else:
            res.to_csv(result_file, float_format='%0.4f', mode='a', header=False)

def run_ascat_eval():

        parts = 14

        # Parallelized processing
        p = Pool(parts)

        part = np.arange(parts) + 1
        parts = repeat(parts, parts)

        p.starmap(run_ascat_eval_part, zip(part, parts))

        res_path = '/Users/u0116961/Documents/work/LDAS/2020-03_scaling/validation'
        merge_files(res_path, pattern='ascat_eval_part*.csv', fname='ascat_eval.csv', delete=True)

def run_ascat_eval_part(part, parts):

    res_path = Path('/Users/u0116961/Documents/work/LDAS/2020-03_scaling/validation')
    result_file = res_path / ('ascat_eval_part%i.csv' % part)

    lut = pd.read_csv(Paths().lut, index_col=0)

    # Split grid cell list for parallelization
    subs = (np.arange(parts + 1) * len(lut) / parts).astype('int')
    subs[-1] = len(lut)
    start = subs[part - 1]
    end = subs[part]

    # Look-up table that contains the grid cells to iterate over
    lut = lut.iloc[start:end, :]

    names = ['open_loop', 'SMOSSMAP_short', 'MadKF_SMOS40']
    runs = ['US_M36_SMAP_TB_OL_noScl', 'US_M36_SMAP_TB_DA_scl_SMOSSMAP_short', 'US_M36_SMOS40_TB_MadKF_DA_it613']

    dss = [LDAS_io('xhourly', run).timeseries for run in runs]
    grid = LDAS_io().grid

    # t_ana = pd.DatetimeIndex(LDAS_io('ObsFcstAna', runs[0]).timeseries.time.values).sort_values()
    ds_obs_smap = (LDAS_io('ObsFcstAna', 'US_M36_SMAP_TB_OL_noScl').timeseries['obs_ana'])
    ds_obs_smos = (LDAS_io('ObsFcstAna', 'US_M36_SMOS40_TB_MadKF_DA_it613').timeseries['obs_ana'])

    modes = ['absolute','longterm','shortterm']

    ascat = HSAF_io()

    for cnt, (gpi, data) in enumerate(lut.iterrows()):
        print('%i / %i' % (cnt, len(lut)))

        col = int(data.ease2_col - grid.tilegrids.loc['domain', 'i_offg'])
        row = int(data.ease2_row - grid.tilegrids.loc['domain', 'j_offg'])

        res = pd.DataFrame(index=(gpi,))
        res['col'] =int(data.ease2_col)
        res['row'] = int(data.ease2_row)
        res['lcol'] = col
        res['lrow'] = row

        try:
            ts_ascat = ascat.read(data['ascat_gpi'], resample_time=False).resample('1d').mean().dropna()
            ts_ascat = ts_ascat[~ts_ascat.index.duplicated(keep='first')]
            ts_ascat.name = 'ASCAT'
        except:
            continue

        t_df_smap = ds_obs_smap.sel(species=[1, 2]).isel(lat=row, lon=col).to_pandas()
        t_df_smos = ds_obs_smos.sel(species=[1, 2]).isel(lat=row, lon=col).to_pandas()
        t_ana_smap = t_df_smap[~np.isnan(t_df_smap[1]) | ~np.isnan(t_df_smap[2])].resample('1d').mean().index
        t_ana_smos = t_df_smos[~np.isnan(t_df_smos[1]) | ~np.isnan(t_df_smos[2])].resample('1d').mean().index

        var = 'sm_surface'
        for mode in modes:

            if mode == 'absolute':
                ts_ref = ts_ascat.copy()
            else:
                ts_ref = calc_anom(ts_ascat.copy(), longterm=(mode == 'longterm')).dropna()

            for run, ts_model in zip(names, dss):

                t_ana = t_ana_smos if run == 'MadKF_SMOS40' else t_ana_smap

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
                # res['p_' + run + '_' + mode] = p
                # res['rmsd_' + run + '_' + mode] = np.sqrt(((tmp[1] - tmp[2]) ** 2).mean())
                res['ubrmsd_' + run + '_' + mode] = np.sqrt(
                    (((tmp[1] - tmp[1].mean()) - (tmp[2] - tmp[2].mean())) ** 2).mean())

                tmp = pd.DataFrame({1: ts_ref, 2: ts_mod}).reindex(t_ana).dropna()
                res['ana_len_' + run + '_' + mode] = len(tmp)
                r, p = pearsonr(tmp[1], tmp[2]) if len(tmp) > 10 else (np.nan, np.nan)
                res['ana_r_' + run + '_' + mode] = r
                # res['ana_p_' + run + '_' + mode] = p
                # res['ana_rmsd_' + run + '_' + mode] = np.sqrt(((tmp[1] - tmp[2]) ** 2).mean())
                res['ana_ubrmsd_' + run + '_' + mode] = np.sqrt(
                    (((tmp[1] - tmp[1].mean()) - (tmp[2] - tmp[2].mean())) ** 2).mean())

        if not result_file.exists():
            res.to_csv(result_file, float_format='%0.3f')
        else:
            res.to_csv(result_file, float_format='%0.3f', mode='a', header=False)


## --------------------------------------- PLOTS ---------------------------------------

def plot_ismn_eval():

    analysis_only = True
    n_min = 1000

    fout = '/Users/u0116961/Documents/work/LDAS/2020-03_scaling/plots/R_stats_ismn.png'

    offs = 1 if analysis_only else 0

    # extract length columns into separate data frame and select columns with all time steps or only analysis time steps
    df = pd.read_csv('/Users/u0116961/Documents/work/LDAS/2020-03_scaling/validation/ismn_eval.csv', index_col=0)
    df = df[[x for x in df.columns if 'sm_profile' not in x]]
    len_df = df[[x for x in df.columns if "len" in x]]
    df = df.iloc[:, 6::].drop([x for x in df.columns if "len" in x], axis='columns')
    if analysis_only:
        df = df.drop([x for x in df.columns if not "ana_" in x], axis='columns')
        len_df = len_df.drop([x for x in len_df.columns if not "ana_" in x], axis='columns')
    else:
        df = df.drop([x for x in df.columns if "ana_" in x], axis='columns')
        len_df = len_df.drop([x for x in len_df.columns if "ana_" in x], axis='columns')

    # mask out all values below a certain sample size
    info = []
    for col, val  in df.iteritems():
        c = '_'.join(col.split('_')[-3::])
        ind = (len_df[[x for x in len_df.columns if c in x]] < n_min).values.flatten()
        val.loc[ind] = np.nan
        info += [f'{"_".join(col.split("_")[-2::])}: {len(val.dropna())} stations with more than {n_min} measurements.']
    [print(i) for i in np.unique(info)]

    # mask out all correlations that are below a certain threshold
    tmp = df[[x for x in df.columns if '_r_' in x]].values
    tmp[tmp < 0.2] = np.nan
    df[[x for x in df.columns if '_r_' in x]] = tmp



    # extract information about individual columns (which experiment, surface/root zone, r / rmsd, ...)
    exp = np.array(['_'.join(x.split('_')[1+offs:-3]) for x in df.columns])
    param = np.array([x.split('_')[0+offs] for x in df.columns])
    mode = np.array([f"{'_'.join(x.split('_')[-2:])} / {x.split('_')[-3]}" for x in df.columns])

    # Calculate all values relative to the open-loop
    tmpdf = df.copy()
    for e in np.unique(exp):
        for m in np.unique(mode):
            tmpdf.loc[:, df.columns[(exp==e) & (mode==m)]] = \
                df.loc[:, df.columns[(exp==e) & (mode==m)]].values - df.loc[:, df.columns[(exp=='open_loop') & (mode==m)]].values

    # Prepare data frame for seaborn plotting
    tmpdf.columns = [exp, param, mode]
    tmpdf = tmpdf.melt(var_name=['exp', 'par', 'mod'], value_name='val')
    tmpdf = tmpdf[(tmpdf['exp']!='open_loop') & (tmpdf['par']=='r')]


    ylim = (-0.5,0.5)

    sns.set_context('talk', font_scale=0.8)
    g = sns.catplot(x='exp', y='val', data=tmpdf, col='mod', kind='violin', sharey=False, gridsize=500, aspect=1.2, ylim=[0.5,1], col_wrap=3)
    [ax.set(ylim=ylim) for ax in g.axes]
    [ax.axhline(color='black', linestyle='--', linewidth=1.5)  for ax in g.axes]
    g.set_titles('{col_name}')
    g.set_ylabels('')
    g.set_xlabels('')
    g.set_xticklabels(rotation=15)

    # plt.show()

    g.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ascat_eval():

    analysis_only = True
    relative = True
    ref_exp = 'open_loop'

    sub = 'ana_' if analysis_only else ''
    fext = '_rel' if relative else '_abs'

    res = pd.read_csv('/Users/u0116961/Documents/work/LDAS/2020-03_scaling/validation/ascat_eval.csv', index_col=0)

    cbr_r = [0.6, 1.4] if relative else [0.2,0.7]
    cbr_r = [-0.3, 0.3] if relative else [0.2,0.7]
    cbr_len = [500, 1500]

    exps = ['SMOSSMAP_short','MadKF_SMOS40']
    # exps = ['SMOS_long', 'SMOSSMAP_long', 'SMOS_short', 'SMOSSMAP_short', 'SMOSSMAP_short_PCA', 'SMAP_short']
    modes = ['absolute', 'longterm', 'shortterm']

    params = ['len', 'r']
    cbranges = [cbr_len, cbr_r]

    # params = ['r']
    # cbranges = [cbr_r]

    fontsize = 12

    for p, cbr in zip(params, cbranges):

        cmap = 'seismic_r' if (p == 'r') & relative else 'YlGn'

        f = plt.figure(figsize=(10,8))

        for i, m in enumerate(modes):
            for j, e in enumerate(exps):

                plt.subplot(len(modes),len(exps), len(exps)*i + j + 1)

                col = f'{sub}{p}_{e}_{m}' if p != 'len' else  f'{sub}{p}_{e}{m}'
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
        pos1 = f.axes[-2].get_position()
        pos2 = f.axes[-1].get_position()
        x1 = (pos1.x0 + pos1.x1) / 2
        x2 = (pos2.x0 + pos2.x1) / 2
        cbar_ax = f.add_axes([x1, 0.03, x2-x1, 0.03])
        cbar = f.colorbar(im, orientation='horizontal', cax=cbar_ax)
        for t in cbar.ax.get_xticklabels():
            t.set_fontsize(fontsize)

        fout = f'/Users/u0116961/Documents/work/LDAS/2020-03_scaling/plots/ascat_eval_{p}{sub}{fext}.png'
        f.savefig(fout, dpi=300, bbox_inches='tight')
        plt.close()

        # plt.show()


def plot_da_performance_vs_madkf_uncertainty():

    fout = '/Users/u0116961/Documents/work/LDAS/2020-03_scaling/plots/skillgain_vs_pert.png'

    io = LDAS_io()

    # fA = '/Users/u0116961/Documents/work/MadKF/CLSM/iter_612/error_files/SMOS_fit_Tb_A_uncorr.bin'
    # fD = '/Users/u0116961/Documents/work/MadKF/CLSM/iter_612/error_files/SMOS_fit_Tb_D_uncorr.bin'
    # dtype, hdr, length = template_error_Tb40()
    # imgA = io.read_fortran_binary(fA, dtype, hdr=hdr, length=length)
    # imgD = io.read_fortran_binary(fD, dtype, hdr=hdr, length=length)
    # imgA.index += 1
    # imgD.index += 1
    # pert = (imgA['err_Tbh'] + imgA['err_Tbv'] + imgD['err_Tbh'] + imgD['err_Tbv'])/4
    # pert.index = colrow2easegpi(io.grid.tilecoord.i_indg,io.grid.tilecoord.j_indg, glob=True)

    fname = '/Users/u0116961/Documents/work/MadKF/CLSM/SMOS40/iter_612/result_files/mse.csv'
    mse = pd.read_csv(fname, index_col=0)
    mse.index = colrow2easegpi(io.grid.tilecoord.i_indg,io.grid.tilecoord.j_indg, glob=True)

    mse.loc[:, f'rel_mse_spc_avg'] = 0
    for spc in np.arange(1,5):
        mse.loc[:,f'rel_mse_spc{spc}'] = mse['mse_fcst_spc%i'%spc] / mse['mse_obs_spc%i'%spc]
        mse.loc[:, f'rel_mse_spc_avg'] += mse.loc[:,f'rel_mse_spc{spc}']
    mse.loc[:, f'rel_mse_spc_avg'] /= 4

    res = pd.read_csv('/Users/u0116961/Documents/work/LDAS/2020-03_scaling/validation/ascat_eval.csv', index_col=0)
    exps = ['SMOSSMAP_short', 'MadKF_SMOS40']
    modes = ['absolute', 'longterm', 'shortterm']
    cols = []
    for m in modes:
        for e in exps:
            col = f'ana_r_{e}_{m}'
            ref_col = f'ana_r_open_loop_{m}'
            res[f'{e} - {m}'] = (res[col] - res[ref_col]) / (1 - res[ref_col])
            cols += [f'{e} - {m}',]
    # res = res[[x for x in res.columns if 'diff_' in x]]
    res['P/R'] = mse[f'rel_mse_spc_avg']

    df = res.melt('P/R', cols, 'experiment', 'Relative Skill Gain')

    plt.figure(figsize=(15,5))
    sns.set_context('talk', font_scale=0.8)
    g = sns.relplot(x="P/R", y="Relative Skill Gain", col = "experiment", kind = "scatter", data = df, col_wrap=2, aspect=2)
    [ax.set_xlim(0,2) for ax in g.axes]
    [ax.set_ylim(-1,1) for ax in g.axes]
    [ax.axhline(0, color='black', linestyle='--', linewidth=1.5) for ax in g.axes]
    [ax.axvline(1, color='black', linestyle='--', linewidth=1.5) for ax in g.axes]

    g.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

    # plt.show()

def skill_vs_weight_density_plot():

    fout = '/Users/u0116961/Documents/work/LDAS/2020-03_scaling/plots/skillgain_vs_pert_density.png'

    io = LDAS_io()

    xlim = [0, 0.5]
    ylim = [-0.6, 0.6]

    fname = '/Users/u0116961/Documents/work/MadKF/CLSM/SMOS40/iter_612/result_files/mse.csv'
    mse = pd.read_csv(fname, index_col=0)
    mse.index = colrow2easegpi(io.grid.tilecoord.i_indg, io.grid.tilecoord.j_indg, glob=True)

    mse.loc[:, f'rel_mse_spc_avg'] = 0
    for spc in np.arange(1, 5):
        mse.loc[:, f'rel_mse_spc{spc}'] = mse['mse_fcst_spc%i' % spc] / mse['mse_obs_spc%i' % spc]
        mse.loc[:, f'rel_mse_spc_avg'] += mse.loc[:, f'rel_mse_spc{spc}']
    mse.loc[:, f'rel_mse_spc_avg'] /= 4

    res = pd.read_csv('/Users/u0116961/Documents/work/LDAS/2020-03_scaling/validation/ascat_eval_old.csv', index_col=0)
    exps = ['SMOSSMAP_short', 'MadKF_SMOS40']
    modes = ['absolute', 'longterm', 'shortterm']
    cols = []
    for m in modes:
        for e in exps:
            col = f'ana_r_{e}_{m}'
            ref_col = f'ana_r_open_loop_{m}'
            res[f'{e} - {m}'] = (res[col] - res[ref_col]) / (1 - res[ref_col])
            cols += [f'{e} - {m}', ]
    # res = res[[x for x in res.columns if 'diff_' in x]]
    res['P/R'] = mse[f'rel_mse_spc_avg']
    res = res[(res['P/R'] >= xlim[0]) & (res['P/R'] <= xlim[1])]

    sns.set_context('notebook', font_scale=0.8)
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=False)

    f = plt.figure(figsize=(12, 8))

    for i,m in enumerate(modes):
        for j,e in enumerate(exps):
            ax = plt.subplot(len(modes),len(exps), i*len(exps) + j + 1)
            tmp_df = res[['P/R', f'{e} - {m}']].dropna()
            med = res[f"{e} - {m}"].median()
            ax.text(0.35, -0.5, f'Median = {med:.3f}', fontsize=10)

            plt.hexbin(tmp_df['P/R'], tmp_df[f'{e} - {m}'], bins= 'log', gridsize=50, cmap=cmap)
            plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
            plt.axvline(1, color='black', linestyle='--', linewidth=1.5)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.title(f'{e} - {m}')

            if i == 2:
                plt.xlabel('P / R')
            if (i == 1) & (j == 0):
                plt.ylabel('Skill gain w.r.t. open-loop')

    f.subplots_adjust(hspace=0.35)
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

    # plt.show()

def compare_files():

    res = pd.read_csv('/Users/u0116961/Documents/work/LDAS/2020-03_scaling/validation/ascat_eval.csv', index_col=0)

    exps = ['SMOS_long', 'SMOSSMAP_long', 'SMOS_short', 'SMOSSMAP_short', 'SMOSSMAP_short_PCA', 'SMAP_short']
    # modes = ['absolute', 'longterm', 'shortterm']
    modes = ['absolute',]

    df = pd.DataFrame()
    for m in modes:
        for e in exps:
            col = f'ana_r_{e}_{m}'
            ref_col = f'ana_r_open_loop_{m}'
            df[f'{e}_{m}'] = (res[col] - res[ref_col]) / (1 - res[ref_col])

    for m in modes:

        xcol = f'SMOSSMAP_short_{m}'

        fout = f'/Users/u0116961/Documents/work/LDAS/2020-03_scaling/plots/scaling_comparison_{m}.png'
        cols = [x for x in df.columns if (x != xcol) and (m in x)]
        tmp_df = df.melt(xcol, cols, 'Exp.', 'Skill Gain')

        sns.set_context('talk', font_scale=0.8)
        g = sns.relplot(x=xcol, y="Skill Gain", col="Exp.", kind="scatter", data=tmp_df, aspect=1)

        for col, ax in zip(cols, g.axes[0]):
            ax.plot([-1, 1], [-1, 1], linewidth=2, color='black', linestyle='--')
            ax.set(ylim=(-1.1, 1.1))
            diff = (df[xcol] - df[col]).median()
            ax.text(0.2, -1, f'Median = {diff:.3f}', fontsize=14)


        g.savefig(fout, dpi=300, bbox_inches='tight')
        plt.close()

#     plt.show()


def compare_scaling_density_plot():

    res = pd.read_csv('/Users/u0116961/Documents/work/LDAS/2020-03_scaling/validation/ascat_eval.csv', index_col=0)

    exps = ['SMOS_long', 'SMOSSMAP_long', 'SMOS_short', 'SMOSSMAP_short', 'SMOSSMAP_short_PCA', 'SMAP_short']
    # modes = ['absolute', 'longterm', 'shortterm']
    modes = ['absolute',]

    df = pd.DataFrame()
    for m in modes:
        for e in exps:
            col = f'ana_r_{e}_{m}'
            ref_col = f'ana_r_open_loop_{m}'
            df[f'{e}_{m}'] = (res[col] - res[ref_col]) / (1 - res[ref_col])

    for m in modes:

        xcol = f'SMOSSMAP_short_{m}'

        fout = f'/Users/u0116961/Documents/work/LDAS/2020-03_scaling/plots/scaling_comparison_{m}.png'
        cols = [x for x in df.columns if (x != xcol) and (m in x)]
        tmp_df = df.melt(xcol, cols, 'Exp.', 'Skill Gain')

        sns.set_context('talk', font_scale=0.8)
        cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=False)
        sns.jointplot(df[xcol], df[f'{e}_{m}'], cmap=cmap, kind='hex', bins=30, xlim=[-0.4,0.4], ylim=[-0.4,0.4] )
        # g = sns.relplot(x=xcol, y="Skill Gain", col="Exp.", kind="scatter", data=tmp_df, aspect=1)

        # for col, ax in zip(cols, g.axes[0]):
        #     ax.plot([-1, 1], [-1, 1], linewidth=2, color='black', linestyle='--')
        #     ax.set(ylim=(-1.1, 1.1))
        #     diff = (df[xcol] - df[col]).median()
        #     ax.text(0.2, -1, f'Median = {diff:.3f}', fontsize=14)


        # g.savefig(fout, dpi=300, bbox_inches='tight')
        # plt.close()

    plt.show()

if __name__=='__main__':

    # run_ismn_eval()
    # plot_ismn_eval()

    # run_ascat_eval()
    # run_ascat_eval_part(1, 1)
    plot_ascat_eval()

    # skill_vs_weight_density_plot()

    # compare_scaling_files()
