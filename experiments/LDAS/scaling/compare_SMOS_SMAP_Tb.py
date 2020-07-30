
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

from itertools import combinations

from multiprocessing import Pool

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import seaborn as sns

from pyldas.grids import EASE2
from pyldas.interface import LDAS_io
from pyldas.templates import template_scaling
from pyldas.visualize.plots import plot_ease_img

from pytesmo.temporal_matching import matching, df_match
from myprojects.experiments.LDAS.scaling.create_scaling_parameters import PCA

def getcols():
    return np.array(['OBS_H_A', 'MOD_H_A', 'OBS_V_A', 'MOD_V_A', 'OBS_H_D', 'MOD_H_D', 'OBS_V_D', 'MOD_V_D'])

def abbr2per(abbr):
    return '2015-04-01_2020-04-01' if abbr.lower() == 'short' else '2010-01-01_2020-04-01'

def abbr2pent(abbr):
    return '2015_p19_2020_p19' if abbr.lower() == 'short' else '2010_p01_2020_p19'

def create_climatology_ts(experiment):

    sensor, date = experiment[0], abbr2pent(experiment[1])

    root = Path(f'/Users/u0116961/data_sets/LDASsa_runs/scaling_files')
    dir_out = Path(f'/Users/u0116961/Documents/work/LDAS/2020-03_scaling/climatologies')
    fout = f'{experiment[0]}_{experiment[1]}'

    if not dir_out.exists():
        Path.mkdir(dir_out, parents=True)

    fbase = str(list(root.glob(f'*src_{sensor}_trg*{date}*.bin'))[0].name)[0:-10]

    io = LDAS_io()
    idx = io.grid.tilecoord.tile_id.values
    dtype, hdr, length = template_scaling(sensor='SMOS40')

    res = np.empty((9911, 73, 8, 2))

    # Thrid axis: ['OBS_H_A', 'MOD_H_A', 'OBS_V_A', 'MOD_V_A', 'OBS_H_D', 'MOD_H_D', 'OBS_V_D', 'MOD_V_D']
    # --> use getcols()
    # Fourth axis: mean / stddev.

    for pent in range(1,74):
        for node in ['A','D']:

            print(f'{node}sc., pentad {pent}')

            fname = root / f'{fbase}_{node}_p{pent:02}.bin'

            tmp = io.read_fortran_binary(fname, dtype, hdr, length)
            tmp.index = tmp.tile_id

            offs = 0 if node == 'A' else 4
            res[:, pent-1, 0+offs, 0] = tmp['m_obs_H_40'].reindex(idx)
            res[:, pent-1, 1+offs, 0] = tmp['m_mod_H_40'].reindex(idx)
            res[:, pent-1, 2+offs, 0] = tmp['m_obs_V_40'].reindex(idx)
            res[:, pent-1, 3+offs, 0] = tmp['m_mod_V_40'].reindex(idx)

            res[:, pent-1, 0+offs, 1] = tmp['s_obs_H_40'].reindex(idx)
            res[:, pent-1, 1+offs, 1] = tmp['s_mod_H_40'].reindex(idx)
            res[:, pent-1, 2+offs, 1] = tmp['s_obs_V_40'].reindex(idx)
            res[:, pent-1, 3+offs, 1] = tmp['s_mod_V_40'].reindex(idx)

    np.place(res, res == -9999, np.nan)

    np.save(dir_out / fout, res)

    '''
    import numpy as np
    import pandas as pd
    arr = np.load('/Users/u0116961/Documents/work/LDAS/2020-03_scaling/scaling_files_reshuffled/climatologies_SMAP_short.npy')
    cols = ['OBS_H_A', 'MOD_H_A', 'OBS_V_A', 'MOD_V_A', 'OBS_H_D', 'MOD_H_D', 'OBS_V_D', 'MOD_V_D']
    pd.DataFrame(arr[100,:,:,0], columns=cols).plot()
    '''

def calc_stats(experiments):

    root = Path(f'/Users/u0116961/Documents/work/LDAS/2020-03_scaling')

    ds = []
    for (sens, abbr) in experiments:
        data = np.load(root / 'climatologies' / f'{sens}_{abbr}.npy')
        # Only necessary to match SMOS and SMAP node directions in case of different scaling targets
        # if sens == 'SMAP':
        #     tmp_data = data.copy()
        #     data[:,:,0,:], data[:,:,4,:] = tmp_data[:,:,4,:], tmp_data[:,:,0,:]
        #     data[:,:,1,:], data[:,:,5,:] = tmp_data[:,:,5,:], tmp_data[:,:,1,:]
        #     data[:,:,2,:], data[:,:,6,:] = tmp_data[:,:,6,:], tmp_data[:,:,2,:]
        #     data[:,:,3,:], data[:,:,7,:] = tmp_data[:,:,7,:], tmp_data[:,:,3,:]
        ds += [data]

    cols = getcols()

    names = combinations(['_'.join(exp) for exp in experiments], 2)
    dss = combinations(ds, 2)

    for ((exp1, exp2), (ds1, ds2)) in zip(names, dss):
        print(f'{exp1}, {exp2}')
        res_cols = [f'{var}_{metric}_{col}_{exp1}_{exp2}' for var in ['mean', 'std'] for metric in ['bias','mae','corr'] for col in cols]
        res = pd.DataFrame(columns=res_cols, index=range(1,9912))

        for i_v, var in enumerate(['mean','std']):
            for i_c, col in enumerate(cols):
                res[f'{var}_bias_{col}_{exp1}_{exp2}'] = np.nanmean(ds1[:, :, i_c, i_v] - ds2[:, :, i_c, i_v],axis=1)
                res[f'{var}_mae_{col}_{exp1}_{exp2}'] = np.nanmean(np.abs(ds1[:, :, i_c, i_v] - ds2[:, :, i_c, i_v]),axis=1)
                res[f'{var}_corr_{col}_{exp1}_{exp2}'] = np.diag(pd.DataFrame(np.vstack((ds1[:, :, i_c, i_v], ds2[:, :, i_c, i_v])).T).corr().loc[range(0,9911),range(9911,2*9911)])

        fout = root / 'difference_stats' / f'{exp1}_{exp2}.csv'
        res.to_csv(fout, float_format='%.4f')

def plot_climatology_ts(experiments):

    # lat, lon = 41.509352, -110.254093 # Wyoming (high bias)
    # lat, lon = 32.300219, -107.117220 # New Mexico (low corr)
    # lat, lon = 48.206665, -100.257308 # North Dacota (good)

    gpis = [[41.509352, -110.254093], # Wyoming (high bias)
            [32.300219, -107.117220], # New Mexico (low corr)
            [48.206665, -100.257308]] # North Dacota (good)

    root = Path(f'/Users/u0116961/Documents/work/LDAS/2020-03_scaling/scaling_files_reshuffled')
    names = ['_'.join(exp) for exp in experiments]
    cols = getcols()
    dss = []
    for (sens, abbr) in experiments:
        dss += [np.load(root / f'climatologies_{sens}_{abbr}.npy')]

    fontsize=10
    colors = ['orange','b','r','g','grey','magenta']

    for lat, lon in gpis:
        tile = LDAS_io().grid.lonlat2tilenum(lon, lat)

        plt.figure(figsize=(18,10))

        for n, node in enumerate(['A','D']):
            for s, src in enumerate(['OBS','MOD']):

                idx1 = np.where(cols == f'{src}_V_{node}')[0][0]
                idx2 = np.where(cols == f'{src}_H_{node}')[0][0]

                for p, param in enumerate(['Mean', 'Std.dev']):

                    ax = plt.subplot(4,2, (p+1) + 2*s + 4*n)
                    res_v = pd.DataFrame(index=range(73))
                    res_h = pd.DataFrame(index=range(73))
                    for ds, name in zip(dss, names):
                        res_v[name] = ds[tile, :, idx1, p]
                        res_h[name] = ds[tile, :, idx2, p]

                    res_v.plot(ax=ax, xlim=(-2,75), fontsize=fontsize, style='-', color=colors, legend=True if n+s+p == 0 else False)
                    res_h.plot(ax=ax, xlim=(-2,75), fontsize=fontsize, style='--', color=colors, legend=False)
                    plt.title(f'{param} / {src}. / {node}sc.     - V-pol / -- H-pol', fontsize=fontsize)

                    if 2*s + 4*n != 6:
                        ax.set_xticks([])
                    else:
                        plt.xlabel('pentad', fontsize=fontsize)


        plt.gcf().savefig(root / f'clim_ts_{tile}.png', dpi=300, bbox_inches='tight')
        plt.close()
        # plt.tight_layout()
        # plt.show()

def plot_Tb_ts():

    # lat, lon = 41.509352, -110.254093 # Wyoming (high bias)
    # lat, lon = 32.300219, -107.117220 # New Mexico (low corr)
    lat, lon = 48.206665, -100.257308 # North Dacota (good)

    root = Path('/Users/u0116961/Documents/work/LDAS/2020-03_scaling')

    io_smos = LDAS_io('ObsFcstAna', exp='US_M36_SMOS40_TB_OL_noScl')
    io_smap = LDAS_io('ObsFcstAna', exp='US_M36_SMAP_TB_OL_noScl')
    # stats = np.load(root / 'TB_stats_lon_lat_bias_corr.npy')

    idx_lon, idx_lat = io_smos.grid.lonlat2colrow(lon, lat, domain=True)
    print(idx_lon, idx_lat)

    ts_smos = io_smos.timeseries['obs_obs'].isel(lat=idx_lat, lon=idx_lon).to_pandas()
    ts_smap = io_smap.timeseries['obs_obs'].isel(lat=idx_lat, lon=idx_lon).to_pandas()

    plt.figure(figsize=(18,11))
    fontsize=12

    sns.set_context('talk', font_scale=0.8)
    pal1 = sns.color_palette()
    pal2 = pal1.copy()
    pal3 = pal1.copy()
    pal2[0], pal2[1] = pal2[1], pal2[0]
    pal3[0] = pal1[2]

    experiments = [[['SMOS'], 'long'],
                   [['SMOS', 'SMAP'], 'long'],
                   [['SMOS'], 'short'],
                   [['SMAP'], 'short'],
                   [['SMOS', 'SMAP'], 'short'],
                   [['PC-1'], 'short']]

    spc_smos = 1
    spc_smap = 2
    ts_pca = PCA(ts_smos[spc_smos], ts_smap[spc_smap], window=1.5)['PC-1']
    df = pd.concat((ts_smos[spc_smos], ts_smap[spc_smap], ts_pca), axis=1)
    df.columns = ['SMOS', 'SMAP', 'PC-1']
    df['time'] = df.index


    for i, (sensors, period) in enumerate(experiments):

        ax = plt.subplot(len(experiments),1,i+1)
        #

        tmp_df = df.copy()
        if period == 'short':
            tmp_df.loc[tmp_df.index < '2015-04-01',:] = np.nan

        if sensors[0] == 'SMAP':
            sns.set_palette(sns.color_palette(pal2))
        elif sensors[0] == 'PC-1':
            sns.set_palette(sns.color_palette(pal3))
        else:
            sns.set_palette(sns.color_palette(pal1))

        sns.lineplot(x='time', y='Tb', hue='Sensor', data=tmp_df.melt('time', sensors, 'Sensor', 'Tb'), ax=ax)

        # df.interpolate('linear').plot(fontsize=fontsize, ax=plt.gca(), legend=(False if i != 0 else True))
        # plt.xlabel('')
        # plt.title(f'Bias: {stats[i+2, idx_lat, idx_lon]:.2f} , Correlation: {stats[i+6, idx_lat, idx_lon]:.2f}', fontsize=fontsize)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xlim([df.index[0],df.index[-1]])
        if i != len(experiments)-1:
            ax.set_xticks([])

    plt.gcf().savefig(root / 'plots' / f'Tb_ts_{idx_lat}_{idx_lon}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # plt.tight_layout()
    # plt.show()


def plot_Tb_clim_years_ts():

    src = 'obs'

    lat, lon = 41.509352, -110.254093 # Wyoming (high bias)
    # lat, lon = 32.300219, -107.117220 # New Mexico (low corr)
    # lat, lon = 48.206665, -100.257308 # North Dacota (good)

    root = Path('/Users/u0116961/Documents/work/LDAS/2020-03_scaling')

    io_smos = LDAS_io('ObsFcstAna', exp='US_M36_SMOS40_TB_OL_noScl')
    io_smap = LDAS_io('ObsFcstAna', exp='US_M36_SMAP_TB_OL_noScl')
    # stats = np.load(root / 'TB_stats_lon_lat_bias_corr.npy')

    idx_lon, idx_lat = io_smos.grid.lonlat2colrow(lon, lat, domain=True)

    ts_smos = io_smos.timeseries[f'obs_{src}'].isel(lat=idx_lat, lon=idx_lon).to_pandas()['2015-04-01':'2020-04-01']
    ts_smap = io_smap.timeseries[f'obs_{src}'].isel(lat=idx_lat, lon=idx_lon).to_pandas()['2015-04-01':'2020-04-01']

    plt.figure(figsize=(18,11))
    fontsize=12

    # for i, (spc_smos, spc_smap) in enumerate(zip([1, 2, 3, 4], [2, 1, 4, 3])):

    i=0
    spc_smos = 1
    spc_smap = 2

    ts_pca = PCA(ts_smos[spc_smos], ts_smap[spc_smap], window=1.5)['PC-1']

    df = pd.concat((ts_smos[spc_smos], ts_smap[spc_smap], ts_pca), axis=1)
    df.columns = ['SMOS', 'SMAP', 'PC-1']

    if i != 3:
        plt.gca().set_xticks([])
    for j, sens in enumerate(df.columns.values):
        plt.subplot(3,1,j+1)
        for yr in range(2015,2021):
            tmp_df = df[df.index.year==yr][sens]
            tmp_df.index = tmp_df.index.dayofyear
            tmp_df.interpolate('linear').plot(fontsize=fontsize, ax=plt.gca(), legend=True, linewidth=1)
        plt.xlabel('')

            # plt.title(f'Bias: {stats[i+2, idx_lat, idx_lon]:.2f} , Correlation: {stats[i+6, idx_lat, idx_lon]:.2f}', fontsize=fontsize)

    # plt.gcf().savefig(root / f'Tb_ts_{idx_lat}_{idx_lon}.png', dpi=300, bbox_inches='tight')
    # plt.close()

    plt.tight_layout()
    plt.show()

def calc_Tb_stats():

    dir_out = Path('/Users/u0116961/Documents/work/LDAS/2020-03_scaling')

    io_smos = LDAS_io('ObsFcstAna', exp='US_M36_SMOS40_TB_OL_noScl')
    io_smap = LDAS_io('ObsFcstAna', exp='US_M36_SMAP_TB_OL_noScl')

    lons = io_smap.timeseries['lon']
    lats = io_smap.timeseries['lat']
    lons, lats = np.meshgrid(lons, lats)

    # lon, lat, bias (spc 1-4), MAD (spc 1-4), corr (spc 1-4)
    res = np.full(((14,) + lons.shape), np.nan)
    res[[0,1],:,:] = (lons, lats)

    for idx_lat in range(lons.shape[0]):
        for idx_lon in range(lons.shape[1]):
            print(f'idx_lat: {idx_lat}, idx_lon: {idx_lon}')

            ts_smos = io_smos.timeseries['obs_obs'].isel(lat=idx_lat, lon=idx_lon).to_pandas()['2015-04-01':'2020-04-01']
            ts_smap = io_smap.timeseries['obs_obs'].isel(lat=idx_lat, lon=idx_lon).to_pandas()['2015-04-01':'2020-04-01']

            for i, (spc_smos, spc_smap) in enumerate(zip([1, 2, 3, 4], [2, 1, 4, 3])):

                tmp_smos = ts_smos[spc_smos].dropna(); tmp_smos.name = 'SMOS'
                tmp_smap = ts_smap[spc_smap].dropna(); tmp_smap.name = 'SMAP'
                if (len(tmp_smos) == 0) | (len(tmp_smap) == 0):
                    continue

                try:
                    if len(tmp_smos) < len(tmp_smap):
                        df = matching(tmp_smos, tmp_smap, window=1)
                    else:
                        df = matching(tmp_smap, tmp_smos, window=1)
                except:
                    print('matching error')
                    continue

                res[2+i, idx_lat, idx_lon] = np.mean(np.diff(df))
                res[6+i, idx_lat, idx_lon] = np.mean(np.abs(np.diff(df)))
                res[10+i, idx_lat, idx_lon] = np.corrcoef(df.T)[0,1]

    np.save(dir_out / 'TB_stats_lon_lat_bias_corr', res)

def plot_tb_res_img(data,idx,
                      cbrange=(-20,20),
                      cmap='jet',
                      title='',
                      fontsize=20):

    lons, lats = data[[0,1],:,:]
    img_masked = np.ma.masked_invalid(data[idx,:,:])

    m = Basemap(projection='mill',
                llcrnrlat=24,
                urcrnrlat=51,
                llcrnrlon=-128,
                urcrnrlon=-64,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])

    cbstep = round((cbrange[1]-cbrange[0])*100)/400
    labels = ['{:.2f}'.format(x) for x in np.arange(cbrange[0], cbrange[1] + cbstep, cbstep)]
    cb = m.colorbar(im, "bottom", size="7%", pad=0.05, ticks=[float(x) for x in labels])
    cb.ax.set_xticklabels(labels, fontsize=fontsize)

    plt.title(title,fontsize=fontsize)

def plot_tb_stats_map():

    root = Path('/Users/u0116961/Documents/work/LDAS/2020-03_scaling')
    res = np.load(root / 'TB_stats_lon_lat_bias_corr.npy')

    f = plt.figure(figsize=(18,9))

    cbrange_bias = (-4, 4)
    cbrange_mad = (0, 6)
    cbrange_corr = (0.8, 1)

    fontsize=12

    for spc in range(4):

        plt.subplot(3, 4, spc+1)
        plot_tb_res_img(res, spc+2, cbrange=cbrange_bias, title=f'Bias species {spc+1}', fontsize=fontsize, cmap='jet')

        plt.subplot(3, 4, spc+5)
        plot_tb_res_img(res, spc+6, cbrange=cbrange_mad, title=f'MAD species {spc+1}', fontsize=fontsize, cmap='hot_r')

        plt.subplot(3, 4, spc+9)
        plot_tb_res_img(res, spc+10, cbrange=cbrange_corr, title=f'Correlation species {spc+1}', fontsize=fontsize, cmap='hot_r')

    plt.gcf().savefig(root / 'plots' / 'Tb_stats.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_tb_stats_violins():

    root = Path('/Users/u0116961/Documents/work/LDAS/2020-03_scaling')
    df = pd.DataFrame(np.load(root / 'TB_stats_lon_lat_bias_corr.npy')[2::].reshape((12,-1)).T)
    df.columns = [np.tile([1,2,3,4], 3),np.repeat(['Bias','MAD','Correlation'],4)]
    df = df.melt(var_name=['spc', 'var'], value_name='val')

    ylims = [(-3, 6), (0, 7), (0.7, 1)]

    sns.set_context('talk', font_scale=0.8)
    g = sns.catplot(x='spc', y='val', data=df, col='var', kind='violin', sharey=False, gridsize=500, aspect=1.2)
    [ax.set(ylim=lim, xlabel=lab) for ax, lim, lab in zip(g.axes[0], ylims, ['', 'Species', ''])]
    g.axes[0][0].axhline(color='black', linestyle='--', linewidth=1.5)
    g.set_titles('{col_name}')
    g.set_ylabels('')

    g.savefig(root / 'plots' / f'Tb_stats_violins.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_stats(experiments):

    names = ['_'.join(c) for c in combinations(['_'.join(exp) for exp in experiments], 2)]

    root = Path(f'/Users/u0116961/Documents/work/LDAS/2020-03_scaling')
    cols = getcols()

    for name in names:

        res = pd.read_csv(root / 'difference_stats' / f'{name}.csv', index_col=0)

        # for param in ['bias', 'mae', 'corr']:
        for param in ['mae']:

            fout = root / 'plots' / 'difference_stats' / f'{param}_{name}.png'

            f = plt.figure(figsize=(20, 6))

            cbrange = [0.9, 1] if param == 'corr' else [-3,3] if param == 'bias' else [0, 5]
            cmap = 'jet'

            for i,col in enumerate(cols):
                plt.subplot(2,4,i+1)
                plot_ease_img(res, f'mean_{param}_{col}_{name}', cbrange=cbrange, cmap=cmap, fontsize=14,
                              plot_cb=(True if i > 3 else False), title=col)

            f.savefig(fout, dpi=300, bbox_inches='tight')
            plt.close()

def run(fct, experiments):

    xeps = np.atleast_2d(experiments)
    nprocs = xeps.shape[0]

    print(f'Running "{fct.__name__}" for {nprocs} experiment(s)...')

    p = Pool(nprocs)
    p.map(fct, xeps)


if __name__ == '__main__':

    experiments = [['SMOSSMAP_PCA', 'short'],
                   ['SMOSSMAP', 'long'],
                   ['SMOSSMAP', 'short'],
                   ['SMOS', 'long'],
                   ['SMOS', 'short'],
                   ['SMAP', 'short']]

    # experiments = ['SMAP', 'short']

    # run(create_climatology_ts, experiments)
    # plot_climatology_ts(experiments)

    # calc_stats(experiments)
    # plot_stats(experiments)

    plot_Tb_ts()
    # plot_Tb_clim_years_ts()

    # calc_Tb_stats()
    # plot_tb_stats_map()
    # plot_tb_stats_violins()
