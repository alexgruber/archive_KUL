
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import seaborn as sns

from pyldas.grids import EASE2
from pyldas.interface import LDAS_io
from pyldas.templates import template_scaling
from pyldas.visualize.plots import plot_ease_img

from pytesmo.temporal_matching import matching, df_match

def create_climatology_ts(exp):

    dir_out = Path(f'/Users/u0116961/Documents/work/LDAS/2020-03_scaling/{exp}/scaling_files')

    root = Path(f'/Users/u0116961/data_sets/LDASsa_runs/{exp}/scaling_files')
    fbase = str(list(root.glob('*.bin'))[0])[0:-10]

    io = LDAS_io()
    idx = io.grid.tilecoord.tile_id.values
    dtype, hdr, length = template_scaling(sensor='SMOS40')

    res = np.empty((9911, 73, 8))
    # Third axis:
    # ['OBS_A_H', 'OBS_A_V', 'OBS_D_H', 'OBS_D_V', 'MOD_A_H', 'MOD_A_V', 'MOD_D_H', 'MOD_D_V']

    for pent in range(1,74):
        for node in ['A','D']:

            print(f'{node}sc., pentad {pent}')

            fname = root / f'{fbase}_{node}_p{pent:02}.bin'

            tmp = io.read_fortran_binary(fname, dtype, hdr, length)
            tmp.index = tmp.tile_id

            offs = 0 if node == 'A' else 2
            res[:, pent-1, 0+offs] = tmp['m_obs_H_40'].reindex(idx)
            res[:, pent-1, 1+offs] = tmp['m_obs_V_40'].reindex(idx)
            res[:, pent-1, 4+offs] = tmp['m_mod_H_40'].reindex(idx)
            res[:, pent-1, 5+offs] = tmp['m_mod_V_40'].reindex(idx)

    np.place(res, res == -9999, np.nan)

    np.save(dir_out / 'climatologies', res)

    '''
    arr = np.load('/Users/u0116961/data_sets/LDASsa_runs/US_M36_SMOS40_TB_OL_noScl/scaling_files/climatologies.npy')
    pd.DataFrame(arr[100,:,:], columns=['OBS_A_H', 'OBS_A_V', 'OBS_D_H', 'OBS_D_V', 'MOD_A_H', 'MOD_A_V', 'MOD_D_H', 'MOD_D_V']).plot()
    '''


def calc_stats(exp):

    root = Path(f'/Users/u0116961/Documents/work/LDAS/2020-03_scaling')
    fout = root / exp / 'scaling_files' / 'difference_stats.csv'

    ref = np.load(root / 'scaling_files' / 'climatologies.npy')
    new = np.load(root / exp / 'scaling_files' / 'climatologies.npy')

    if 'SMAP' in exp:
        tmp_new = new.copy()
        new[:,:,0], new[:,:,2] = tmp_new[:,:,2], tmp_new[:,:,0]
        new[:,:,1], new[:,:,3] = tmp_new[:,:,3], tmp_new[:,:,1]
        new[:,:,4], new[:,:,6] = tmp_new[:,:,6], tmp_new[:,:,4]
        new[:,:,5], new[:,:,7] = tmp_new[:,:,7], tmp_new[:,:,5]

    cols = ['OBS_A_H', 'OBS_A_V', 'OBS_D_H', 'OBS_D_V', 'MOD_A_H', 'MOD_A_V', 'MOD_D_H', 'MOD_D_V']

    res_cols = ['bias_' + col for col in cols] + ['corr_' + col for col in cols]

    res = pd.DataFrame(columns=res_cols, index=range(1,9912))

    for idx, col in enumerate(cols):
        print(col)
        # res['bias_' + col] = np.nanmean(new[:, :, idx],axis=1) - np.nanmean(ref[:, :, idx],axis=1)
        res['bias_' + col] = np.nanmean(new[:, :, idx] - ref[:, :, idx],axis=1)
        res['corr_' + col] = np.diag(pd.DataFrame(np.vstack((ref[:, :, idx], new[:, :, idx])).T).corr().loc[range(0,9911),range(9911,2*9911)])

    res.to_csv(fout, float_format='%.4f')

def plot_climatology_ts(exp):

    root = Path(f'/Users/u0116961/Documents/work/LDAS/2020-03_scaling')
    arr = np.load(root / exp / 'scaling_files' / 'climatologies.npy')
    cols = ['OBS_A_H', 'OBS_A_V', 'OBS_D_H', 'OBS_D_V', 'MOD_A_H', 'MOD_A_V', 'MOD_D_H', 'MOD_D_V']
    res = pd.DataFrame(arr[100,:,:], columns=cols)

    res.plot(figsize=(14,6), xlim=(-2,75), ylim=(240,300), fontsize=12)
    plt.title('SMAP: 2015 - 2020', fontsize=14)
    plt.gcf().savefig(root / exp / 'scaling_files' / 'plot_ts.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_Tb_ts():

    root = Path('/Users/u0116961/Documents/work/LDAS/2020-03_scaling')

    io_smos = LDAS_io('ObsFcstAna', exp='US_M36_SMOS40_TB_OL_noScl')
    io_smap = LDAS_io('ObsFcstAna', exp='US_M36_SMAP_TB_OL_noScl')

    stats = np.load(root / 'TB_stats_lon_lat_bias_corr.npy')

    lat, lon = 41.509352, -110.254093 # Wyoming (high bias)
    # lat, lon = 32.300219, -107.117220 # New Mexico (low corr)
    # lat, lon = 48.206665, -100.257308 # North Dacota (good)

    idx_lon, idx_lat = io_smos.grid.lonlat2colrow(lon, lat, domain=True)

    ts_smos = io_smos.timeseries['obs_obs'].isel(lat=idx_lat, lon=idx_lon).to_pandas()['2015-01-03':'2020-01-01']
    ts_smap = io_smap.timeseries['obs_obs'].isel(lat=idx_lat, lon=idx_lon).to_pandas()['2015-01-03':'2020-01-01']

    plt.figure(figsize=(18,11))

    for i, (spc_smos, spc_smap) in enumerate(zip([1, 2, 3, 4], [2, 1, 4, 3])):

        tmp_smos = ts_smos[spc_smos].dropna(); tmp_smos.name = 'SMOS'
        tmp_smap = ts_smap[spc_smap].dropna(); tmp_smap.name = 'SMAP'
        if len(tmp_smos) < len(tmp_smap):
            df = matching(tmp_smos, tmp_smap, window=1)
        else:
            df = matching(tmp_smap, tmp_smos, window=1)

        plt.subplot(4,1,i+1)
        df.plot(xlim=('2015-01-01','2020-01-01'), fontsize=12, ax=plt.gca(), legend=(False if i != 0 else True))
        plt.xlabel('')
        plt.title(f'Bias: {stats[i+2, idx_lat, idx_lon]:.2f} , Correlation: {stats[i+6, idx_lat, idx_lon]:.2f}', fontsize=12)
        if i != 3:
            plt.gca().set_xticks([])

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

            ts_smos = io_smos.timeseries['obs_obs'].isel(lat=idx_lat, lon=idx_lon).to_pandas()['2015-01-03':'2020-01-01']
            ts_smap = io_smap.timeseries['obs_obs'].isel(lat=idx_lat, lon=idx_lon).to_pandas()['2015-01-03':'2020-01-01']

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

    plt.gcf().savefig(root / f'Tb_stats.png', dpi=300, bbox_inches='tight')
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

    g.savefig(root / f'Tb_stats_violins.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_stats(exp):

    root = Path(f'/Users/u0116961/Documents/work/LDAS/2020-03_scaling')
    res = pd.read_csv(root / exp / 'scaling_files' / 'difference_stats.csv', index_col=0)

    cols = ['OBS_A_H', 'OBS_A_V', 'OBS_D_H', 'OBS_D_V', 'MOD_A_H', 'MOD_A_V', 'MOD_D_H', 'MOD_D_V']

    # for param in ['bias', 'corr']:
    for param in ['bias', ]:
        f = plt.figure(figsize=(20, 6))

        cbrange = [0.7, 1] if param == 'corr' else [-11,11]
        cmap = 'jet' if param == 'corr' else 'jet'

        for i,col in enumerate(cols):
            plt.subplot(2,4,i+1)
            plot_ease_img(res, f'{param}_{col}', cbrange=cbrange, cmap=cmap, fontsize=14,
                          plot_cb=(True if i > 3 else False), title=col)

        f.savefig(root / exp / 'scaling_files' / f'plot_{param}.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':

    # exp = 'US_M36_SMOS40_TB_OL_noScl'
    # create_climatology_ts(exp)
    # calc_stats(exp)
    # plot_stats(exp)

    # calc_Tb_stats()
    plot_Tb_ts()
    # plot_tb_stats_map()
    # plot_tb_stats_violins()
