
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from math import floor
from pathlib import Path

from mpl_toolkits.basemap import Basemap
from matplotlib import gridspec, cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import colorcet as cc
sns.set_context('talk', font_scale=0.8)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pymannkendall as mk
from datetime import datetime
from bfast import BFASTMonitor

from myprojects.timeseries import calc_anom, calc_anomaly

from myprojects.publications.deforestation_paper.interface import io

def plot_img(lons, lats, data,
              llcrnrlat=-56.,
              urcrnrlat=13.,
              llcrnrlon=-82,
              urcrnrlon=-34,
              cbrange=None,
              cmap='jet',
              title='',
              fontsize=16,
              plot_cmap=True,
              return_map=False,
              return_cbar=False):

    img_masked = np.ma.masked_invalid(data)

    m = Basemap(projection='mill',
                llcrnrlat=llcrnrlat,
                urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,
                urcrnrlon=urcrnrlon,
                resolution='c')

    m.drawcoastlines()
    m.drawcountries()

    # m.drawstates()

    im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)

    if cbrange:
        im.set_clim(vmin=cbrange[0], vmax=cbrange[1])

    if plot_cmap:
        cb = m.colorbar(im, "bottom", size="4%", pad="2%")
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(fontsize)

    if title != '':
        plt.title(title,fontsize=fontsize)

    # x, y = m(-79, 27.5)
    # plt.text(x, y, 'mean', fontsize=fontsize - 5)
    # x, y = m(-59, -51)
    # plt.text(x, y, f'm. = {np.ma.mean(img_masked):.2f}' , fontsize=fontsize)
    # # x, y = m(-79, 25)
    # plt.text(x, y, 'std.', fontsize=fontsize - 5)
    # x, y = m(-74, 25)
    # plt.text(x, y, '   =%.2f' % np.ma.std(img_masked), fontsize=fontsize - 5)

    if return_map:
        return im, m
    elif return_cbar:
        return im, cb
    else:
        return im

def plot_centered_cbar(f, im, n_cols, wspace=0.04, hspace=0.025, bottom=0.06, pad=0.03, wdth=0.03, fontsize=12, col_offs=0):

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

    cbar_ax = f.add_axes([x1, pad, x2 - x1, wdth])
    cbar = f.colorbar(im, orientation='horizontal', cax=cbar_ax)
    for t in cbar.ax.get_xticklabels():
        t.set_fontsize(fontsize)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def AGB_VOD_LAI():

    print_agb_vod_lai = False
    print_agb_change = False
    print_agb_tcl = True

    ds_agb = io('AGB')
    ds_vod = io('SMOS_IC')
    ds_lai = io('LAI')
    # ds_sif = io('SIF')
    ds_tcl = io('TCL')

    trends = pd.read_csv('/Users/u0116961/Documents/work/deforestation_paper/trends.csv', index_col=0)
    lut = ds_agb.lut.reindex(trends.index)

    sig = np.full(ds_agb.lat.shape, 1.)
    sig[lut[trends[f'p_vod'] > 0.05].row_ease, lut[trends[f'p_vod'] > 0.05].col_ease] = np.nan

    corr = pd.read_csv('/Users/u0116961/Documents/work/deforestation_paper/lagged_corr.csv', index_col=0)
    lut2 = ds_agb.lut.reindex(corr.index)
    t = np.full(ds_agb.lat.shape, np.nan)
    p = np.full(ds_agb.lat.shape, np.nan)
    r = np.full(ds_agb.lat.shape, np.nan)
    t[lut.row_ease, lut.col_ease] = corr[f'R_VOD_T_-3']
    p[lut.row_ease, lut.col_ease] = corr[f'R_VOD_P_-3']
    r[lut.row_ease, lut.col_ease] = corr[f'R_VOD_R_-2']
    mask = np.full(ds_agb.lat.shape, np.nan)
    mask[(t < 0.5) & (p < 0.5) & (r < 0.5)] = 1

    tcl = ds_tcl.read_img('TCL')

    agb = ds_agb.read_img('AGB')
    agb_err = ds_agb.read_img('AGB_err')
    agb[agb_err > 160] = np.nan

    date_from = '2010-01-01'
    date_to = '2010-12-31'

    lai = np.nanmean(ds_lai.read_img('LAI', date_from=date_from, date_to=date_to), axis=0)

    vod = ds_vod.read_img('VOD', date_from=date_from, date_to=date_to)
    invalid = (ds_vod.read_img('Flags', date_from=date_from, date_to=date_to) > 0) | \
              (ds_vod.read_img('RMSE', date_from=date_from, date_to=date_to) > 8) | \
              (ds_vod.read_img('VOD_StdErr', date_from=date_from, date_to=date_to) > 1.2)
    vod[invalid] = np.nan
    vod2010 = np.nanmean(vod, axis=0)

    date_from = '2019-01-01'
    date_to = '2019-12-31'
    vod = ds_vod.read_img('VOD', date_from=date_from, date_to=date_to)
    invalid = (ds_vod.read_img('Flags', date_from=date_from, date_to=date_to) > 0) | \
              (ds_vod.read_img('RMSE', date_from=date_from, date_to=date_to) > 8) | \
              (ds_vod.read_img('VOD_StdErr', date_from=date_from, date_to=date_to) > 1.2)
    vod[invalid] = np.nan
    vod2019 = np.nanmean(vod, axis=0)

    # sif = ds_sif.read_img('sif_dc', date_from=date_from, date_to=date_to)
    # invalid = (ds_sif.read_img('n', date_from=date_from, date_to=date_to) <= 1) | \
    #           (ds_sif.read_img('cloud_fraction', date_from=date_from, date_to=date_to) > 0.7)
    # sif[invalid] = np.nan
    # sif = np.nanmean(sif, axis=0)
    # sif_ease = np.full(vod2019.shape, np.nan)
    # sif_ease[ds_sif.lut.row_ease, ds_sif.lut.col_ease] = sif[ds_sif.lut.row_sif, ds_sif.lut.col_sif]

    # Collocation
    invalid = np.where(np.isnan(agb) | np.isnan(vod2010) | np.isnan(vod2019) | np.isnan(lai))
    agb[invalid] = np.nan
    lai[invalid] = np.nan
    vod2010[invalid] = np.nan
    vod2019[invalid] = np.nan
    tcl[invalid] = np.nan
    # sif_ease[invalid] = np.nan
    valid = np.where(~np.isnan(agb))

    fontsize = 10

    if print_agb_vod_lai:

        f = plt.figure(figsize=(16,11))

        gs = gridspec.GridSpec(3, 4, height_ratios=[2,1,0.025], wspace=0.2, hspace=0.20)

        plt.subplot(gs[0,0])
        plot_img(ds_agb.lon, ds_agb.lat, agb, cbrange=(0,100), cmap='viridis_r', title='AGB [Mg/ha]', fontsize=fontsize)

        plt.subplot(gs[0,1])
        plot_img(ds_agb.lon, ds_agb.lat, lai, cbrange=(0,5), cmap='viridis_r', title='LAI [-]', fontsize=fontsize)

        plt.subplot(gs[0,2])
        plot_img(ds_agb.lon, ds_agb.lat, vod2010, cbrange=(0,0.8), cmap='viridis_r', title='VOD [-]', fontsize=fontsize)

        # plt.subplot(gs[0,3])
        # plot_img(ds_agb.lon, ds_agb.lat, sif_ease, cbrange=(0, 0.8), cmap='viridis_r', title='SIF [mW/m$^2$/sr/nm]', fontsize=fontsize)

        p_order = 3

        ax = plt.subplot(gs[1,1])
        x, y = lai[valid], agb[valid]
        ax.hexbin(x, y,
                  gridsize=35, bins='log',
                  cmap='viridis', mincnt=1)
        xs = np.linspace(x.min(),x.max(), 100)
        p = np.poly1d(np.polyfit(x, y, p_order))
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        plt.plot(xs, p(xs), linestyle='-.', linewidth=1, color='k')
        plt.xlim(xlim)
        plt.ylim(ylim)
        corr = np.corrcoef(x, y)[0,1]
        plt.title(f'R = {corr:.3f}', fontsize=fontsize)
        plt.xlabel('LAI [-]', fontsize=fontsize)
        plt.ylabel('AGB [Mg/ha]', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        ax = plt.subplot(gs[1,2])
        x, y = vod2010[valid], agb[valid]
        ax.hexbin(x, y,
                  gridsize=35, bins='log',
                  cmap='viridis', mincnt=1)
        xs = np.linspace(x.min(),x.max(), 100)
        p = np.poly1d(np.polyfit(x, y, p_order))
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        plt.plot(xs, p(xs), linestyle='-.', linewidth=1, color='k')
        plt.xlim(xlim)
        plt.ylim(ylim)
        corr = np.corrcoef(x, y)[0,1]
        plt.title(f'R = {corr:.3f}', fontsize=fontsize)
        plt.xlabel('VOD [-]', fontsize=fontsize)
        # plt.ylabel('AGB [Mg/ha]', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks([])

        # ax = plt.subplot(gs[1,3])
        # x, y = sif_ease[valid], agb[valid]
        # ax.hexbin(x, y,
        #           gridsize=35, bins='log',
        #           cmap='viridis', mincnt=1)
        # xs = np.linspace(x.min(),x.max(), 100)
        # p = np.poly1d(np.polyfit(x, y, p_order))
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        # plt.plot(xs, p(xs), linestyle='-.', linewidth=1, color='k')
        # plt.xlim(xlim)
        # plt.ylim(ylim)
        # corr = np.corrcoef(x, y)[0,1]
        # plt.title(f'R = {corr:.3f}', fontsize=fontsize)
        # plt.xlabel('SIF [mW/m$^2$/sr/nm]', fontsize=fontsize)
        # # plt.ylabel('AGB [Mg/ha]', fontsize=fontsize)
        # plt.xticks(fontsize=fontsize)
        # plt.yticks([])


        fname = '/Users/u0116961/Documents/work/deforestation_paper/plots/AGB_LAI_VOD_w_SIF.png'
        f.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()


    p = np.poly1d(np.polyfit(vod2010[valid], agb[valid], 3))
    diff = p(vod2019) - p(vod2010)

    # Various masking options

    # diff[np.isnan(sig)] = np.nan
    # tcl[np.isnan(sig)] = np.nan

    # diff[np.isnan(mask)] = np.nan
    # tcl[np.isnan(mask)] = np.nan

    # tcl[diff > 0] = np.nan
    # diff[diff > 0] = np.nan

    diff[tcl < 0.1] = np.nan
    tcl[tcl < 0.1] = np.nan

    valid = np.where(~np.isnan(diff))
    print(np.corrcoef(diff[valid], tcl[valid]))

    if print_agb_change:

        f = plt.figure(figsize=(12,10))

        plt.subplot(1, 2, 1)

        plot_img(ds_agb.lon, ds_agb.lat, diff / 9, cbrange=(-4,4), cmap=cc.cm.bjy,
                 title='$\Delta$ AGB [Mg/ha/yr] (2019 - 2010)', fontsize=fontsize + 4)

        plt.subplot(1, 2, 2)
        plot_img(ds_agb.lon, ds_agb.lat, tcl, cbrange=(0.1 , 0.6), cmap=truncate_colormap(cc.cm.bjy_r,0.5,1), title='Tree cover loss [-] (2019 - 2010)', fontsize=fontsize+4)

        fname = '/Users/u0116961/Documents/work/deforestation_paper/plots/AGB_change_2019_2010_diff_lt_0_p_lt_0.05.png'
        f.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

    if print_agb_tcl:

        f = plt.figure(figsize=(12,10))

        fontsize += 6
        x, y = diff[valid], tcl[valid]
        plt.hexbin(x, y,
                  gridsize=100, bins='log',
                  cmap='viridis', mincnt=1, )
        plt.axvline(color='k', linestyle='--', linewidth=1.5)
        corr = np.corrcoef(x, y)[0,1]
        plt.title(f'R = {corr:.3f}', fontsize=fontsize)
        plt.xlabel('$\Delta$ AGB', fontsize=fontsize)
        plt.ylabel('Tree cover loss', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlim([-50,50])
        plt.ylim(0,0.8)

        fname = '/Users/u0116961/Documents/work/deforestation_paper/plots/AGB_diff_vs_TCL_tcl_gt_0.1.png'
        f.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()


def mean_veg_met():

    ds_lai = io('LAI')
    ds_vod = io('SMOS_IC')
    ds_met = io('MERRA2')

    date_from = '2010-01-01'
    date_to = '2019-12-31'

    lai = np.nanmean(ds_lai.read_img('LAI', date_from=date_from, date_to=date_to), axis=0)

    vod = ds_vod.read_img('VOD', date_from=date_from, date_to=date_to)
    invalid = (ds_vod.read_img('Flags', date_from=date_from, date_to=date_to) > 0) | \
              (ds_vod.read_img('RMSE', date_from=date_from, date_to=date_to) > 8) | \
              (ds_vod.read_img('VOD_StdErr', date_from=date_from, date_to=date_to) > 1.2)
    vod[invalid] = np.nan

    vod = np.nanmean(vod, axis=0)

    temp = np.nanmean(ds_met.read_img('T2M', date_from=date_from, date_to=date_to), axis=0)
    prec = np.nanmean(ds_met.read_img('PRECTOTLAND', date_from=date_from, date_to=date_to), axis=0)
    rad = np.nanmean(ds_met.read_img('LWLAND', date_from=date_from, date_to=date_to)+
                     ds_met.read_img('SWLAND', date_from=date_from, date_to=date_to), axis=0)

    temp_ease = np.full(lai.shape, np.nan)
    prec_ease = np.full(lai.shape, np.nan)
    rad_ease = np.full(lai.shape, np.nan)
    temp_ease[ds_lai.lut.row_ease, ds_lai.lut.col_ease] = temp[ds_lai.lut.row_merra, ds_lai.lut.col_merra]
    prec_ease[ds_lai.lut.row_ease, ds_lai.lut.col_ease] = prec[ds_lai.lut.row_merra, ds_lai.lut.col_merra]
    rad_ease[ds_lai.lut.row_ease, ds_lai.lut.col_ease] = rad[ds_lai.lut.row_merra, ds_lai.lut.col_merra]

    # Collocation
    invalid = np.where(np.isnan(vod) | np.isnan(lai) | np.isnan(temp_ease) | np.isnan(prec_ease) | np.isnan(rad_ease) )
    lai[invalid] = np.nan
    vod[invalid] = np.nan
    temp_ease[invalid] = np.nan
    prec_ease[invalid] = np.nan
    rad_ease[invalid] = np.nan

    fontsize = 12

    f = plt.figure(figsize=(18,6))

    # gs = gridspec.GridSpec(3, 3, height_ratios=[2,1,0.025], wspace=0.35, hspace=0.20)

    plt.subplot(1, 5, 1)
    plot_img(ds_lai.lon, ds_lai.lat, lai, cbrange=(0,6), cmap='viridis_r', title='LAI [-]', fontsize=fontsize)

    plt.subplot(1, 5, 2)
    plot_img(ds_lai.lon, ds_lai.lat, vod, cbrange=(0,0.8), cmap='viridis_r', title='VOD [-]', fontsize=fontsize)

    plt.subplot(1, 5, 3)
    plot_img(ds_lai.lon, ds_lai.lat, temp_ease, cbrange=(5, 30), cmap='viridis_r', title='T [$^\circ$C]', fontsize=fontsize)

    plt.subplot(1, 5, 4)
    plot_img(ds_lai.lon, ds_lai.lat, prec_ease * 1e3, cbrange=(0,2), cmap='viridis_r', title='P [kg / m2 / d] * 1e-2', fontsize=fontsize)

    plt.subplot(1, 5, 5)
    plot_img(ds_lai.lon, ds_lai.lat, rad_ease, cbrange=(60,160), cmap='viridis_r', title='Rad. [W / m2]', fontsize=fontsize)

    # plt.tight_layout()
    # plt.show()

    fname = '/Users/u0116961/Documents/work/deforestation_paper/plots/mean_veg_met.png'
    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def calc_lagged_corr():

    fout = Path('/Users/u0116961/Documents/work/deforestation_paper/lagged_corr_w_sif.csv')

    ds_lai = io('LAI')
    ds_vod = io('SMOS_IC')
    ds_met = io('MERRA2')
    ds_sif = io('SIF')

    date_from = '2010-01-01'
    date_to = '2019-12-31'

    for i, val in ds_lai.lut.iterrows():

        print(f'{i} / {len(ds_lai.lut)}')

        lai = ds_lai.read('LAI', i, date_from=date_from, date_to=date_to).dropna()
        if len(lai) == 0:
            continue
        vod = ds_vod.read('VOD', i, date_from=date_from, date_to=date_to)

        if (len(lai)>0) & (len(vod)>0):
            invalid = (ds_vod.read('Flags', i, date_from=date_from, date_to=date_to) > 0) | \
                      (ds_vod.read('RMSE', i, date_from=date_from, date_to=date_to) > 8) | \
                      (ds_vod.read('VOD_StdErr', i, date_from=date_from, date_to=date_to) > 1.2)
            vod[invalid] = np.nan
            vod = vod.dropna()
        if len(vod) == 0:
            continue

        sif = ds_sif.read('sif_dc', i, date_from=date_from, date_to=date_to)
        invalid = (ds_sif.read('n', i, date_from=date_from, date_to=date_to) <= 1) | \
                  (ds_sif.read('cloud_fraction', i, date_from=date_from, date_to=date_to) > 0.7)
        sif[invalid] = np.nan
        sif = sif.dropna()
        if len(sif) == 0:
            continue

        df_veg = pd.concat((lai, vod, sif), axis=1, keys=['LAI', 'VOD', 'SIF']).resample('M').mean().dropna()
        for col in df_veg:
            df_veg[f'{col}_anom'] = calc_anomaly(df_veg[col], method='harmonic', longterm=True, n=3)

        temp = ds_met.read('T2M', i, date_from=date_from, date_to=date_to)
        prec = ds_met.read('PRECTOTLAND', i, date_from=date_from, date_to=date_to)
        rad = ds_met.read('LWLAND', i, date_from=date_from, date_to=date_to) + \
              ds_met.read('SWLAND', i, date_from=date_from, date_to=date_to)
        df_met = pd.concat((temp,prec,rad), axis=1, keys=['T','P','R']).resample('M').mean().dropna()
        if len(df_met) == 0:
            continue

        df_met['T_anom'] = calc_anomaly(df_met['T'], method='harmonic', longterm=True, n=3)
        df_met['P_anom'] = calc_anomaly(df_met['P'], method='harmonic', longterm=True, n=3)
        df_met['R_anom'] = calc_anomaly(df_met['R'], method='harmonic', longterm=True, n=3)

        tmp_df_met = df_met.copy()
        tmp_df_veg = df_veg.reindex(df_met.index).copy()
        tmp_df_veg.columns = tmp_df_veg.columns + '_nolag'
        tmp_df_met = pd.concat((tmp_df_met, tmp_df_veg), axis=1)

        tmp_df_met.index = np.arange(len(df_met))

        res = pd.DataFrame(index=(i,))
        for lag in np.arange(-6,7):
            tmp_df_veg = df_veg.reindex(df_met.index)
            tmp_df_veg.index = np.arange(len(tmp_df_veg)) + lag
            corr = pd.concat((tmp_df_met, tmp_df_veg), axis=1).corr()
            res[f'R_LAI_T_{lag}'] = corr['R']['LAI']
            res[f'R_LAI_P_{lag}'] = corr['T']['LAI']
            res[f'R_LAI_R_{lag}'] = corr['P']['LAI']
            res[f'R_VOD_T_{lag}'] = corr['R']['VOD']
            res[f'R_VOD_P_{lag}'] = corr['T']['VOD']
            res[f'R_VOD_R_{lag}'] = corr['P']['VOD']
            res[f'R_SIF_T_{lag}'] = corr['R']['SIF']
            res[f'R_SIF_P_{lag}'] = corr['T']['SIF']
            res[f'R_SIF_R_{lag}'] = corr['P']['SIF']

            res[f'R_LAI_VOD_{lag}'] = corr['LAI_nolag']['VOD']
            res[f'R_LAI_SIF_{lag}'] = corr['LAI_nolag']['SIF']
            res[f'R_VOD_SIF_{lag}'] = corr['VOD_nolag']['SIF']

            res[f'R_anom_LAI_T_{lag}'] = corr['R_anom']['LAI_anom']
            res[f'R_anom_LAI_P_{lag}'] = corr['T_anom']['LAI_anom']
            res[f'R_anom_LAI_R_{lag}'] = corr['P_anom']['LAI_anom']
            res[f'R_anom_VOD_T_{lag}'] = corr['R_anom']['VOD_anom']
            res[f'R_anom_VOD_P_{lag}'] = corr['T_anom']['VOD_anom']
            res[f'R_anom_VOD_R_{lag}'] = corr['P_anom']['VOD_anom']
            res[f'R_anom_SIF_T_{lag}'] = corr['R_anom']['SIF_anom']
            res[f'R_anom_SIF_P_{lag}'] = corr['T_anom']['SIF_anom']
            res[f'R_anom_SIF_R_{lag}'] = corr['P_anom']['SIF_anom']

            res[f'R_anom_LAI_VOD_{lag}'] = corr['LAI_anom_nolag']['VOD_anom']
            res[f'R_anom_LAI_SIF_{lag}'] = corr['LAI_anom_nolag']['SIF_anom']
            res[f'R_anom_VOD_SIF_{lag}'] = corr['VOD_anom_nolag']['SIF_anom']

        if fout.exists():
            res.to_csv(fout, float_format='%0.4f', mode='a', header=False)
        else:
            res.to_csv(fout, float_format='%0.4f')


def plot_lagged_corr_spatial():

    res = pd.read_csv('/Users/u0116961/Documents/work/deforestation_paper/lagged_corr.csv', index_col=0)

    ds = io('LAI')
    lut = ds.lut.reindex(res.index)

    fontsize = 12

    variables = ['T', 'P', 'R']
    titles = ['temperature', 'precipitation', 'radiation']
    lags = [-5, -4, -3, -2, -1, 0]

    cbr = (-0.3,0.4)

    cmap = cc.cm.bjy

    for met, tit in zip(variables, titles):
        f = plt.figure(figsize=(22, 11))
        plt.suptitle(f'Lagged correlation ({tit}) [-]', fontsize=fontsize+2, y=0.93)

        for i, veg in enumerate(['LAI', 'VOD']):
            for j, lag in enumerate(lags):

                var = f'R_anom_{veg}_{met}_{lag}'

                arr = np.full(ds.lat.shape, np.nan)
                arr[lut.row_ease, lut.col_ease] = res[var]

                plt.subplot(2, len(lags), i*len(lags) + j + 1)

                title = f'lag: {lag} months' if i == 0 else ''
                ylabel = veg if j == 0 else ''

                im, m = plot_img(ds.lon, ds.lat, arr, cbrange=cbr, cmap=cmap, title=title, fontsize=fontsize,
                         plot_cmap=False, return_map=True)
                plt.ylabel(ylabel, fontsize=fontsize)

                x, y = m(-59, -48)
                plt.text(x, y, f'm(+) = {np.nanmean(arr[arr>0]):.2f}', fontsize=fontsize)
                x, y = m(-59, -51)
                plt.text(x, y, f'm(a)  = {np.nanmean(arr):.2f}', fontsize=fontsize)
                x, y = m(-59, -54)
                plt.text(x, y, f'm(-)   = {np.nanmean(arr[arr<0]):.2f}', fontsize=fontsize)

        plot_centered_cbar(f, im, len(lags), fontsize=fontsize, pad=0.02, wdth=0.02, hspace=0.01)


        fname = f'/Users/u0116961/Documents/work/deforestation_paper/plots/lagged_corr_anom_{met}.png'
        f.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

def plot_lagged_corr_spatial_LAI_VOD_SIF():

    res = pd.read_csv('/Users/u0116961/Documents/work/deforestation_paper/lagged_corr_w_sif.csv', index_col=0)

    ds = io('LAI')
    lut = ds.lut.reindex(res.index)

    fontsize = 12

    lags = [-2, -1, 0, 1, 2]

    cbr = (-1,1)

    cmap = cc.cm.bjy

    f = plt.figure(figsize=(20, 12))
    plt.suptitle(f'Lagged correlation (SIF) [-]', fontsize=fontsize+2, y=0.93)

    for i, var in enumerate(['LAI_SIF', 'VOD_SIF']):
        for j, lag in enumerate(lags):

            tag = f'R_{var}_{lag}'

            arr = np.full(ds.lat.shape, np.nan)
            arr[lut.row_ease, lut.col_ease] = res[tag]

            plt.subplot(2, len(lags), i*len(lags) + j + 1)

            title = f'lag: {lag} months' if i == 0 else ''
            ylabel = var.split('_')[0] if j == 0 else ''

            im, m = plot_img(ds.lon, ds.lat, arr, cbrange=cbr, cmap=cmap, title=title, fontsize=fontsize,
                     plot_cmap=False, return_map=True)
            plt.ylabel(ylabel, fontsize=fontsize)

            x, y = m(-59, -48)
            plt.text(x, y, f'm(+) = {np.nanmean(arr[arr>0]):.2f}', fontsize=fontsize)
            x, y = m(-59, -51)
            plt.text(x, y, f'm(a)  = {np.nanmean(arr):.2f}', fontsize=fontsize)
            x, y = m(-59, -54)
            plt.text(x, y, f'm(-)   = {np.nanmean(arr[arr<0]):.2f}', fontsize=fontsize)

    plot_centered_cbar(f, im, len(lags), fontsize=fontsize, pad=0.02, wdth=0.02, hspace=0.01)

    fname = f'/Users/u0116961/Documents/work/deforestation_paper/plots/lagged_corr_SIF.png'
    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def plot_best_corr_spatial_vod():

    res = pd.read_csv('/Users/u0116961/Documents/work/deforestation_paper/lagged_corr.csv', index_col=0)

    ds = io('LAI')
    lut = ds.lut.reindex(res.index)

    fontsize = 12

    tags = ['R_VOD_T', 'R_VOD_P', 'R_VOD_R']
    tags_anom = ['R_anom_VOD_T', 'R_anom_VOD_P', 'R_anom_VOD_R']

    lags = [-3, -3, -2]
    lags_anom = [0, 0, -1]

    titles = ['Temperature', 'Precipitation', 'Radiation']

    cbr = (-0.9,0.9)
    cbr_anom = (-0.4,0.4)

    cmap = cc.cm.bjy

    f = plt.figure(figsize=(10, 10))

    for i, (tag, tag_anom, lag, lag_anom, tit) in enumerate(zip(tags, tags_anom, lags,lags_anom, titles)):

        var = f'{tag}_{lag}'
        var_anom = f'{tag_anom}_{lag_anom}'

        arr = np.full(ds.lat.shape, np.nan)
        arr[lut.row_ease, lut.col_ease] = res[var]

        arr_anom = np.full(ds.lat.shape, np.nan)
        arr_anom[lut.row_ease, lut.col_ease] = res[var_anom]

        plt.subplot(2, 3, i+1)
        im = plot_img(ds.lon, ds.lat, arr, cbrange=cbr, cmap=cmap, title=tit, fontsize=fontsize,
                 plot_cmap=True)
        if i == 0:
            plt.ylabel('Total signal', fontsize=fontsize)

        plt.subplot(2, 3, i+4)
        if i == 1:
            cm = cc.cm.bjy_r
        else:
            cm = cmap
        im = plot_img(ds.lon, ds.lat, arr_anom, cbrange=cbr_anom, cmap=cm, title='', fontsize=fontsize,
                 plot_cmap=True)
        if i == 0:
            plt.ylabel('Anomalies', fontsize=fontsize)

    fname = f'/Users/u0116961/Documents/work/deforestation_paper/plots/best_correlations.png'
    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def plot_lagged_corr_boxplot():

    res = pd.read_csv('/Users/u0116961/Documents/work/deforestation_paper/lagged_corr_w_sif.csv', index_col=0)

    ds = io('LAI')
    lut = ds.lut.reindex(res.index)

    variables = ['T', 'P', 'R']
    titles = ['Temperature', 'Precipitation', 'Radiation']
    datasets = ['LAI', 'SIF', 'VOD']

    lags = np.arange(-6, 1, 1)

    pos = [i + j for i in np.arange(1, len(variables) + 1) for j in np.linspace(-0.4, 0.4, len(lags))]
    colors = [f'{s}' for n in np.arange(len(variables)) for s in np.linspace(0.2, 0.98, len(lags))]
    # colors = [s for n in np.arange(len(variables)) for s in ['lightblue', 'lightgreen', 'coral']]

    f = plt.figure(figsize=(13, 14))
    fontsize = 14

    for n, d in enumerate(datasets):
        ax = plt.subplot(3, 1, n+1)
        if n == 0:
            axpos = ax.get_position()

        data = list()

        for v, t in zip(variables, titles):
            for lag in lags:
                var = f'R_anom_{d}_{v}_{lag}'
                tmp = res[var].values
                # tmp = tmp[tmp>0]
                data.append(tmp[~np.isnan(tmp)])

        box = ax.boxplot(data, whis=[10, 90], showfliers=False, positions=pos, widths=0.07, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set(color='black', linewidth=2)
            patch.set_facecolor(color)
        for patch in box['medians']:
            patch.set(color='black', linewidth=2)
        for patch in box['whiskers']:
            patch.set(color='black', linewidth=1)

        plt.xlim(0.5, len(variables)+0.5)
        if n < 2:
            plt.xticks([], [], fontsize=fontsize)
        else:
            plt.xticks(np.arange(1,len(variables)+1), titles , fontsize=fontsize)
        plt.ylim(-0.5,0.5)
        plt.yticks(np.arange(-0.5,0.5,0.2))
        # plt.ylim(0,0.9)
        # plt.yticks(np.arange(0,1,0.2))
        plt.axhline(color='k', linestyle='--', linewidth=1)

        for i in np.arange(1,len(variables)):
            plt.axvline(i + 0.5, linewidth=1.5, color='k')

        plt.ylabel(d, fontsize=fontsize)

        if n==0:
            plt.title('Lagged correlation [-] (anomalies)',fontsize=fontsize)

    f.subplots_adjust(hspace=0.1)

    plt.figlegend((box['boxes'][0:len(lags)]), lags, 'upper right', title='Lag [m]',
                  bbox_to_anchor=(axpos.x1+0.09,axpos.y1+0.013), fontsize=fontsize-2)

    fname = f'/Users/u0116961/Documents/work/deforestation_paper/plots/lagged_corr_anom_boxplot_w_sif.png'
    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    # plt.tight_layout()
    # plt.show()

def normalize(Ser):
   return (Ser - Ser.mean()) / Ser.std()

def calc_trends():

    fout = Path('/Users/u0116961/Documents/work/deforestation_paper/trends.csv')

    ds_lai = io('LAI')
    ds_vod = io('SMOS_IC')
    ds_met = io('MERRA2')

    date_from = '2010-01-01'
    date_to = '2019-12-31'

    for i, val in ds_lai.lut.iterrows():

        print(f'{i} / {len(ds_lai.lut)}')

        lai = ds_lai.read('LAI', i, date_from=date_from, date_to=date_to).dropna()
        if len(lai) == 0:
            continue

        vod = ds_vod.read('VOD', i, date_from=date_from, date_to=date_to)
        if len(vod) > 0:
            invalid = (ds_vod.read('Flags', i, date_from=date_from, date_to=date_to) > 0) | \
                      (ds_vod.read('RMSE', i, date_from=date_from, date_to=date_to) > 8) | \
                      (ds_vod.read('VOD_StdErr', i, date_from=date_from, date_to=date_to) > 1.2)
            vod[invalid] = np.nan
            vod = vod.dropna()
        if len(vod) == 0:
            continue

        temp = ds_met.read('T2M', i, date_from=date_from, date_to=date_to)
        prec = ds_met.read('PRECTOTLAND', i, date_from=date_from, date_to=date_to)
        rad = ds_met.read('LWLAND', i, date_from=date_from, date_to=date_to) + \
              ds_met.read('SWLAND', i, date_from=date_from, date_to=date_to)

        if len(temp)+len(prec)+len(rad) == 0:
            continue

        res = pd.DataFrame(index=(i,))

        mk_lai = mk.original_test(normalize(lai.resample('M').mean()))
        mk_vod = mk.original_test(normalize(vod.resample('M').mean()))
        mk_rad = mk.original_test(normalize(rad.resample('M').mean()))
        mk_temp = mk.original_test(normalize(temp.resample('M').mean()))
        mk_prec = mk.original_test(normalize(prec.resample('M').sum()))

        mk_vod_orig = mk.original_test(vod.resample('M').mean())

        res[f'slope_lai'] = mk_lai.slope
        res[f'slope_vod'] = mk_vod.slope
        res[f'slope_rad'] = mk_rad.slope
        res[f'slope_temp'] = mk_temp.slope
        res[f'slope_prec'] = mk_prec.slope
        res[f'slope_vod_orig'] = mk_vod_orig.slope


        res[f'p_lai'] = mk_lai.p
        res[f'p_vod'] = mk_vod.p
        res[f'p_rad'] = mk_rad.p
        res[f'p_temp'] = mk_temp.p
        res[f'p_prec'] = mk_prec.p
        res[f'p_vod_orig'] = mk_vod_orig.p

        if fout.exists():
            res.to_csv(fout, float_format='%0.8f', mode='a', header=False)
        else:
            res.to_csv(fout, float_format='%0.8f')

def plot_trends():

    res = pd.read_csv('/Users/u0116961/Documents/work/deforestation_paper/trends.csv', index_col=0)

    ds = io('LAI')
    lut = ds.lut.reindex(res.index)

    fontsize = 12

    variables = ['lai', 'vod', 'temp', 'prec', 'rad']
    titles = ['LAI [yr$^{-1}$]', 'VOD [yr$^{-1}$]', 'T [$^\circ$C / yr]', 'P [kg / m2 / mo / yr]', 'Rad. [W / m2 / yr]']

    cmap = cc.cm.bjy
    cbr = [-0.015, 0.015]

    f = plt.figure(figsize=(18, 6))

    for n, (variable, tit) in enumerate(zip(variables, titles)):

        tmp =  res[f'slope_{variable}']
        tmp[res[f'p_{variable}'] > 0.05] = np.nan

        arr = np.full(ds.lat.shape, np.nan)
        arr[lut.row_ease, lut.col_ease] = tmp

        plt.subplot(1, 5, n+1)

        im = plot_img(ds.lon, ds.lat, arr, cbrange=cbr, cmap=cmap, title=tit, fontsize=fontsize,
                      plot_cmap=True)

    fname = f'/Users/u0116961/Documents/work/deforestation_paper/plots/trends_sig.png'
    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def plot_trend_tcl():

    res = pd.read_csv('/Users/u0116961/Documents/work/deforestation_paper/trends.csv', index_col=0)

    ds_vod = io('SMOS_IC')
    ds_tcl = io('TCL')
    ds_agb = io('AGB')
    lut = ds_vod.lut.reindex(res.index)

    agb = ds_agb.read_img('AGB')
    agb_err = ds_agb.read_img('AGB_err')
    agb[agb_err > 160] = np.nan

    date_from = '2010-01-01'
    date_to = '2010-12-31'
    vod = ds_vod.read_img('VOD', date_from=date_from, date_to=date_to)
    invalid = (ds_vod.read_img('Flags', date_from=date_from, date_to=date_to) > 0) | \
              (ds_vod.read_img('RMSE', date_from=date_from, date_to=date_to) > 8) | \
              (ds_vod.read_img('VOD_StdErr', date_from=date_from, date_to=date_to) > 1.2)
    vod[invalid] = np.nan
    vod = np.nanmean(vod, axis=0)

    invalid = np.where(np.isnan(agb) | np.isnan(vod))
    agb[invalid] = np.nan
    vod[invalid] = np.nan
    valid = np.where(~np.isnan(agb))

    p = np.poly1d(np.polyfit(vod[valid], agb[valid], 1))

    fontsize = 16

    f = plt.figure(figsize=(18, 10))

    tmp =  res[f'slope_vod_orig']
    tmp[res[f'p_vod_orig'] > 0.05] = np.nan
    tmp[tmp > 0] = np.nan

    arr = np.full(ds_vod.lat.shape, np.nan)
    arr[lut.row_ease, lut.col_ease] = tmp


    plt.subplot(1, 3, 1)
    im = plot_img(ds_vod.lon, ds_vod.lat, arr*12, cbrange=[-0.01,0], cmap=truncate_colormap(cc.cm.bjy, 0, 0.5), title='VOD trend [yr$^{-1}$]', fontsize=fontsize,
                  plot_cmap=True)

    plt.subplot(1, 3, 2)
    im = plot_img(ds_vod.lon, ds_vod.lat, p[0]*arr*(-12), cbrange=[-0.3,0], cmap=truncate_colormap(cc.cm.bjy, 0, 0.5), title='AGB trend [Mg/ha/yr]', fontsize=fontsize,
                  plot_cmap=True)

    plt.subplot(1, 3, 3)
    tcl = ds_tcl.read_img('TCL')
    tcl[tcl < 0.07] = np.nan
    im = plot_img(ds_tcl.lon, ds_tcl.lat, tcl, cbrange=[0.1,0.4], cmap=truncate_colormap(cc.cm.bjy_r, 0.5, 1), title='Tree cover loss [-]', fontsize=fontsize,
                  plot_cmap=True)

    fname = f'/Users/u0116961/Documents/work/deforestation_paper/plots/vod_trend_tcl.png'
    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mask():

    res = pd.read_csv('/Users/u0116961/Documents/work/deforestation_paper/lagged_corr.csv', index_col=0)

    ds = io('LAI')
    lut = ds.lut.reindex(res.index)

    fontsize = 16

    t = np.full(ds.lat.shape, np.nan)
    p = np.full(ds.lat.shape, np.nan)
    r = np.full(ds.lat.shape, np.nan)
    t[lut.row_ease, lut.col_ease] = res[f'R_VOD_T_-3']
    p[lut.row_ease, lut.col_ease] = res[f'R_VOD_P_-3'] # possibly -2
    r[lut.row_ease, lut.col_ease] = res[f'R_VOD_R_-2']

    mask = np.full(ds.lat.shape, np.nan)
    # mask[lut.row_ease, lut.col_ease] = 0
    mask[(t < 0.5) & (p < 0.5) & (r < 0.5)] = 1

    f = plt.figure(figsize=(6,10))
    plot_img(ds.lon, ds.lat, mask, cbrange=None, cmap='viridis', title='Correlation mask', fontsize=fontsize,
             plot_cmap=False)

    plt.tight_layout()
    plt.show()


    # fname = f'/Users/u0116961/Documents/work/deforestation_paper/plots/mask.png'
    # f.savefig(fname, dpi=300, bbox_inches='tight')
    # plt.close()


def calc_trend_breakpoints():

    fout = Path('/Users/u0116961/Documents/work/deforestation_paper/breakpoints.csv')

    ds_lai = io('LAI')
    ds_vod = io('SMOS_IC')

    date_from = '2010-01-01'
    date_to = '2019-12-31'

    x0s = np.arange(2010, 2020)
    rss_cols = [f'RSS_LAI_{x0}' for x0 in x0s] + [f'RSS_VOD_{x0}' for x0 in x0s] + \
               ['bfast_lai_b1', 'bfast_lai_b2', 'bfast_lai_m1', 'bfast_lai_m2' ] + \
               ['bfast_vod_b1', 'bfast_vod_b2', 'bfast_vod_m1', 'bfast_vod_m2' ]

    start_date = datetime(2012, 1, 1)
    bfast = BFASTMonitor(
        start_date,
        freq=365,
        k=3,
        hfrac=0.25,
        trend=False,
        level=0.05,
        verbose=0,
        backend='python',
        device_id=0,
    )

    for i, _ in ds_lai.lut.iterrows():
        print(f'{i} / {len(ds_lai.lut)}')

        lai = ds_lai.read('LAI', i, date_from=date_from, date_to=date_to).dropna()

        vod = ds_vod.read('VOD', i, date_from=date_from, date_to=date_to)
        if len(vod) > 0:
            invalid = (ds_vod.read('Flags', i, date_from=date_from, date_to=date_to) > 0) | \
                      (ds_vod.read('RMSE', i, date_from=date_from, date_to=date_to) > 8) | \
                      (ds_vod.read('VOD_StdErr', i, date_from=date_from, date_to=date_to) > 1.2)
            vod[invalid] = np.nan
            vod = vod.dropna()

        if (len(lai) < 1) | (len(vod) < 1):
            continue

        mlai = pd.DataFrame(normalize(lai.resample('M').mean()), columns=['LAI'])
        mvod = pd.DataFrame(normalize(vod.resample('M').mean()), columns=['VOD'])

        rss = pd.DataFrame(columns=rss_cols, index=(i,))

        data = np.reshape(mvod.values, (len(mvod), 1, 1))
        dates = mvod.index.to_pydatetime()
        bfast.fit(data, dates, n_chunks=5, nan_value=-32768)
        rss.loc[i,'bfast_vod_m1'] = bfast.magnitudes[0][0]
        if bfast.breaks[0][0] >= 0:
            dt = dates[dates >= datetime(2011, 1, 1)][bfast.breaks[0][0]]
            rss.loc[i,'bfast_vod_b1'] = dt.year + dt.month/12
        if len(bfast.breaks) > 1:
            dt = dates[dates > datetime(2011, 1, 1)][bfast.breaks.flatten()[1]]
            rss.loc[i,'bfast_vod_b2'] = dt.year + dt.month/12
            rss.loc[i,'bfast_vod_m2'] = bfast.magnitudes.flatten()[1]

        data = np.reshape(mlai.values, (len(mlai), 1, 1))
        dates = mlai.index.to_pydatetime()
        bfast.fit(data, dates, n_chunks=None, nan_value=-32768)
        rss.loc[i,'bfast_lai_m1'] = bfast.magnitudes[0][0]
        if bfast.breaks[0][0] >= 0:
            dt = dates[dates >= start_date][bfast.breaks[0][0]]
            rss.loc[i,'bfast_lai_b1'] = dt.year + dt.month/12
        # if len(bfast.breaks) > 1:
        #     dt = dates[dates > datetime(2011, 1, 1)][bfast.breaks.flatten()[1]]
        #     rss.loc[i,'bfast_lai_b2'] = dt.year + dt.month/12
        #     rss.loc[i,'bfast_lai_m2'] = bfast.magnitudes.flatten()[1]

        for x0 in x0s:

            mlai[f'LAI_trend_{x0}'] = np.nan
            mvod[f'VOD_trend_{x0}'] = np.nan

            lai_l = mlai[date_from:f'{x0}']['LAI']
            lai_u = mlai[f'{x0 + 1}':date_to]['LAI']

            vod_l = mvod[date_from:f'{x0}']['VOD']
            vod_u = mvod[f'{x0 + 1}':date_to]['VOD']

            for lai, vod in zip([lai_l, lai_u], [vod_l, vod_u]):

                try:
                    mk_lai = mk.original_test(lai)
                    mk_vod = mk.original_test(vod)

                    x = (lai.index.year - lai.index.year.min()) * 12 + lai.index.month
                    mlai.loc[lai.index, f'LAI_trend_{x0}'] = mk_lai.intercept + mk_lai.slope * x.values

                    x = (vod.index.year - vod.index.year.min()) * 12 + vod.index.month
                    mvod.loc[vod.index, f'VOD_trend_{x0}'] = mk_vod.intercept + mk_vod.slope * x.values

                except:
                    continue

            rss.loc[i, f'RSS_LAI_{x0}'] = ((mlai[f'LAI'] - mlai[f'LAI_trend_{x0}']) ** 2).sum()
            rss.loc[i, f'RSS_VOD_{x0}'] = ((mvod[f'VOD'] - mvod[f'VOD_trend_{x0}']) ** 2).sum()

        if fout.exists():
            rss.to_csv(fout, float_format='%0.4f', mode='a', header=False)
        else:
            rss.to_csv(fout, float_format='%0.4f')

def plot_breakpoints():

    res = pd.read_csv('/Users/u0116961/Documents/work/deforestation_paper/breakpoints.csv', index_col=0)

    ds = io('LAI')
    lut = ds.lut.reindex(res.index)

    fontsize = 15

    variables = ['LAI', 'VOD']

    cmap_d = colors.ListedColormap(sns.color_palette("viridis", n_colors=10))
    cmap_d2 = colors.ListedColormap(sns.color_palette('viridis', n_colors=11))

    # cmap_d = colors.ListedColormap(cm.get_cmap('jet')(np.linspace(0, 1, 10)))
    # jet_12_colors = jet(np.linspace(0, 1, 10))

    cbr = [2010, 2020]

    f = plt.figure(figsize=(24, 10))

    for n, variable in enumerate(variables):

        cols = [col for col in res.columns.values if f'RSS_{variable}' in col]
        year = [int(col.split('_')[-1]) for col in cols]

        res[f'bp_{variable}'] = -1
        min_rss = res[cols].min(axis=1)
        for col, yr in zip(cols, year):
            res[f'bp_{variable}'][res[col] == min_rss] = yr+1

        arr = np.full(ds.lat.shape, np.nan)
        arr[lut.row_ease, lut.col_ease] = res[f'bp_{variable}']
        plt.subplot(1, 4, 2*n+1)
        im = plot_img(ds.lon, ds.lat, arr, cbrange=cbr, cmap=cmap_d, title=f'{variable} (RSS)', fontsize=fontsize,
                      plot_cmap=True)

        arr = np.full(ds.lat.shape, np.nan)
        arr[lut.row_ease, lut.col_ease] = res[f'bfast_{variable.lower()}_b1']
        plt.subplot(1, 4, 2*n+2)
        im = plot_img(ds.lon, ds.lat, arr, cbrange=cbr, cmap=cmap_d, title=f'{variable} (BFAST)', fontsize=fontsize,
                      plot_cmap=True)

    # plt.subplot(1, 3, 3)
    # arr = np.full(ds.lat.shape, np.nan)
    # arr[lut.row_ease, lut.col_ease] = res[f'bp_VOD'] - res[f'bp_LAI']
    # im = plot_img(ds.lon, ds.lat, arr, cbrange=[-5, 5], cmap=cmap_d2, title='VOD - LAI', fontsize=fontsize,
    #               plot_cmap=True)

    # fname = f'/Users/u0116961/Documents/work/deforestation_paper/plots/breakpoints.png'
    # f.savefig(fname, dpi=300, bbox_inches='tight')
    # plt.close()

    plt.tight_layout()
    plt.show()

def plot_bp_rss_stats():

    res = pd.read_csv('/Users/u0116961/Documents/work/deforestation_paper/breakpoints.csv', index_col=0)

    ds = io('LAI')
    lut = ds.lut.reindex(res.index)

    fontsize = 15

    variables = ['LAI', 'VOD']

    f = plt.figure(figsize=(12, 10))

    for n, variable in enumerate(variables):

        cols = [col for col in res.columns.values if variable in col]

        arr = np.full(ds.lat.shape, np.nan)
        arr[lut.row_ease, lut.col_ease] = res[cols].max(axis=1) - res[cols].min(axis=1)

        plt.subplot(1, 2, n+1)

        im = plot_img(ds.lon, ds.lat, arr, cbrange=(0, 25), cmap='viridis', title=variable, fontsize=fontsize,
                      plot_cmap=True)

    fname = f'/Users/u0116961/Documents/work/deforestation_paper/plots/rss_range.png'
    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    # plt.tight_layout()
    # plt.show()


def plot_trend_ts():

    ds_lai = io('LAI')
    ds_vod = io('SMOS_IC')

    date_from = '2010-01-01'
    date_to = '2019-12-31'

    lat, lon = -22.87578546915353, -62.311473240766794

    i = np.argmin((ds_lai.lut.lat - lat) ** 2 + (ds_lai.lut.lon - lon) ** 2)
    print(i)
    # i = 3000

    lai = ds_lai.read('LAI', i, date_from=date_from, date_to=date_to).dropna()

    vod = ds_vod.read('VOD', i, date_from=date_from, date_to=date_to)
    if len(vod) > 0:
        invalid = (ds_vod.read('Flags', i, date_from=date_from, date_to=date_to) > 0) | \
                  (ds_vod.read('RMSE', i, date_from=date_from, date_to=date_to) > 8) | \
                  (ds_vod.read('VOD_StdErr', i, date_from=date_from, date_to=date_to) > 1.2)
        vod[invalid] = np.nan
        vod = vod.dropna()

    mlai = pd.DataFrame(normalize(lai.resample('M').mean()), columns=['LAI'])
    mvod = pd.DataFrame(normalize(vod.resample('M').mean()), columns=['VOD'])

    x0s = np.arange(2010, 2020)


    rss_lai = pd.DataFrame(columns=['RSS_LAI'], index=x0s)
    rss_vod = pd.DataFrame(columns=['RSS_VOD'], index=x0s)

    for x0 in x0s:

        mlai[f'LAI_trend_{x0}'] = np.nan
        mvod[f'VOD_trend_{x0}'] = np.nan

        lai_l = mlai[date_from:f'{x0}']['LAI']
        lai_u = mlai[f'{x0 + 1}':date_to]['LAI']

        vod_l = mvod[date_from:f'{x0}']['VOD']
        vod_u = mvod[f'{x0 + 1}':date_to]['VOD']

        for lai, vod in zip([lai_l, lai_u], [vod_l, vod_u]):

            try:
                mk_lai = mk.original_test(lai)
                mk_vod = mk.original_test(vod)

                x = (lai.index.year - lai.index.year.min()) * 12 + lai.index.month
                mlai.loc[lai.index, f'LAI_trend_{x0}'] = mk_lai.intercept + mk_lai.slope * x.values

                x = (vod.index.year - vod.index.year.min()) * 12 + vod.index.month
                mvod.loc[vod.index, f'VOD_trend_{x0}'] = mk_vod.intercept + mk_vod.slope * x.values

            except:
                continue

        rss_lai.loc[x0, 'RSS_LAI'] = ((mlai[f'LAI'] - mlai[f'LAI_trend_{x0}']) ** 2).sum()
        rss_vod.loc[x0, 'RSS_VOD'] = ((mvod[f'VOD'] - mvod[f'VOD_trend_{x0}']) ** 2).sum()

    f = plt.figure(figsize=(16, 10))
    ax1 = plt.subplot(2, 2, 1)
    mlai[['LAI', f'LAI_trend_201{np.argmin(rss_lai)}', 'LAI_trend_2019']].plot(ax=ax1, legend=False)
    # plt.ylim(mlai['LAI'].min(), mlai['LAI'].max())

    ax2 = plt.subplot(2, 2, 2)
    mvod[['VOD', f'VOD_trend_201{np.argmin(rss_vod)}', 'VOD_trend_2019']].plot(ax=ax2, legend=False)
    # plt.ylim(mvod['VOD'].min(), mvod['VOD'].max())

    ax3 = plt.subplot(2, 2, 3)
    rss_lai.plot(ax=ax3)
    plt.xlim(2010, 2020)

    ax4 = plt.subplot(2, 2, 4)
    rss_vod.plot(ax=ax4)
    # plt.xlim(mvod.index.min(), mvod.index.max())
    plt.xlim(2010, 2020)
    # plt.ylim(0,100)

    plt.tight_layout()
    plt.show()

def calc_pca_classes():

    ds_met = io('MERRA2')

    date_from = '2010-01-01'
    date_to = '2019-12-31'

    temp = np.nanpercentile(ds_met.read_img('T2M', date_from=date_from, date_to=date_to), [2, 25, 50, 75, 98], axis=0).reshape((5,-1))
    prec = np.nanpercentile(ds_met.read_img('PRECTOTLAND', date_from=date_from, date_to=date_to), [2, 25, 50, 75, 98], axis=0).reshape((5,-1))
    rad = np.nanpercentile(ds_met.read_img('LWLAND', date_from=date_from, date_to=date_to) +
                           ds_met.read_img('SWLAND', date_from=date_from, date_to=date_to), [2, 25, 50, 75, 98], axis=0).reshape((5,-1))

    df = pd.DataFrame({'t_med': temp[2,:],
                       'p_med': prec[2,:],
                       'r_med': rad[2,:],
                       't_iqr': temp[3, :] - temp[1, :],
                       'p_iqr': prec[3, :] - prec[1, :],
                       'r_iqr': rad[3, :] - rad[1, :],
                       't_rng': temp[4, :] - temp[0, :],
                       'p_rng': prec[4, :] - prec[0, :],
                       'r_rng': rad[4, :] - rad[0, :]})

    df = df.iloc[:,0:6]
    df_nonan = df.dropna()

    pca = PCA(n_components=len(df_nonan.columns))

    df_nonan.loc[:,:] = pca.fit_transform(StandardScaler().fit_transform(df_nonan)).copy()
    df_nonan.columns = np.arange(len(df_nonan.columns))+1

    n_cl = 3

    tmp = pd.concat((df_nonan.iloc[:,0:n_cl], df_nonan.iloc[:,0:n_cl]*-1), axis=1)
    tmp.columns = np.arange(len(tmp.columns))+1
    cl = tmp.idxmax(axis=1).reindex(df.index)
    cl.name = 'PCA_class'
    cl.to_csv(f'/Users/u0116961/Documents/work/deforestation_paper/PCA_classes.csv')

    # ----- PLOT CLASSIFICATION  -----

    f = plt.figure(figsize=(6,8))
    cm = colors.ListedColormap(sns.color_palette('muted',n_colors=n_cl*2))
    # plt.subplot(2, 6, 7+n_cl+1)
    arr = cl.values.reshape(ds_met.lat.shape)
    im, cb = plot_img(ds_met.lon, ds_met.lat, arr, cbrange=[1, n_cl*2+1], cmap=cm, title='PCA classes', fontsize=16,
                  plot_cmap=True, return_cbar=True)
    cb.set_ticks(np.arange(6) + 1.5)
    cb.set_ticklabels(['1(+)', '2(+)', '3(+)', '1(-)', '2(-)', '3(-)'])

    fname = f'/Users/u0116961/Documents/work/deforestation_paper/plots/PCA_classes.png'
    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


    df_pca = df_nonan.reindex(df.index)
    df_pca.columns = [f'PCA-{i}' for i in np.arange(len(df_nonan.columns))+1]

    f = plt.figure(figsize=(8,6))
    exp_var = [pca.explained_variance_ratio_[0:i+1].sum() for i in range(len(pca.explained_variance_ratio_))]
    print(pd.DataFrame(pca.components_,index=df_pca.columns, columns=df.columns.values).T)
    plt.plot(np.arange(len(df_nonan.columns))+1, exp_var)
    plt.plot([0,3],[exp_var[2], exp_var[2] ], color='black', linestyle='--', linewidth='1')
    plt.xticks(np.arange(len(df_nonan.columns))+1, df_pca.columns.values)
    plt.ylim(0.55,1.02)
    plt.xlim(0.8,6.2)
    plt.title('Cumulative fraction of variance explained')

    fname = f'/Users/u0116961/Documents/work/deforestation_paper/plots/PCA_var_explained.png'
    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


    # ----- PLOT VARIABLE CONTRIBUTIONS -----

    f, axes = plt.subplots(1, len(df_pca.columns), sharex=True, sharey=True, figsize=(12, 5))
    variables = ['T (median)', 'P (median)', 'R (median)', 'T (IQR)', 'P (IQR)', 'R (IQR)']
    n_vars = len(variables)

    for i, ax in enumerate(axes):
        ax.barh(np.arange(n_vars), (pca.components_[i, :][::-1]), orientation="horizontal", color="teal")
        ax.set_yticks(np.arange(n_vars))
        ax.set_title(f"PCA-{i + 1}", fontsize=12)
        ax.set_xlim(-0.8,0.8)
        ax.axvline(color='black', linestyle='--', linewidth=1)
        ax.grid(c="lightgrey", linewidth=0.5, which="major")
        ax.grid(c="lightgrey", linewidth=0.5, which="minor")

    axes[0].set_yticklabels(variables[::-1])

    fname = f'/Users/u0116961/Documents/work/deforestation_paper/plots/PCA_contribution.png'
    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    # ----- PLOT VARS  -----

    f = plt.figure(figsize=(25,12))
    fontsize = 18

    for i, (col, tit) in enumerate(zip(df,variables)):
        if i > 5:
            continue

        plt.subplot(2, 6, i+1)
        arr = df[col].values.reshape(ds_met.lat.shape)
        im = plot_img(ds_met.lon, ds_met.lat, arr, cbrange=None, cmap='viridis', title=tit, fontsize=fontsize,
                      plot_cmap=True)


    # ----- PLOT FIRST FEW PCAS  -----

    for i, col in enumerate(df_pca):
        # if i > n_cl:
        #     continue

        plt.subplot(2, 6, i+7)
        arr = df_pca[col].values.reshape(ds_met.lat.shape)
        im = plot_img(ds_met.lon, ds_met.lat, arr, cbrange=[-2,2], cmap=cc.cm.bjy, title=col, fontsize=fontsize,
                      plot_cmap=True)

    fname = f'/Users/u0116961/Documents/work/deforestation_paper/plots/PCA_maps.png'
    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()



if __name__=='__main__':
    # AGB_VOD_LAI()
    # mean_veg_met()
    # calc_lagged_corr()
    # plot_lagged_corr_spatial()
    # plot_lagged_corr_spatial_LAI_VOD_SIF()
    # plot_lagged_corr_boxplot()
    # plot_best_corr_spatial_vod()
    # calc_trends()
    # plot_trends()
    # plot_mask()
    # plot_trend_tcl()
    # calc_trend_breakpoints()
    # plot_breakpoints()
    # plot_bp_rss_stats()
    # plot_trend_ts()
    calc_pca_classes()
    pass