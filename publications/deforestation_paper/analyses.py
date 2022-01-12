
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd

from math import floor
from pathlib import Path

from mpl_toolkits.basemap import Basemap
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import colorcet as cc
sns.set_context('talk', font_scale=0.8)

import pymannkendall as mk

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
              return_map=False):

    img_masked = np.ma.masked_invalid(data)

    m = Basemap(projection='mill',
                llcrnrlat=llcrnrlat,
                urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,
                urcrnrlon=urcrnrlon,
                resolution='c')

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

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

    ds_agb = io('AGB')
    ds_vod = io('SMOS_IC')
    ds_lai = io('LAI')
    ds_tcl = io('TCL')

    trends = pd.read_csv('/Users/u0116961/Documents/work/deforestation_paper/trends.csv', index_col=0)
    lut = ds_agb.lut.reindex(trends.index)

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

    # Collocation
    invalid = np.where(np.isnan(agb) | np.isnan(vod2010) | np.isnan(vod2019) | np.isnan(lai))
    agb[invalid] = np.nan
    lai[invalid] = np.nan
    vod2010[invalid] = np.nan
    vod2019[invalid] = np.nan
    tcl[invalid] = np.nan
    valid = np.where(~np.isnan(agb))

    fontsize = 10

    f = plt.figure(figsize=(13,11))

    gs = gridspec.GridSpec(3, 3, height_ratios=[2,1,0.025], wspace=0.35, hspace=0.20)

    plt.subplot(gs[0,0])
    plot_img(ds_agb.lon, ds_agb.lat, agb, cbrange=(0,100), cmap='viridis_r', title='AGB [Mg/ha]', fontsize=fontsize)

    plt.subplot(gs[0,1])
    plot_img(ds_agb.lon, ds_agb.lat, lai, cbrange=(0,5), cmap='viridis_r', title='LAI [-]', fontsize=fontsize)

    plt.subplot(gs[0,2])
    plot_img(ds_agb.lon, ds_agb.lat, vod2010, cbrange=(0,0.8), cmap='viridis_r', title='VOD [-]', fontsize=fontsize)

    ax = plt.subplot(gs[1,0])
    x, y = agb[valid], lai[valid]
    ax.hexbin(x, y,
              gridsize=35, bins='log',
              cmap='viridis', mincnt=1)
    xs = np.linspace(x.min(),x.max(), 100)
    p = np.poly1d(np.polyfit(x, y, 3))
    plt.plot(xs, p(xs), linestyle='-.', linewidth=1, color='k')
    corr = np.corrcoef(x, y)[0,1]
    plt.title(f'R = {corr:.3f}', fontsize=fontsize)
    plt.xlabel('AGB [Mg/ha]', fontsize=fontsize)
    plt.ylabel('LAI [-]', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    ax = plt.subplot(gs[1,1])
    x, y = agb[valid], vod2010[valid]
    ax.hexbin(x, y,
              gridsize=35, bins='log',
              cmap='viridis', mincnt=1)
    xs = np.linspace(x.min(),x.max(), 100)
    p = np.poly1d(np.polyfit(x, y, 3))
    plt.plot(xs, p(xs), linestyle='-.', linewidth=1, color='k')
    corr = np.corrcoef(x, y)[0,1]
    plt.title(f'R = {corr:.3f}', fontsize=fontsize)
    plt.xlabel('AGB [Mg/ha]', fontsize=fontsize)
    plt.ylabel('VOD [-]', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    ax = plt.subplot(gs[1,2])
    x, y = lai[valid], vod2010[valid]
    ax.hexbin(x, y,
              gridsize=35, bins='log',
              cmap='viridis', mincnt=1)
    xs = np.linspace(x.min(),x.max(), 100)
    p = np.poly1d(np.polyfit(x, y, 3))
    plt.plot(xs, p(xs), linestyle='-.', linewidth=1, color='k')
    corr = np.corrcoef(x, y)[0,1]
    plt.title(f'R = {corr:.3f}', fontsize=fontsize)
    plt.xlabel('LAI [-]', fontsize=fontsize)
    plt.ylabel('VOD [-]', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    fname = '/Users/u0116961/Documents/work/deforestation_paper/plots/AGB_LAI_VOD.png'
    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


    sig = np.full(ds_agb.lat.shape, 1.)
    sig[lut[trends[f'p_vod'] > 0.05].row_ease, lut[trends[f'p_vod'] > 0.05].col_ease] = np.nan

    f = plt.figure(figsize=(12,10))

    plt.subplot(1, 2, 1)
    p = np.poly1d(np.polyfit(vod2010[valid], agb[valid], 3))
    diff = p(vod2019) - p(vod2010)

    diff[np.isnan(mask)] = np.nan
    tcl[np.isnan(mask)] = np.nan
    # tcl[diff<0.05] = np.nan
    # diff[tcl<0.05] = np.nan

    plot_img(ds_agb.lon, ds_agb.lat, diff / 9, cbrange=(-4,4), cmap=cc.cm.bjy,
             title='$\Delta$ AGB [Mg/ha/yr] (2019 - 2010)', fontsize=fontsize + 4)

    plt.subplot(1, 2, 2)


    plot_img(ds_agb.lon, ds_agb.lat, tcl, cbrange=(0 , 0.4), cmap=truncate_colormap(cc.cm.bjy_r,0.5,1), title='Tree cover loss [-] (2019 - 2010)', fontsize=fontsize+4)

    valid = np.where(~np.isnan(diff))
    print(np.corrcoef(diff[valid], tcl[valid]))

    fname = '/Users/u0116961/Documents/work/deforestation_paper/plots/AGB_change_2019_2010_corr_masked.png'
    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    # f = plt.figure(figsize=(12,10))
    #
    # fontsize += 6
    # x, y = diff[valid], tcl[valid]
    # plt.hexbin(x, y,
    #           gridsize=100, bins='log',
    #           cmap='viridis', mincnt=1, )
    # # xs = np.linspace(x.min(),x.max(), 100)
    # # p = np.poly1d(np.polyfit(x, y, 1))
    # # plt.plot(xs, p(xs), linestyle='-.', linewidth=1, color='k')
    # plt.axvline(color='k', linestyle='--', linewidth=1.5)
    # corr = np.corrcoef(x, y)[0,1]
    # plt.title(f'R = {corr:.3f}', fontsize=fontsize)
    # plt.xlabel('$\Delta$ AGB', fontsize=fontsize)
    # plt.ylabel('Tree cover loss', fontsize=fontsize)
    # plt.xticks(fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    # plt.xlim([-50,50])
    # plt.ylim(0,0.8)
    #
    # fname = '/Users/u0116961/Documents/work/deforestation_paper/plots/AGB_diff_vs_TCL.png'
    # f.savefig(fname, dpi=300, bbox_inches='tight')
    # plt.close()


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

    fout = Path('/Users/u0116961/Documents/work/deforestation_paper/lagged_corr.csv')

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

        if (len(lai)>0) & (len(vod)>0):
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

        df_met = pd.concat((temp,prec,rad), axis=1, keys=['T','P','R']).resample('M').mean().dropna()
        df_veg = pd.concat((lai,vod), axis=1, keys=['LAI','VOD']).resample('M').mean().dropna()

        index = df_veg.index.copy()

        tmp_df_met = df_met.copy()
        tmp_df_met.index = np.arange(len(df_met))

        res = pd.DataFrame(index=(i,))
        for lag in np.arange(-12,12):
            tmp_df_veg = df_veg.reindex(df_met.index)
            tmp_df_veg.index = np.arange(len(tmp_df_veg)) + lag
            corr = pd.concat((tmp_df_met, tmp_df_veg), axis=1).corr()
            res[f'R_LAI_T_{lag}'] = corr['R']['LAI']
            res[f'R_LAI_P_{lag}'] = corr['T']['LAI']
            res[f'R_LAI_R_{lag}'] = corr['P']['LAI']
            res[f'R_VOD_T_{lag}'] = corr['R']['VOD']
            res[f'R_VOD_P_{lag}'] = corr['T']['VOD']
            res[f'R_VOD_R_{lag}'] = corr['P']['VOD']

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

    cbr = (-0.9,0.9)

    cmap = cc.cm.bjy

    for met, tit in zip(variables, titles):
        f = plt.figure(figsize=(22, 11))
        plt.suptitle(f'Lagged correlation ({tit}) [-]', fontsize=fontsize+2, y=0.93)

        for i, veg in enumerate(['LAI', 'VOD']):
            for j, lag in enumerate(lags):

                var = f'R_{veg}_{met}_{lag}'

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


        fname = f'/Users/u0116961/Documents/work/deforestation_paper/plots/lagged_corr_{met}.png'
        f.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()


def plot_lagged_corr_boxplot():

    res = pd.read_csv('/Users/u0116961/Documents/work/deforestation_paper/lagged_corr.csv', index_col=0)

    ds = io('LAI')
    lut = ds.lut.reindex(res.index)

    variables = ['T', 'P', 'R']
    titles = ['Temperature', 'Precipitation', 'Radiation']
    datasets = ['LAI', 'VOD']

    lags = np.arange(-6, 1, 1)

    pos = [i + j for i in np.arange(1, len(variables) + 1) for j in np.linspace(-0.4, 0.4, len(lags))]
    colors = [f'{s}' for n in np.arange(len(variables)) for s in np.linspace(0.2, 0.98, len(lags))]
    # colors = [s for n in np.arange(len(variables)) for s in ['lightblue', 'lightgreen', 'coral']]

    f = plt.figure(figsize=(14, 8))
    fontsize = 14

    for n, d in enumerate(datasets):
        ax = plt.subplot(2, 1, n+1)
        if n == 0:
            axpos = ax.get_position()

        data = list()

        for v, t in zip(variables, titles):
            for lag in lags:
                var = f'R_{d}_{v}_{lag}'
                tmp = res[var].values
                tmp = tmp[tmp>0]
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
        if n == 0:
            plt.xticks([], [], fontsize=fontsize)
        else:
            plt.xticks(np.arange(1,len(variables)+1), titles , fontsize=fontsize)
        # plt.ylim(-0.8,0.9)
        # plt.yticks(np.arange(-0.8,0.9,0.2))
        plt.ylim(0,0.9)
        plt.yticks(np.arange(0,1,0.2))
        plt.axhline(color='k', linestyle='--', linewidth=1)

        for i in np.arange(1,len(variables)):
            plt.axvline(i + 0.5, linewidth=1.5, color='k')

        plt.ylabel(d, fontsize=fontsize)

        if n==0:
            plt.title('Lagged correlation [-]',fontsize=fontsize)

    f.subplots_adjust(hspace=0.1)

    plt.figlegend((box['boxes'][0:len(lags)]), lags, 'upper right', title='Lag [m]',
                  bbox_to_anchor=(axpos.x1+0.09,axpos.y1+0.013), fontsize=fontsize-2)

    fname = f'/Users/u0116961/Documents/work/deforestation_paper/plots/lagged_corr_boxplot_pos.png'
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
    #

if __name__=='__main__':
    AGB_VOD_LAI()
    # mean_veg_met()
    # calc_lagged_corr()
    # plot_lagged_corr_spatial()
    # plot_lagged_corr_boxplot()
    # calc_trends()
    # plot_trends()
    # plot_mask()

