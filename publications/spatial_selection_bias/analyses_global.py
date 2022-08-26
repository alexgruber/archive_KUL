import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

import seaborn as sns
sns.set_context('talk', font_scale=0.6)
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import colorcet as cc

from myprojects.publications.spatial_selection_bias.analyses_insitu import get_lc_names
from myprojects.readers.insitu import ISMN_io
from validation_good_practice.plots import plot_ease_img

def plot_landcover_class(dir_out):

    fname = dir_out / 'lc_class.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result_global.csv", index_col=0)

    lc = get_lc_names()
    x_legend = '\n'.join(f'{n} - {name}' for n, name in zip(lc.index, lc.values))

    freq = pd.Series(0, index=lc.index.values)
    for cl in np.unique(res['landcover_class']):
        tmp_freq = res[res.landcover_class == cl]['landcover_class'].value_counts().sort_index()
        freq.loc[tmp_freq.index] = tmp_freq.values
    # freq = res['landcover_class'].value_counts().sort_index()
    # freq.index = freq.index.values.astype('int')

    fontsize = 12

    fig = plt.figure(figsize=(15,7))

    ax = freq.plot.bar()

    plt.xticks(rotation=0, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Number of grid cells', fontsize=fontsize)
    plt.title('MODIS IGBP landcover class', fontsize=fontsize)

    t = ax.text(.56, .12, x_legend, transform=ax.figure.transFigure, fontsize = fontsize-2)
    fig.subplots_adjust(right=.55)

    # plt.tight_layout()
    # plt.show()

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def plot_vegetation_fraction(dir_out):

    fname = dir_out / 'vegetation_global.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result_global.csv", index_col=0)

    cols = ['vegetation_opacity_med', 'vegetation_water_content_med',
            'vegetation_opacity_iqr', 'vegetation_water_content_iqr']

    lims = [[-0.05, 0.65], [-0.1, 5.1],
            [-0.01, 0.21], [-0.1, 2.1]]

    axs = res[cols].hist(bins=15, figsize=(10,8), grid=False)

    for ax, lim in zip(axs.flatten(), lims):
        ax.set_xlim(lim)

    # plt.tight_layout()
    # plt.show()

    plt.gcf().savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def plot_vegetation_maps(dir_out):

    fname = dir_out / 'vegetation_maps.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result_global.csv", index_col=0)
    res['row'] = res['ease_row']
    res['col'] = res['ease_col']

    cols = ['vegetation_opacity_med', 'vegetation_water_content_med']
    titles = ['Vegetation Opacity', 'Vegetation Water content']
    lims = [(0.01,1), (0.1, 10)]

    plt.figure(figsize=(16,10))

    for i, (col, tit, lim) in enumerate(zip(cols, titles, lims)):

        plt.subplot(2,1,i+1)
        plot_ease_img(res, col,
                      llcrnrlat=-58,
                      urcrnrlat=80,
                      llcrnrlon=-180,
                      urcrnrlon=180,
                      log=True,
                      cbrange=lim, title=tit, cmap='viridis',
                      fontsize=16, plot_cb=True)


    # plt.tight_layout()
    # plt.show()

    plt.gcf().savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def boxplot_tcr_gldas_per_veg(dir_out):

    fname = dir_out / 'boxplot_tcr_per_veg.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result_global.csv", index_col=0)
    res = res[res['len'] > 100]

    sensors = ['ASCAT', 'AMSR2', 'SMAP']

    snr_cols = [f'snr_{sensor}' for sensor in sensors]
    r_cols = [f'r_{sensor}' for sensor in sensors]
    p_cols = [f'p_GLDAS_and_{sensor}' for sensor in sensors]

    res = res[res['len'] > 100]
    for pcol in p_cols:
        res = res[res[pcol] < 0.05]

    for r_col, snr_col in zip(r_cols, snr_cols):
        res[r_col] = res[snr_col] / (1 + res[snr_col])
        res.loc[(res[r_col] > 1)|(res[r_col] < 0.), r_col] = np.nan
    res.dropna(inplace=True)

    offsets = [-0.25, 0, 0.25]
    colors = ['red','orange','green']

    vod = res['vegetation_opacity_med']
    vwc = res['vegetation_water_content_med']
    titles = ['Vegetation Opacity', 'Vegetation Water Content']

    lim_vod = np.linspace(0, 0.4, 5)
    lim_vwc = np.linspace(0, 4, 5)

    fig, axs = plt.subplots(1, 2, figsize=(15,6), sharey='all')
    fontsize = 12

    for i, (veg, lim, tit) in enumerate(zip([vod,vwc],[lim_vod,lim_vwc], titles)):
        # r, c = np.unravel_index(i, (1, 2))
        ticks, data, pos, colorss = [], [], [], []
        for j in np.arange(len(lim)):
            if j < 4:
                ind = (veg >= lim[j]) & (veg < lim[j+1])
                ticks.append(f'{lim[j]:.1f} - {lim[j+1]:.1f} \n  (n = {len(np.where(ind)[0])})')
            else:
                ind = veg >= lim[j]
                ticks.append(f'>={lim[j]:.1f} \n  (n = {len(np.where(ind)[0])})')
            for k, (offs, col, color) in enumerate(zip(offsets, r_cols, colors)):
                data.append(res.loc[ind, col])
                pos.append(j + 1 + offs)
                colorss.append(color)

        box = axs[i].boxplot(data, whis=[5, 95], showfliers=False, positions=pos, widths=0.12, patch_artist=True)
        for patch, col in zip(box['boxes'],colorss):
            patch.set(color='black', linewidth=2)
            patch.set_facecolor(col)
        for patch in box['medians']:
            patch.set(color='black', linewidth=2)
        for patch in box['whiskers']:
            patch.set(color='black', linewidth=1)
        plt.figlegend((box['boxes'][0:4]), sensors, 'upper right', fontsize=fontsize)
        axs[i].set_xticks(np.arange(len(ticks)) + 1, ticks, fontsize=fontsize)
        axs[i].set_xlim(0.5, len(ticks) + 0.5)
        axs[i].tick_params(axis='y', labelsize=fontsize)
        axs[i].set_ylim(0.5, 1.0)
        for j in np.arange(len(ticks)):
            axs[i].axvline(j + 0.5, linewidth=2, linestyle='--',  color='k')
        axs[i].set_title(tit, fontsize=fontsize)

    # plt.tight_layout()
    # plt.show()

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def boxplot_tcr(dir_out):

    fname = dir_out / 'boxplot_tcr.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result_global.csv", index_col=0)

    sensors = ['ASCAT','AMSR2','SMAP']
    labels = sensors
    snr_cols = [f'snr_{sensor}' for sensor in sensors]
    r_cols = [f'r_{sensor}' for sensor in sensors]
    p_cols = [f'p_GLDAS_and_{sensor}' for sensor in sensors]

    res = res[res['len'] > 100]
    for pcol in p_cols:
        res = res[res[pcol] < 0.05]

    for r_col, snr_col in zip(r_cols, snr_cols):
        res[r_col] = res[snr_col] / (1 + res[snr_col])
        res.loc[(res[r_col] > 1)|(res[r_col] < 0.), r_col] = np.nan

    fig, axs = plt.subplots(3, 5, figsize=(17,10), sharex='all', sharey='all')
    fontsize = 12

    lcs = np.unique(res['landcover_class'])
    for i, lc in enumerate(lcs):
        r, c = np.unravel_index(i,(3,5))
        tmp_res = res[res['landcover_class']==lc][r_cols]

        data = [tmp_res[col].dropna().values for col in tmp_res]
        bp = axs[r, c].boxplot(data, whis=[5, 95], showfliers=False,
                               labels=labels, patch_artist=True)

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black')
        for idx, patch in enumerate(bp['boxes']):
            if idx == np.argmin(tmp_res.median(axis=0)):
                patch.set(facecolor='red')
            elif idx == np.argmax(tmp_res.median(axis=0)):
                patch.set(facecolor='green')
            else:
                patch.set(facecolor='orange')

        axs[r, c].set_title(f'LC class: {lc} (n = {len(tmp_res)})', fontsize=fontsize)
        axs[r, c].tick_params(axis='x', labelsize=fontsize)
        axs[r, c].tick_params(axis='y', labelsize=fontsize)
        axs[r, c].set_ylim([0.3,1])

    # plt.tight_layout()
    # plt.show()

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


if __name__=='__main__':

    dir_out = Path(r"H:\work\publications\spatial_selection_bias\plots_global")
    if not dir_out.exists():
        Path.mkdir(dir_out, parents=True)

    # plot_vegetation_fraction(dir_out)
    # plot_vegetation_maps(dir_out)
    # boxplot_tcr(dir_out)
    # boxplot_tcr_gldas_per_veg(dir_out)
    plot_landcover_class(dir_out)