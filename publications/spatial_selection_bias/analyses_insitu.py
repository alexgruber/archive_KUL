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

from myprojects.readers.insitu import ISMN_io

def get_lc_names():

    lcs = {1: 'Evergreen Needleleaf Forest',
           2: 'Evergreen Broadleaf Forest',
           3: 'Deciduous Needleleaf Forest',
           4: 'Deciduous Broadleaf Forest',
           5: 'Mixed Forests',
           6: 'Closed Shrublands',
           7: 'Open Shrublands',
           8: 'Woody Savannas',
           9: 'Savannas',
           10: 'Grasslands',
           11: 'Permanent Wetlands',
           12: 'Croplands',
           13: 'Urban and Built-Up',
           14: 'Cropland / Natrual Vegetation Mosaic',
           15: 'Snow and Ice',
           16: 'Barren or Sparsely Vegetated'}

    return pd.Series(lcs)

def plot_network(dir_out):

    fname = dir_out / 'network.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result.csv", index_col=0)
    res.loc[res.network=='FLUXNET-AMERIFLUX','network'] = 'FLUXNET'
    freq = res['network'].value_counts().sort_index()

    fontsize = 12

    fig = plt.figure(figsize=(10,5))

    freq.plot.bar()
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Number of stations', fontsize=fontsize)
    plt.title('Network station frequency')

    # plt.tight_layout()
    # plt.show()

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sand_clay_fraction(dir_out):

    fname = dir_out / 'sand_clay.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result.csv", index_col=0)
    res['tmp1'] = ''
    res['tmp2'] = ''

    for i in np.arange(10)*10:
        res.loc[(res['sand_fraction']>=i)&(res['sand_fraction']<(i+10)), 'tmp1'] = f'{i}-{i+10}'
        res.loc[(res['clay_fraction']>=i)&(res['clay_fraction']<(i+10)), 'tmp2'] = f'{i}-{i+10}'

    freq1 = res['tmp1'].value_counts().sort_index()
    freq2 = res['tmp2'].value_counts().sort_index().append(pd.Series({'80-90':0,'90-100':0}))

    fontsize = 12

    fig = plt.figure(figsize=(8,12))

    plt.subplot(2,1,1)
    freq1.plot.bar()
    plt.xticks(rotation=15, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Number of stations', fontsize=fontsize)
    plt.title('Sand fraction distribution', fontsize=fontsize)

    plt.subplot(2,1,2)
    freq2.plot.bar()
    plt.xticks(rotation=15, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Number of stations', fontsize=fontsize)
    plt.title('Clay fraction distribution', fontsize=fontsize)

    # plt.tight_layout()
    # plt.show()

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def plot_vegetation_fraction(dir_out):

    fname = dir_out / 'vegetation_insitu.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result_insitu.csv", index_col=0)

    cols = ['vegetation_opacity_med', 'vegetation_water_content_med',
            'vegetation_opacity_iqr', 'vegetation_water_content_iqr']

    lims = [[-0.05,0.65], [-0.1,5.1],
            [-0.01,0.21], [-0.1,2.1]]

    axs = res[cols].hist(bins=8, figsize=(10,8), grid=False)

    for ax, lim in zip(axs.flatten(),lims):
        ax.set_xlim(lim)

    # plt.tight_layout()
    # plt.show()

    plt.gcf().savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

# def plot_vegetation_fraction(dir_out):
#
#     fname = dir_out / 'vegetation.png'
#
#     res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result.csv", index_col=0)
#     res['tmp1'] = ''
#     res['tmp2'] = ''
#     res['tmp3'] = ''
#     res['tmp4'] = ''
#
#     lim_vo_med = np.linspace(0, 0.6, 7)
#     lim_vwc_med = np.linspace(0, 5, 6)
#     lim_vo_std = np.linspace(0, 0.18, 7)
#     lim_vwc_std = np.linspace(0, 2, 9)
#
#     res.loc[res['vegetation_opacity_iqr']>=lim_vo_std[-1],'vegetation_opacity_iqr'] = 0.1799
#     res.loc[res['vegetation_water_content_iqr']>=lim_vwc_std[-1],'vegetation_water_content_iqr'] = 0.1999
#
#     labels_vo_med = []
#     labels_vwc_med = []
#     labels_vo_std = []
#     labels_vwc_std = []
#
#     for i in np.arange(len(lim_vo_med)-1):
#         res.loc[(res['vegetation_opacity_med'] >= lim_vo_med[i]) & (res['vegetation_opacity_med'] < lim_vo_med[i+1]), 'tmp1'] = i
#         labels_vo_med.append(f'{lim_vo_med[i]:.1f}-{lim_vo_med[i+1]:.1f}')
#     for i in np.arange(len(lim_vwc_med)-1):
#         res.loc[(res['vegetation_water_content_med'] >= lim_vwc_med[i]) & (res['vegetation_water_content_med'] < lim_vwc_med[i+1]), 'tmp2'] = i
#         labels_vwc_med.append(f'{lim_vwc_med[i]:.0f}-{lim_vwc_med[i+1]:.0f}')
#     for i in np.arange(len(lim_vo_std)-1):
#         res.loc[(res['vegetation_opacity_iqr'] >= lim_vo_std[i]) & (res['vegetation_opacity_iqr'] < lim_vo_std[i+1]), 'tmp3'] = i
#         labels_vo_std.append(f'{lim_vo_std[i]:.2f}-{lim_vo_std[i+1]:.2f}')
#     for i in np.arange(len(lim_vwc_std)-1):
#         res.loc[(res['vegetation_water_content_iqr'] >= lim_vwc_std[i]) & (res['vegetation_water_content_iqr'] < lim_vwc_std[i+1]), 'tmp4'] = i
#         labels_vwc_std.append(f'{lim_vwc_std[i]:.2f}-{lim_vwc_std[i+1]:.2f}')
#
#     freq1 = res['tmp1'].value_counts().sort_index()
#     freq2 = res['tmp2'].value_counts().sort_index()
#     freq3 = res['tmp3'].value_counts().sort_index()
#     freq4 = res['tmp4'].value_counts().sort_index()
#
#     fontsize = 10
#
#     fig = plt.figure(figsize=(12,11))
#
#     plt.subplot(2, 2, 1)
#     freq1.plot.bar()
#     plt.xlim(-0.5, len(labels_vo_med)-0.5)
#     plt.xticks(np.arange(len(labels_vo_med)), labels_vo_med, rotation=15, fontsize=fontsize)
#     plt.yticks(fontsize=fontsize)
#     plt.ylabel('Number of stations', fontsize=fontsize)
#     plt.title('Vegetation Opacity (Median)', fontsize=fontsize)
#
#     plt.subplot(2, 2, 2)
#     freq2.plot.bar()
#     plt.xlim(-0.5, len(labels_vwc_med)-0.5)
#     plt.xticks(np.arange(len(labels_vwc_med)), labels_vwc_med, rotation=15, fontsize=fontsize)
#     plt.yticks(fontsize=fontsize)
#     plt.ylabel('Number of stations', fontsize=fontsize)
#     plt.title('Vegetation Water Content (Median)', fontsize=fontsize)
#
#     plt.subplot(2, 2, 3)
#     freq3.plot.bar()
#     plt.xlim(-0.5, len(labels_vo_std)-0.5)
#     plt.xticks(np.arange(len(labels_vo_std)), labels_vo_std, rotation=15, fontsize=fontsize)
#     plt.yticks(fontsize=fontsize)
#     plt.ylabel('Number of stations', fontsize=fontsize)
#     plt.title('Vegetation Opacity (IQR)', fontsize=fontsize)
#
#     plt.subplot(2, 2, 4)
#     freq4.plot.bar()
#     plt.xlim(-0.5, len(labels_vwc_std)-0.5)
#     plt.xticks(np.arange(len(labels_vwc_std)), labels_vwc_std, rotation=15, fontsize=fontsize)
#     plt.yticks(fontsize=fontsize)
#     plt.ylabel('Number of stations', fontsize=fontsize)
#     plt.title('Vegetation Water Content (IQR)', fontsize=fontsize)
#
#     # plt.tight_layout()
#     # plt.show()
#
#     fig.savefig(fname, dpi=300, bbox_inches='tight')
#     plt.close()

def plot_climate_class(dir_out):

    fname = dir_out / 'kg_class.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result.csv", index_col=0)
    freq = res['climate_KG'].value_counts().sort_index()

    kg = pd.Series(ISMN_io().io.get_climate_types())
    kg = kg.loc[freq.index]

    # freq.index = np.arange(len(freq))
    # lc.index = np.arange(len(lc))
    x_legend = '\n'.join(f'{n} - {name}' for n, name in zip(kg.index, kg.values))

    fontsize = 10

    fig = plt.figure(figsize=(15,5))

    ax = freq.plot.bar()
    ax.bar(freq.index, freq.values)

    plt.xticks(rotation=0, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Number of stations', fontsize=fontsize)
    plt.title('Koeppen-Geiger climate class distribution', fontsize=fontsize)

    t = ax.text(.56, .12, x_legend, transform=ax.figure.transFigure, fontsize = fontsize-2)
    fig.subplots_adjust(right=.55)

    # plt.tight_layout()
    # plt.show()

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def plot_climate_class_per_network(dir_out):

    fname = dir_out / 'kg_class_per_net.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result.csv", index_col=0)

    networks = np.unique(res['network'])
    freq = pd.DataFrame(0, columns=networks, index=np.unique(res['climate_KG']))
    for net in networks:
        tmp_freq = res[res.network==net]['climate_KG'].value_counts().sort_index()
        freq.loc[tmp_freq.index, net] = tmp_freq.values

    kg = pd.Series(ISMN_io().io.get_climate_types()).loc[freq.index]

    x_legend = '\n'.join(f'{n} - {name}' for n, name in zip(kg.index, kg.values))

    fontsize = 10

    fig = plt.figure(figsize=(15,5))

    ax = freq.plot.bar(stacked=True, ax=plt.gca())

    plt.xticks(rotation=0, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Number of stations', fontsize=fontsize)
    plt.title('Koeppen-Geiger climate class distribution', fontsize=fontsize)
    plt.legend(loc='upper left',fontsize=fontsize-4)

    ax.text(.56, .12, x_legend, transform=ax.figure.transFigure, fontsize = fontsize-2)
    fig.subplots_adjust(right=.55)

    # plt.tight_layout()
    # plt.show()

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def plot_landcover_class(dir_out):

    fname = dir_out / 'lc_class.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result.csv", index_col=0)
    freq = res['lc_2010'].value_counts().sort_index()

    lc = pd.Series(ISMN_io().io.get_landcover_types())
    lc = lc.loc[freq.index]

    freq.index = np.arange(len(freq))
    lc.index = np.arange(len(lc))
    x_legend = '\n'.join(f'{n} - {name}' for n, name in zip(lc.index, lc.values))

    fontsize = 12

    fig = plt.figure(figsize=(15,5))

    ax = freq.plot.bar()
    ax.bar(freq.index, freq.values)

    plt.xticks(rotation=0, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Number of stations', fontsize=fontsize)
    plt.title('Landcover (2010) distribution', fontsize=fontsize)

    t = ax.text(.56, .12, x_legend, transform=ax.figure.transFigure, fontsize = fontsize-2)
    fig.subplots_adjust(right=.55)

    # plt.tight_layout()
    # plt.show()

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def plot_landcover_class_per_network(dir_out):

    fname = dir_out / 'lc_class_per_net.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result.csv", index_col=0)

    networks = np.unique(res['network'])
    freq = pd.DataFrame(0, columns=networks, index=np.unique(res['lc_2010']))
    for net in networks:
        tmp_freq = res[res.network==net]['lc_2010'].value_counts().sort_index()
        freq.loc[tmp_freq.index, net] = tmp_freq.values

    # lc = pd.Series(ISMN_io().io.get_landcover_types()).loc[freq.index]
    lc = pd.Series(ISMN_io().io.get_landcover_types())

    x_legend = '\n'.join(f'{n} - {name}' for n, name in zip(lc.index, lc.values))

    fontsize = 10

    fig = plt.figure(figsize=(15,7))

    ax = freq.plot.bar(stacked=True, ax=plt.gca())

    plt.xticks(rotation=0, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Number of stations', fontsize=fontsize)
    plt.title('Landcover 2010 class distribution', fontsize=fontsize)
    plt.legend(loc='upper right',fontsize=fontsize-2)

    ax.text(.56, .12, x_legend, transform=ax.figure.transFigure, fontsize = fontsize-2)
    fig.subplots_adjust(right=.55)

    # plt.tight_layout()
    # plt.show()

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def plot_smap_landcover_class_per_network(dir_out):

    fname = dir_out / 'lc_class_per_net_smap.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\result_insitu.csv", index_col=0)

    lc = get_lc_names()
    x_legend = '\n'.join(f'{n} - {name}' for n, name in zip(lc.index, lc.values))

    networks = np.unique(res['network'])
    lcs = np.unique(res['landcover_class'])

    freq = pd.DataFrame(0, columns=networks, index=lc.index.values)

    for net in networks:
        tmp_freq = res[res.network==net]['landcover_class'].value_counts().sort_index()
        freq.loc[tmp_freq.index, net] = tmp_freq.values
    freq.index = freq.index.astype('int')

    # lc = pd.Series(ISMN_io().io.get_landcover_types()).loc[freq.index]

    fontsize = 10

    fig = plt.figure(figsize=(15,7))

    ax = freq.plot.bar(stacked=True, ax=plt.gca())

    plt.xticks(rotation=0, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Number of stations', fontsize=fontsize)
    plt.title('MODIS IGBP landcover class', fontsize=fontsize)
    plt.legend(loc='upper left',fontsize=fontsize-2)

    ax.text(.56, .12, x_legend, transform=ax.figure.transFigure, fontsize = fontsize-2)
    fig.subplots_adjust(right=.55)

    # plt.tight_layout()
    # plt.show()

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def plot_network_per_lc(dir_out):

    fname = dir_out / 'network_per_lc.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result.csv", index_col=0)
    res.loc[res.network == 'FLUXNET-AMERIFLUX', 'network'] = 'FLUXNET'

    networks = np.unique(res['network'])
    lcs = np.unique(res['lc_2010'])
    freq = pd.DataFrame(0, columns=lcs, index=networks)
    for lc in lcs:
        tmp_freq = res[res['lc_2010']==lc]['network'].value_counts().sort_index()
        freq.loc[tmp_freq.index, lc] = tmp_freq.values
    # for net in networks:
    #     tmp_freq = res[res.network==net]['lc_2010'].value_counts().sort_index()
    #     freq.loc[tmp_freq.index, net] = tmp_freq.values

    lc = pd.Series(ISMN_io().io.get_landcover_types()).loc[lcs]
    # lc.index = np.arange(len(lc))
    # freq.columns = np.arange(len(lc))

    x_legend = '\n'.join(f'{n} - {name}' for n, name in zip(lc.index, lc.values))

    fontsize = 10

    fig = plt.figure(figsize=(15,6))

    ax = freq.plot.bar(stacked=True, ax=plt.gca())

    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Number of stations', fontsize=fontsize)
    plt.title('Landcover 2010 class distribution', fontsize=fontsize)
    plt.legend(loc='upper right',fontsize=fontsize-2)

    ax.text(.56, .49, x_legend, transform=ax.figure.transFigure, fontsize = fontsize-2)
    fig.subplots_adjust(right=.55)

    # plt.tight_layout()
    # plt.show()

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def boxplot_r_insitu_per_lc(dir_out):

    fname = dir_out / 'boxplot_r_insitu.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result.csv", index_col=0)
    res = res[res['len'] > 100]
    lcs = np.unique(res['lc_2010'])

    sensors = ['HSAF','ASCAT','AMSR2','SMAP']
    labels = ['HSF','ASC','AMS','SMP']
    r_cols = [f'R_{sensor}_and_ISMN' for sensor in sensors]
    p_cols = [f'p_{sensor}_and_ISMN' for sensor in sensors]

    fig, axs = plt.subplots(3, 6, figsize=(18,10), sharex='all', sharey='all')
    fontsize = 12

    for i, lc in enumerate(lcs):
        r, c = np.unravel_index(i,(3,6))
        tmp_res = res[res['lc_2010']==lc][r_cols + p_cols]
        for r_col, p_col in zip(r_cols, p_cols):
            tmp_res.loc[tmp_res[p_col]>=0.05,r_col] = np.nan
        tmp_res.drop(p_cols, axis='columns', inplace=True)

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
        axs[r, c].set_ylim([-0.4,0.9])

    # plt.tight_layout()
    # plt.show()

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def boxplot_r_gldas_per_lc(dir_out):

    fname = dir_out / 'boxplot_r_gldas.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result.csv", index_col=0)
    res = res[res['len'] > 100]
    lcs = np.unique(res['lc_2010'])

    sensors = ['HSAF','ASCAT','AMSR2','SMAP']
    labels = ['HSF','ASC','AMS','SMP']
    r_cols = [f'R_GLDAS_and_{sensor}' for sensor in sensors]
    p_cols = [f'p_GLDAS_and_{sensor}' for sensor in sensors]

    fig, axs = plt.subplots(3, 6, figsize=(18,10), sharex='all', sharey='all')
    fontsize = 12

    for i, lc in enumerate(lcs):
        r, c = np.unravel_index(i,(3,6))
        tmp_res = res[res['lc_2010']==lc][r_cols + p_cols]
        for r_col, p_col in zip(r_cols, p_cols):
            tmp_res.loc[tmp_res[p_col]>=0.05,r_col] = np.nan
        tmp_res.drop(p_cols, axis='columns', inplace=True)

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
        axs[r, c].set_ylim([-0.4,0.9])

    # plt.tight_layout()
    # plt.show()

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def boxplot_r_insitu_per_smap_lc(dir_out):

    fname = dir_out / 'boxplot_r_insitu.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result_insitu.csv", index_col=0)
    res = res[res['len'] > 100]
    res = res[res['landcover_class'] != 16]
    lcs = np.unique(res['landcover_class'])

    sensors = ['ASCAT','AMSR2','SMAP']
    # labels = ['ASC','AMS','SMP']
    labels = sensors
    r_cols = [f'R_{sensor}_and_ISMN' for sensor in sensors]
    p_cols = [f'p_{sensor}_and_ISMN' for sensor in sensors]

    fig, axs = plt.subplots(2, 3, figsize=(12,6), sharex='all', sharey='all')
    fontsize = 8

    for i, lc in enumerate(lcs):
        r, c = np.unravel_index(i,(2,3))
        tmp_res = res[res['landcover_class']==lc][r_cols + p_cols]
        for r_col, p_col in zip(r_cols, p_cols):
            tmp_res.loc[tmp_res[p_col]>=0.05,r_col] = np.nan
        tmp_res.drop(p_cols, axis='columns', inplace=True)

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
        axs[r, c].set_ylim([-0.8,1])
        axs[r, c].axhline(color='k', linestyle='--', linewidth=1)

    # plt.tight_layout()
    # plt.show()

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def boxplot_r_gldas_per_smap_lc(dir_out):

    fname = dir_out / 'boxplot_r_gldas.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result_insitu.csv", index_col=0)
    res = res[res['len'] > 100]
    res = res[res['landcover_class'] != 16]
    lcs = np.unique(res['landcover_class'])

    sensors = ['ASCAT','AMSR2','SMAP']
    # labels = ['ASC','AMS','SMP']
    labels = sensors
    r_cols = [f'R_GLDAS_and_{sensor}' for sensor in sensors]
    p_cols = [f'p_GLDAS_and_{sensor}' for sensor in sensors]

    fig, axs = plt.subplots(2, 3, figsize=(12,6), sharex='all', sharey='all')
    fontsize = 8

    for i, lc in enumerate(lcs):
        r, c = np.unravel_index(i,(2,3))
        tmp_res = res[res['landcover_class']==lc][r_cols + p_cols]
        for r_col, p_col in zip(r_cols, p_cols):
            tmp_res.loc[tmp_res[p_col]>=0.05,r_col] = np.nan
        tmp_res.drop(p_cols, axis='columns', inplace=True)

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
        axs[r, c].set_ylim([-0.8,1])
        axs[r, c].axhline(color='k', linestyle='--', linewidth=1)

    # plt.tight_layout()
    # plt.show()

    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def boxplot_r_insitu_per_veg(dir_out):

    fname = dir_out / 'boxplot_r_insitu_veg.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result_insitu.csv", index_col=0)
    res = res[res['len'] > 100]

    sensors = ['ASCAT', 'AMSR2', 'SMAP']
    cols = [f'R_{sensor}_and_ISMN' for sensor in sensors]
    for col in [f'p_{sensor}_and_ISMN' for sensor in sensors]:
        res = res[res[col] < 0.05]
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
                ticks.append(f'{lim[j]:.1f} - {lim[j+1]:.1f}')
            else:
                ind = veg >= lim[j]
                ticks.append(f'>={lim[j]:.1f}')
            for k, (offs, col, color) in enumerate(zip(offsets, cols, colors)):
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
        axs[i].set_ylim(0.0, 1.0)
        for j in np.arange(len(ticks)):
            axs[i].axvline(j + 0.5, linewidth=2, linestyle='--',  color='k')
        axs[i].set_title(tit, fontsize=fontsize)

    plt.tight_layout()
    plt.show()

    # fig.savefig(fname, dpi=300, bbox_inches='tight')
    # plt.close()

def boxplot_tcr_insitu_per_veg(dir_out):

    fname = dir_out / 'boxplot_tcr_insitu_veg.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\result_insitu.csv", index_col=0)
    res = res[res['len'] > 100]

    sensors = ['ASCAT', 'AMSR2', 'SMAP']

    snr_cols = [f'snr_{sensor}_ref_ISMN' for sensor in sensors]
    r_cols = [f'r_{sensor}' for sensor in sensors]
    p_cols = [f'p_{sensor}_and_ISMN' for sensor in sensors]
    # p_cols = [f'p_GLDAS_and_{sensor}' for sensor in sensors]

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
                ticks.append(f'{lim[j]:.1f} - {lim[j+1]:.1f}  \n  (n = {len(np.where(ind)[0])})')
            else:
                ind = veg >= lim[j]
                ticks.append(f'>={lim[j]:.1f}  \n  (n = {len(np.where(ind)[0])})')
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

def boxplot_tcr_gldas_per_veg(dir_out):

    fname = dir_out / 'boxplot_tcr_gldas_veg.png'

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\result_insitu.csv", index_col=0)
    res = res[res['len'] > 100]

    sensors = ['ASCAT', 'AMSR2', 'SMAP']

    snr_cols = [f'snr_{sensor}_ref_GLDAS' for sensor in sensors]
    r_cols = [f'r_{sensor}' for sensor in sensors]
    p_cols = [f'p_GLDAS_and_{sensor}' for sensor in sensors]
    # p_cols = [f'p_GLDAS_and_{sensor}' for sensor in sensors]

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
                ticks.append(f'{lim[j]:.1f} - {lim[j+1]:.1f}  \n  (n = {len(np.where(ind)[0])})')
            else:
                ind = veg >= lim[j]
                ticks.append(f'>={lim[j]:.1f}  \n  (n = {len(np.where(ind)[0])})')
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

    res = pd.read_csv(r"H:\work\publications\spatial_selection_bias\results\result.csv", index_col=0)

    sensors = ['HSAF','ASCAT','AMSR2','SMAP']
    labels = ['HSF','ASC','AMS','SMP']
    snr_cols = [f'snr_{sensor}' for sensor in sensors]
    r_cols = [f'r_{sensor}' for sensor in sensors]
    p_cols1 = [f'p_{sensor}_and_ISMN' for sensor in sensors]
    p_cols2 = [f'p_GLDAS_and_{sensor}' for sensor in sensors]

    res = res[res['len'] > 100]
    for pcol in p_cols1:
        res = res[res[pcol] < 0.05]
    for pcol in p_cols2:
        res = res[res[pcol] < 0.05]

    for r_col, snr_col in zip(r_cols, snr_cols):
        res[r_col] = res[snr_col] / (1 + res[snr_col])
        res.loc[(res[r_col] > 1)|(res[r_col] < 0.), r_col] = np.nan

    fig, axs = plt.subplots(3, 6, figsize=(18,10), sharex='all', sharey='all')
    fontsize = 12

    lcs = np.unique(res['lc_2010'])
    for i, lc in enumerate(lcs):
        r, c = np.unravel_index(i,(3,6))
        tmp_res = res[res['lc_2010']==lc][r_cols]

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

    dir_out = Path(r"H:\work\publications\spatial_selection_bias\plots_insitu")
    if not dir_out.exists():
        Path.mkdir(dir_out, parents=True)

    # plot_network(dir_out)
    # plot_sand_clay_fraction(dir_out)
    # plot_vegetation_fraction(dir_out)
    # plot_climate_class(dir_out)
    # plot_climate_class_per_network(dir_out)
    # plot_landcover_class(dir_out)
    # plot_landcover_class_per_network(dir_out)
    plot_smap_landcover_class_per_network(dir_out)
    # plot_network_per_lc(dir_out)

    # boxplot_r_insitu_per_lc(dir_out)
    # boxplot_r_insitu_per_smap_lc(dir_out)
    # boxplot_r_insitu_per_veg(dir_out)
    # boxplot_r_gldas_per_lc(dir_out)
    # boxplot_r_gldas_per_smap_lc(dir_out)

    # boxplot_tcr_insitu_per_veg(dir_out)
    # boxplot_tcr_gldas_per_veg(dir_out)
    # boxplot_tcr(dir_out)