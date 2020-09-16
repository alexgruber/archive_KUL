
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

from scipy.stats import pearsonr

from itertools import repeat, combinations
from multiprocessing import Pool

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import seaborn as sns

from netCDF4 import Dataset, num2date
from pyldas.interface import LDAS_io
from myprojects.readers.ascat import HSAF_io

from myprojects.timeseries import calc_anom
from myprojects.functions import merge_files

# from validation_good_practice.ancillary.paths import Paths
# from validation_good_practice.plots import plot_ease_img

from pytesmo.metrics import ecol


class SMAP_io(object):

    def __init__(self):
        self.path = Path('/Users/u0116961/data_sets/SMAP/timeseries')

        grid = LDAS_io().grid
        lons, lats = np.meshgrid(grid.ease_lons, grid.ease_lats)
        self.lons = lons.flatten()
        self.lats = lats.flatten()


    def latlon2gpi(self, lat, lon):
        return np.argmin((self.lons - lon)**2 + (self.lats - lat)**2)


    def read(self, lat, lon):

        gpi = self.latlon2gpi(lat, lon)

        if (fname := self.path / f'{gpi}.csv').exists():
            ts_smap = pd.read_csv(fname,
                        index_col=0, parse_dates=True, names=('smap',))['smap'].resample('1d').mean().dropna()
        else:
            print(f'No valid SMAP data for gpi {gpi}')
            ts_smap = None

        return ts_smap

def stats(ts1, ts2):

    mdiff = np.mean(ts1) - np.mean(ts2)
    sdiff = np.std(ts1) - np.std(ts2)
    r2 = pearsonr(ts1, ts2)[0]**2

    return mdiff, sdiff, r2


def run():

    n_procs = 16

    part = np.arange(n_procs) + 1
    parts = repeat(n_procs, n_procs)

    p = Pool(n_procs)
    p.starmap(noahmp_version_comparison, zip(part, parts))

    res_path = f'/Users/u0116961/Documents/work/LIS/noahmp_version_comparison/'
    merge_files(res_path, pattern='result_part*.csv', fname='result.csv', delete=True)


def noahmp_version_comparison(part, parts):

    result_file = Path(f'/Users/u0116961/Documents/work/LIS/noahmp_version_comparison/result_part{part}.csv')
    if not result_file.parent.exists():
        Path.mkdir(result_file.parent, parents=True)

    ascat = HSAF_io()
    smap = SMAP_io()

    noah3 = Dataset('/Users/u0116961/data_sets/LIS/noahmp36/timeseries.nc')
    noah4 = Dataset('/Users/u0116961/data_sets/LIS/noahmp401/timeseries.nc')

    lats = noah3['lat'][:, :]
    lons = noah3['lon'][:, :]

    ind_lat, ind_lon = np.where(~lats.mask)

    # Split grid cell list for parallelization
    subs = (np.arange(parts + 1) * len(ind_lat) / parts).astype('int')
    subs[-1] = len(ind_lat)
    start = subs[part - 1]
    end = subs[part]

    # Look-up table that contains the grid cells to iterate over
    ind_lat = ind_lat[start:end]
    ind_lon = ind_lon[start:end]

    for i, (i_r, i_c) in enumerate(zip(ind_lat, ind_lon)):
        i += 1
        logging.info(f'{i} / {len(ind_lat)}')

        lat = lats[i_r, i_c]
        lon = lons[i_r, i_c]

        res = pd.DataFrame({'lat': lat, 'lon': lon}, index=(i,))

        for v in ['SM1','SM2','SM3','SM4','ST1','ST2','ST3','ST4','LAI','SWE']:
            if ('SM' in v) | ('ST' in v):
                res[f'mdiff_{v}'], res[f'sdiff_{v}'], res[f'r2_{v}'] = \
                    stats(noah4[v[0:2]][:, int(v[-1])-1, i_r, i_c], noah3[v[0:2]][:, int(v[-1])-1, i_r, i_c])
            else:
                res[f'mdiff_{v}'], res[f'sdiff_{v}'], res[f'r2_{v}'] = \
                    stats(noah4[v][:, i_r, i_c], noah3[v][:, i_r, i_c])

        time = pd.DatetimeIndex(num2date(noah3['time'][:], units=noah3['time'].units,
                                         only_use_python_datetimes=True, only_use_cftime_datetimes=False))
        df = pd.DataFrame({'noahmp36': noah3['SM'][:, 0, i_r, i_c],
                           'noahmp401': noah4['SM'][:, 0, i_r, i_c]}, index=time)

        ts_ascat = ascat.read(lat, lon)
        if ts_ascat is None:
            ts_ascat = pd.Series(name='ascat')

        ts_smap = smap.read(lat, lon)
        if ts_smap is None:
            ts_smap = pd.Series(name='smap')

        df = pd.concat((df, ts_ascat, ts_smap), axis='columns').dropna()

        for mode in ['abs','anom']:
            if mode == 'anom':
                for c in df.columns.values:
                    df[c] = calc_anom(df[c], longterm=False)

            res[f'len_{mode}'] = len(df)

            ec_res = ecol(df, correlated=[['noahmp36', 'noahmp401']])
            for c in df.columns.values:
                snr = 10**(ec_res[f'snr_{c}']/10)
                res[f'tcr2_{mode}_{c}'] = snr / (1 + snr)

        if not result_file.exists():
            res.to_csv(result_file, float_format='%0.4f')
        else:
            res.to_csv(result_file, float_format='%0.4f', mode='a', header=False)


def plot_img(data, tag,
                  llcrnrlat=24,
                  urcrnrlat=51,
                  llcrnrlon=-128,
                  urcrnrlon=-64,
                  cbrange=None,
                  cmap='jet_r',
                  title='',
                  fontsize=20,
                  plot_cb=False,
                  print_median=False):


    lons = np.arange(-180, 180.25, 0.25)
    lats = np.arange(-90, 90.25, 0.25)

    lons, lats = np.meshgrid(lons, lats)

    img = np.empty(lons.shape, dtype='float32')
    img.fill(None)

    ind_lat = ((data.loc[:, 'lat'].values+90)*4).astype('int')
    ind_lon = ((data.loc[:, 'lon'].values+180)*4).astype('int')

    img[ind_lat, ind_lon] = data.loc[:, tag]
    img_masked = np.ma.masked_invalid(img)

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

    if cbrange is not None:
        im.set_clim(vmin=cbrange[0], vmax=cbrange[1])

    if plot_cb is True:

        # ticks = np.arange(cbrange[0], cbrange[1] + 0.001, (cbrange[1] - cbrange[0]) / 4)
        cb = m.colorbar(im, "bottom", size="8%", pad="4%")
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)

    plt.title(title, fontsize=fontsize)

    if print_median is True:
        x, y = m(-79, 25)
        plt.text(x, y, 'm. = %.3f' % np.ma.median(img_masked), fontsize=fontsize - 2)

    return im


def plot_result_maps():

    root = Path('/Users/u0116961/Documents/work/LIS/noahmp_version_comparison/')
    if not (root / 'plots').exists():
        Path.mkdir(root / 'plots', parents=True)

    plot_mdiff = True
    plot_sdiff = True
    plot_r2 = True
    plot_tca = True
    plot_samples = True

    res = pd.read_csv(root / 'result.csv', index_col=0)

    res.loc[res['len_abs']==0,'len_abs'] = np.nan
    res.loc[res['len_anom']==0,'len_anom'] = np.nan

    for col in [x for x in res.columns.values if 'tcr2' in x]:
        res.loc[res['len_abs'] < 300, col] = np.nan
        res.loc[(res[col] < 0) | (res[col] > 1)] = np.nan

    res['tcr2_abs_noahmp_diff'] = res['tcr2_abs_noahmp401'] - res['tcr2_abs_noahmp36']
    res['tcr2_anom_noahmp_diff'] = res['tcr2_anom_noahmp401'] - res['tcr2_anom_noahmp36']

    params = ['SM1','SM2','SM3','SM4','ST1','ST2','ST3','ST4','LAI','SWE']

    fontsize = 12

    # --- bias ----
    if plot_mdiff:
        f = plt.figure(figsize=(20,9))
        for i, col in enumerate(['mdiff_' + x for x in params]):
            plt.subplot(3,4,i+1)
            if 'SM' in col:
                cbrange = [-0.07, 0.07]
            elif 'ST' in col:
                cbrange = [-1.6, 1.6]
            elif 'LAI' in col:
                cbrange = [-3, 3]
            else:
                cbrange = [-30, 30]
            im = plot_img(res, col, title=col, plot_cb=True, fontsize=fontsize, cmap='seismic_r', cbrange=cbrange)
        f.savefig(root / 'plots' / 'mdiff.png', dpi=300, bbox_inches='tight')
        plt.close()

    # --- sdiff ----
    if plot_sdiff:
        f = plt.figure(figsize=(20,9))
        for i, col in enumerate(['sdiff_' + x for x in params]):
            plt.subplot(3,4,i+1)
            if 'SM' in col:
                cbrange = [-0.05, 0.05]
            elif 'ST' in col:
                cbrange = [-1.4, 1.4]
            elif 'LAI' in col:
                cbrange = [-1.5, 1.5]
            else:
                cbrange = [-30, 30]
            im = plot_img(res, col, title=col, plot_cb=True, fontsize=fontsize, cmap='seismic_r', cbrange=cbrange)
        f.savefig(root / 'plots' / 'sdiff.png', dpi=300, bbox_inches='tight')
        plt.close()

    # --- correlation ----
    if plot_r2:
        f = plt.figure(figsize=(20,9))
        for i, col in enumerate(['r2_' + x for x in params]):
            plt.subplot(3,4,i+1)
            if 'SM' in col:
                cbrange = [0, 1]
            elif 'ST' in col:
                cbrange = [0.5, 1]
            elif 'LAI' in col:
                cbrange = [0, 1]
            else:
                cbrange = [0, 1]
            im = plot_img(res, col, title=col, plot_cb=True, fontsize=fontsize, cmap='jet_r', cbrange=cbrange)
        f.savefig(root / 'plots' / 'r2.png', dpi=300, bbox_inches='tight')
        plt.close()

    # --- TCA ----
    if plot_tca:
        f = plt.figure(figsize=(20,9))
        params = ['abs_ascat', 'abs_smap', 'anom_ascat', 'anom_smap', 'abs_noahmp36', 'abs_noahmp401', 'anom_noahmp36', 'anom_noahmp401']
        for i, col in enumerate(['tcr2_' + x for x in params]):
            plt.subplot(3,4,i+1)
            im = plot_img(res, col, title=col, plot_cb=True, fontsize=fontsize, cmap='jet_r', cbrange=[0,1])

        plt.subplot(3,4,9)
        im = plot_img(res, 'tcr2_abs_noahmp_diff', title='tcr2_abs_noahmp (401 minus 36)', plot_cb=True, fontsize=fontsize, cmap='seismic_r', cbrange=[-0.13,0.13])
        plt.subplot(3,4,11)
        im = plot_img(res, 'tcr2_anom_noahmp_diff', title='tcr2_anom_noahmp (401 minus 36)', plot_cb=True, fontsize=fontsize, cmap='seismic_r', cbrange=[-0.13,0.13])

        f.savefig(root / 'plots' / 'tcr2.png', dpi=300, bbox_inches='tight')
        plt.close()

    # --- Sample size ----
    if plot_samples:
        f = plt.figure(figsize=(10, 8))

        plt.subplot(1, 1, 1)
        im = plot_img(res, 'len_abs', title='TCA sample size', plot_cb=True, fontsize=fontsize, cmap='jet_r', cbrange=[0, 1000])

        f.savefig(root / 'plots' / 'n_samples.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_result_stats():

    root = Path('/Users/u0116961/Documents/work/LIS/noahmp_version_comparison/')
    if not (root / 'plots').exists():
        Path.mkdir(root / 'plots', parents=True)

    res = pd.read_csv(root / 'result.csv', index_col=0)

    res.loc[res['len_abs']==0,'len_abs'] = np.nan
    res.loc[res['len_anom']==0,'len_anom'] = np.nan

    for col in [x for x in res.columns.values if 'tcr2' in x]:
        res.loc[res['len_abs'] < 300, col] = np.nan
        res.loc[(res[col] < 0)|(res[col] > 1)] = np.nan

    res['tcr2_abs_noahmp_diff'] = res['tcr2_abs_noahmp401'] - res['tcr2_abs_noahmp36']
    res['tcr2_anom_noahmp_diff'] = res['tcr2_anom_noahmp401'] - res['tcr2_anom_noahmp36']

    fontsize = 12
    sns.set_context('talk', font_scale=0.75)

    # Standard statistics
    f = plt.figure(figsize=(22,10))
    params = ['SM1','SM2','SM3','SM4','ST1','ST2','ST3','ST4','LAI','SWE']
    df = res.iloc[:,2:3*len(params)+2].dropna()
    for i, met in enumerate(['mdiff', 'sdiff', 'r2']):
        for j, param in enumerate(params):
            ax = plt.subplot(3,10,i*len(params) + j + 1)
            sns.distplot(df[f'{met}_{param}'])
            plt.xlim(np.percentile(df[f'{met}_{param}'], [1,99]))
            ax.set_yticks([])
            plt.xlabel('')
            if j == 0:
                plt.ylabel(met, fontsize=fontsize)
            if i == 0:
                plt.title(param, fontsize=fontsize)
            if met != 'r2':
                plt.axvline(color='black', linestyle='--', linewidth=1.5)
    f.savefig(root / 'plots' / f'histograms.png', dpi=300, bbox_inches='tight')
    plt.close()

    # TCA stats
    f = plt.figure(figsize=(20, 15))
    params = ['abs_ascat', 'abs_smap', 'anom_ascat', 'anom_smap', 'abs_noahmp36', 'abs_noahmp401', 'anom_noahmp36', 'anom_noahmp401']
    for i, param in enumerate(params):
        ax = plt.subplot(3, 4, i+1)
        sns.distplot(res[f'tcr2_{param}'])
        plt.xlim([0,1])
        ax.set_yticks([])
        plt.xlabel('')
        plt.title(param, fontsize=fontsize)

    for idx, param in zip([9, 11], ['tcr2_abs_noahmp_diff', 'tcr2_anom_noahmp_diff']):
        ax = plt.subplot(3, 4, idx)
        plt.axvline(color='black', linestyle='--', linewidth=1.5)
        sns.distplot(res[param])
        plt.xlim([-0.2, 0.2])
        ax.set_yticks([])
        plt.xlabel('')
        plt.title(param, fontsize=fontsize)

    # plt.show()
    f.savefig(root / 'plots' / f'histograms_tca.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_timeseries():

    # lat, lon, title = 41.509352, -110.254093, 'Wyoming'
    # lat, lon, title = 47.943089, -93.737495, 'Minnesota'
    # lat, lon, title = 39.307215, -116.486053, 'Nevada'
    # lat, lon, title = 30.686218, -100.295540, 'Texas'
    # lat, lon, title = 32.068951, -84.302455, 'Georgia'

    tss = [[41.509352, -110.254093, 'Wyoming'],
           [47.943089, -93.737495, 'Minnesota'],
           [39.307215, -116.486053, 'Nevada'],
           [30.686218, -100.295540, 'Texas'],
           [32.068951, -84.302455, 'Georgia']]

    noah3 = Dataset('/Users/u0116961/data_sets/LIS/noahmp36/timeseries.nc')
    noah4 = Dataset('/Users/u0116961/data_sets/LIS/noahmp401/timeseries.nc')

    lats = noah3['lat'][:, :]
    lons = noah3['lon'][:, :]

    for lat, lon, title in tss:

        i_r, i_c = np.unravel_index(np.argmin((lats - lat) ** 2 + (lons - lon) ** 2), lats.shape)

        time = pd.DatetimeIndex(num2date(noah3['time'][:], units=noah3['time'].units,
                                         only_use_python_datetimes=True, only_use_cftime_datetimes=False))

        df = pd.DataFrame({'time': time,
                           'SM-1 (3.6)': noah3['SM'][:, 0, i_r, i_c],
                           'SM-1 (4.0.1)': noah4['SM'][:, 0, i_r, i_c],
                           'SM-3 (3.6)': noah3['SM'][:, 2, i_r, i_c],
                           'SM-3 (4.0.1)': noah4['SM'][:, 2, i_r, i_c],
                           'ST-1 (3.6)': noah3['ST'][:, 0, i_r, i_c],
                           'ST-1 (4.0.1)': noah4['ST'][:, 0, i_r, i_c],
                           'ST-3 (3.6)': noah3['ST'][:, 2, i_r, i_c],
                           'ST-3 (4.0.1)': noah4['ST'][:, 2, i_r, i_c],
                           'LAI (3.6)': noah3['LAI'][:, i_r, i_c],
                           'LAI (4.0.1)': noah4['LAI'][:, i_r, i_c]})

        sns.set_context('talk', font_scale=0.75, rc={"lines.linewidth": 1})

        f = plt.figure(figsize=(18, 10))

        ax = plt.subplot(5,1,1)
        g = sns.lineplot(x='time', y='SM', hue='Variable', data=df.melt('time', df.columns[1:3], 'Variable', 'SM'), ax=ax)
        plt.legend(loc='upper right')
        g.set(xticklabels=[])
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.title(title)

        ax = plt.subplot(5,1,2)
        g = sns.lineplot(x='time', y='SM', hue='Variable', data=df.melt('time', df.columns[3:5], 'Variable', 'SM'), ax=ax)
        plt.legend(loc='upper right')
        g.set(xticklabels=[])
        ax.set_xlabel('')
        ax.set_ylabel('')

        ax = plt.subplot(5,1,3)
        g = sns.lineplot(x='time', y='ST', hue='Variable', data=df.melt('time', df.columns[5:7], 'Variable', 'ST'), ax=ax)
        plt.legend(loc='upper right')
        g.set(xticklabels=[])
        ax.set_xlabel('')
        ax.set_ylabel('')

        ax = plt.subplot(5,1,4)
        g = sns.lineplot(x='time', y='ST', hue='Variable', data=df.melt('time', df.columns[7:9], 'Variable', 'ST'), ax=ax)
        plt.legend(loc='upper right')
        g.set(xticklabels=[])
        ax.set_xlabel('')
        ax.set_ylabel('')

        ax = plt.subplot(5,1,5)
        g = sns.lineplot(x='time', y='LAI', hue='Variable', data=df.melt('time', df.columns[9::], 'Variable', 'LAI'), ax=ax)
        plt.legend(loc='upper right')
        # g.set(xticks=[], xticklabels=[])
        ax.set_xlabel('')
        ax.set_ylabel('')

        f.savefig(f'/Users/u0116961/Documents/work/LIS/noahmp_version_comparison/plots/ts_{title}.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__=='__main__':

    run()
    # noahmp_version_comparison(5,16)

    plot_result_maps()
    plot_result_stats()
    plot_timeseries()


