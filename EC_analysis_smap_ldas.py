
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

from pyldas.interface import LDAS_io
from myprojects.readers.insitu import ISMN_io
from myprojects.readers.ascat import HSAF_io

from myprojects.timeseries import calc_anom
from myprojects.functions import merge_files

from validation_good_practice.ancillary.paths import Paths
from validation_good_practice.plots import plot_ease_img

from pytesmo.metrics import ecol


class SMAP_io(object):

    def __init__(self):
        self.path = Path('/Users/u0116961/data_sets/SMAP/timeseries')

    def read(self, gpi):

        if (fname := self.path / f'{gpi}.csv').exists():
            ts_smap = pd.read_csv(fname,
                        index_col=0, parse_dates=True, names=('smap',))['smap'].resample('1d').mean().dropna()
        else:
            ts_smap = None

        return ts_smap

def EC_ascat_smap_ismn_ldas():

    result_file = Path('/Users/u0116961/Documents/work/extended_collocation/ec_ascat_smap_ismn_ldas.csv')

    names = ['insitu', 'ascat', 'smap', 'ol', 'da']
    combs = list(combinations(names, 2))

    ds_ol = LDAS_io('xhourly', 'US_M36_SMAP_TB_OL_noScl').timeseries
    ds_da = LDAS_io('xhourly', 'US_M36_SMAP_TB_MadKF_DA_it11').timeseries
    ds_da_ana = LDAS_io('ObsFcstAna', 'US_M36_SMAP_TB_MadKF_DA_it11').timeseries['obs_ana']
    tg = LDAS_io().grid.tilegrids

    modes = ['absolute','longterm','shortterm']

    ismn = ISMN_io()
    ismn.list = ismn.list.iloc[70::]
    ascat = HSAF_io()
    smap = SMAP_io()

    lut = pd.read_csv(Paths().lut, index_col=0)

    i = 0
    for meta, ts_insitu in ismn.iter_stations(surface_only=True):
        i += 1
        logging.info('%i/%i' % (i, len(ismn.list)))

        try:
            if len(ts_insitu := ts_insitu['2015-04-01':'2020-04-01'].resample('1d').mean().dropna()) < 25:
                continue
        except:
            continue

        res = pd.DataFrame(meta.copy()).transpose()
        col = meta.ease_col
        row = meta.ease_row

        colg = col + tg.loc['domain', 'i_offg']  # col / lon
        rowg = row + tg.loc['domain', 'j_offg']  # row / lat

        tmp_lut = lut[(lut.ease2_col == colg) & (lut.ease2_row == rowg)]
        if len(tmp_lut) == 0:
            continue

        gpi_smap = tmp_lut.index.values[0]
        gpi_ascat = tmp_lut.ascat_gpi.values[0]

        try:
            ts_ascat = ascat.read(gpi_ascat, resample_time=False).resample('1d').mean().dropna()
            ts_ascat = ts_ascat[~ts_ascat.index.duplicated(keep='first')]
            ts_ascat.name = 'ASCAT'
        except:
            continue

        ts_smap = smap.read(gpi_smap)

        if (ts_ascat is None) | (ts_smap is None):
            continue

        ind = (ds_ol['snow_mass'][:, row, col].values == 0)&(ds_ol['soil_temp_layer1'][:, row, col].values > 277.15)
        ts_ol = ds_ol['sm_surface'][:, row, col].to_series().loc[ind].dropna()
        ts_ol.index += pd.to_timedelta('2 hours')

        ind = (ds_da['snow_mass'][:, row, col].values == 0)&(ds_da['soil_temp_layer1'][:, row, col].values > 277.15)
        ts_da = ds_da['sm_surface'][:, row, col].to_series().loc[ind].dropna()
        ts_da.index += pd.to_timedelta('2 hours')

        for mode in modes:

            if mode == 'absolute':
                ts_ins = ts_insitu.copy()
                ts_asc = ts_ascat.copy()
                ts_smp = ts_smap.copy()
                ts_ol = ts_ol.copy()
                ts_da = ts_da.copy()
            else:
                ts_ins = calc_anom(ts_ins.copy(), longterm=(mode=='longterm')).dropna()
                ts_asc = calc_anom(ts_asc.copy(), longterm=(mode == 'longterm')).dropna()
                ts_smp = calc_anom(ts_smp.copy(), longterm=(mode == 'longterm')).dropna()
                ts_ol = calc_anom(ts_ol.copy(), longterm=(mode == 'longterm')).dropna()
                ts_da = calc_anom(ts_da.copy(), longterm=(mode == 'longterm')).dropna()

            tmp = pd.DataFrame(dict(zip(names, [ts_ins, ts_asc, ts_smp, ts_ol, ts_da]))).dropna()

            corr = tmp.corr()
            ec_res = ecol(tmp[['insitu', 'ascat', 'smap', 'ol', 'da']], correlated=[['smap', 'ol'], ['smap', 'da'], ['ol', 'da']])

            res[f'len_{mode}'] = len(tmp)
            for c in combs:
                res[f'corr_{"_".join(c)}'] = corr.loc[c]
            res[f'err_corr_smap_ol_{mode}'] = ec_res['err_corr_smap_ol']
            res[f'err_corr_smap_da_{mode}'] = ec_res['err_corr_smap_da']
            res[f'err_corr_ol_da_{mode}'] = ec_res['err_corr_ol_da']

        if not result_file.exists():
            res.to_csv(result_file, float_format='%0.4f')
        else:
            res.to_csv(result_file, float_format='%0.4f', mode='a', header=False)


def plot_result_stats():

    n_min = 1000

    fout = '/Users/u0116961/Documents/work/extended_collocation/EC_stats.png'

    res = pd.read_csv('/Users/u0116961/Documents/work/extended_collocation/ec_ascat_smap_ismn_ldas.csv', index_col=0)

    # mask out all values below a certain sample size and with negative correlation, or meaninglase correlation ranges
    for col in [x for x in res.columns if x.split('_')[0] == 'len']:
        res.loc[res[col]<200,:] = np.nan
    for col in [x for x in res.columns if x.split('_')[0] == 'corr']:
        res.loc[res[col]<0.2,:] = np.nan
    for col in [x for x in res.columns if x.split('_')[0] == 'err_corr']:
        res.loc[(res[col]<=-1.0) | (res[col]>=1.0),:] = np.nan

    cols = ['smap_ol', 'smap_da', 'ol_da']
    modes = ['absolute', 'longterm', 'shortterm']

    res = res[[f'err_corr_{col}_{mode}' for mode in modes for col in cols]].dropna()
    print(len(res))

    # extract information about individual columns (which experiment, surface/root zone, r / rmsd, ...)
    col = np.array(['_'.join(x.split('_')[-3:-1]) for x in res.columns])
    mode = np.array([x.split('_')[-1] for x in res.columns])

    # Prepare data frame for seaborn plotting
    res.columns = [col, mode]
    res = res.melt(var_name=['sensor', 'mode'], value_name='val')

    ylim = (-1,1)

    sns.set_context('talk', font_scale=0.8)
    g = sns.catplot(x='sensor', y='val', data=res, col='mode', kind='box', sharey=False, col_wrap=3)
    [ax.set(ylim=ylim) for ax in g.axes]
    [ax.axhline(color='black', linestyle='--', linewidth=1.5)  for ax in g.axes]
    g.set_titles('{col_name}')
    g.set_ylabels('')
    g.set_xlabels('')

    # plt.show()

    g.savefig(fout, dpi=200, bbox_inches='tight')
    plt.close()


def plot_result_map():

    fout = '/Users/u0116961/Documents/work/extended_collocation/EC_maps.png'

    res = pd.read_csv('/Users/u0116961/Documents/work/extended_collocation/ec_ascat_smap_ismn_ldas.csv', index_col=0)

    # mask out all values below a certain sample size and with negative correlation, or meaninglase correlation ranges
    for col in [x for x in res.columns if x.split('_')[0] == 'len']:
        res.loc[res[col]<200,:] = np.nan
    for col in [x for x in res.columns if x.split('_')[0] == 'corr']:
        res.loc[res[col]<0.2,:] = np.nan
    for col in [x for x in res.columns if x.split('_')[0] == 'err_corr']:
        res.loc[(res[col]<=-1.0) | (res[col]>=1.0),:] = np.nan

    cols = ['smap_ol', 'smap_da', 'ol_da']
    modes = ['absolute', 'longterm', 'shortterm']

    res = res[['lat', 'lon'] + [f'err_corr_{col}_{mode}' for mode in modes for col in cols]].dropna()
    print(len(res))

    lats = res['lat'].values
    lons = res['lon'].values

    vmin = -1.0
    vmax = 1.0

    marker_size = 45
    cmap='seismic_r'

    llcrnrlat = 24
    urcrnrlat = 51
    llcrnrlon = -128
    urcrnrlon = -64

    f = plt.figure(figsize=(18, 9))

    for j, mod in enumerate(modes):
        for i, col in enumerate(cols):

            ax = plt.subplot(len(cols), len(modes), j * len(cols) + i + 1)
            m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                        resolution='c')
            m.drawcoastlines()
            m.drawcountries()
            m.drawstates()
            x, y = m(lons, lats)
            sc = ax.scatter(x, y, s=marker_size, c=res[f'err_corr_{col}_{mod}'], marker='o', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f'{col} / {mod}')


    f.subplots_adjust(wspace=0.04, hspace=0.025, bottom=0.06)
    pos = f.axes[-2].get_position()
    x1 = pos.x0
    x2 = pos.x1
    cbar_ax = f.add_axes([x1, 0.03, x2 - x1, 0.03])
    cbar = f.colorbar(sc, orientation='horizontal', cax=cbar_ax)
    for t in cbar.ax.get_xticklabels():
        t.set_fontsize(10)

    f.savefig(fout, dpi=200, bbox_inches='tight')
    plt.close()

    # plt.show()

if __name__=='__main__':

    # EC_ascat_smap_ismn_ldas()

    # plot_result_stats()
    plot_result_map()


