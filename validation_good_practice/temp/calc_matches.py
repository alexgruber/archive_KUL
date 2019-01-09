
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from pyldas.grids import EASE2

from myprojects.validation_good_practice.data_readers.interface import reader

def calc_matches():

    io = reader()
    result_file = r'D:\work\validation_good_practice\temporal_matches\result.csv'
    lut = pd.read_csv(r"D:\data_sets\EASE2_grid\grid_lut.csv", index_col=0)

    for cnt, (gpi, data) in enumerate(lut.iterrows()):
        print '%i / %i' % (cnt, len(lut))
        try:
            dts = np.arange(0, 24, 0.5)
            N_matched = np.full(len(dts), np.nan)
            for i, dt in enumerate(dts):
                try:
                    df = io.read(gpi, sensors=['ASCAT','AMSR2','MERRA2'], match=True, dt=dt)
                    N_matched[i] = len(df.dropna())
                except:
                    break

            res_dict = dict(zip(['row','col'] + ['lag_%.1f' % dt for dt in dts],
                                [data.ease2_row,data.ease2_col] + N_matched.tolist()))

            result = pd.DataFrame(res_dict, index=(gpi,))
        except:
            continue

        if (os.path.isfile(result_file) == False):
            result.to_csv(result_file, float_format='%0.1f')
        else:
            result.to_csv(result_file, float_format='%0.1f', mode='a', header=False)

def plot_ease_img(data,tag,
                  llcrnrlat=24,
                  urcrnrlat=51,
                  llcrnrlon=-128,
                  urcrnrlon=-64,
                  cbrange=(-20,20),
                  cmap='jet',
                  title='',
                  fontsize=16):

    grid = EASE2()
    lons,lats = np.meshgrid(grid.londim,grid.latdim)

    ind_lat = data['row'].values.astype('int')
    ind_lon = data['col'].values.astype('int')
    img = np.full(lons.shape, np.nan)
    img[ind_lat,ind_lon] = data[tag]
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
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="8%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    if title == '':
        title = tag.lower()
    plt.title(title,fontsize=fontsize)

def plot_matches():

    res = pd.read_csv(r"D:\work\validation_good_practice\temporal_matches\result.csv", index_col=0)

    res = res.drop(['row', 'col'], axis='columns').replace(0,np.nan).dropna()

    # lags = np.array([col[4::] for col in res.columns.values]).astype('float')
    # df = pd.DataFrame({'average # matches': res.mean(axis='index').values}, index=lags)
    # ax = df.sort_index().plot()
    # ax.set_xlim(-1,24)
    # ax.set_xlabel('window center [hour of day]')

    w = 301
    ratio = res.max(axis='columns') / res.min(axis='columns').sort_index()
    ratio.iloc[:] = np.convolve(ratio, np.ones(w) / float(w))[w/2:-(w/2)]
    ratio.name='Max / Min [# matches]'
    ax = pd.DataFrame(ratio.iloc[w/2:-(w/2)]).plot()
    ax.set_xlabel('GPI')
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(20, 9))
    # cbrange = [0, 1400]
    # sensors = ['AMSR2', 'SMOS', 'SMAP', 'ASCAT', 'MERRA2']
    # tags = ['N_' + s for s in sensors]
    # for i, tag in enumerate(tags):
    #     plt.subplot(2, 3, i + 1)
    #     plot_ease_img(res, tag, cbrange=cbrange, title=sensors[i])
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure(figsize=(14, 5))
    # cbrange = [0, 400]
    # tags = ['N_matched_min', 'N_matched_max']
    # for i, tag in enumerate(tags):
    #     plt.subplot(1, 2, i + 1)
    #     plot_ease_img(res.dropna(), tag, cbrange=cbrange, title=tag)
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure(figsize=(8, 5))
    # cbrange = [0, 15]
    # tags = ['dt',]
    # for i, tag in enumerate(tags):
    #     plot_ease_img(res.dropna(), tag, cbrange=cbrange, title='window center [hr]')
    # plt.tight_layout()
    # plt.show()


if __name__=='__main__':
    plot_matches()