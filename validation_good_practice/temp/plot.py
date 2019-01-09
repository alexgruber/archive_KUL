

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pyldas.grids import EASE2

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="7%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, orientation='horizontal')

def plot_ease_img(data,tag,
                  llcrnrlat=24,
                  urcrnrlat=51,
                  llcrnrlon=-128,
                  urcrnrlon=-64,
                  cbrange=(-20,20),
                  cmap='YlGn',
                  plot_cb =True,
                  title='',
                  fontsize=14):

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

    if plot_cb is True:
        # colorbar(im)
        cb = m.colorbar(im, "bottom", size="7%", pad=0.05)
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(fontsize)

    if title != '':
        plt.title(title,fontsize=fontsize)


def plot_ci_test():

    res = pd.read_csv(r'D:\work\validation_good_practice\confidence_invervals\result.csv',index_col=0)

    block_lengths = ['1', '5', '10', '15', '25', '50', 'opt']
    # block_lengths_anom = ['1', '2', '3', '5', '8', '15', 'opt']
    sensors = ['ASCAT','AMSR2','MERRA2']
    modes = ['ci_l', 'p50', 'ci_u']
    metrics = ['r2', 'ubrmse']

    for met in metrics:

        cbrange = [0, 1] if met == 'r2' else [0, 0.25]
        cmap = 'YlGn' if met == 'r2' else 'YlOrRd'
        # ------------------------------------------------------------------------------------------------------------------
        f = plt.figure(figsize=(17,7))

        for i,m in enumerate(['abs', 'anom']):
            for j,s in enumerate(sensors):

                tag = met + '_' + m + '_' + s
                plt.subplot(2, 3, 3 * i + (j + 1))

                title = s if i == 0 else ''
                # plot_cb = True if (i == 1) else False

                plot_ease_img(res, tag, cbrange=cbrange, title = title, plot_cb = True, cmap = cmap)

                if j ==0:
                    plt.text(-0.5e6,1.7e6, m, fontsize=16, rotation=90)

        plt.savefig(r'D:\work\validation_good_practice\confidence_invervals\plots' + '\\' + met + '_abs_anom.png', dpi=f.dpi)
        plt.close()

        # ------------------------------------------------------------------------------------------------------------------
        for bl in block_lengths:
            f = plt.figure(figsize=(17,9))

            for i,m in enumerate(modes):
                for j,s in enumerate(sensors):

                    tag = met + '_abs_' + s + '_' + m + '_bl_' + bl
                    plt.subplot(3, 3, 3 * i + (j + 1))

                    title = s if i == 0 else ''
                    plot_cb = True if (i == 2) else False

                    plot_ease_img(res, tag, cbrange=cbrange, title = title, plot_cb = plot_cb, cmap = cmap)

                    if j ==0:
                        plt.text(-0.5e6,1.7e6, m, fontsize=16, rotation=90)

            plt.savefig(r'D:\work\validation_good_practice\confidence_invervals\plots' + '\\' + met + '_abs_bl_' + bl + '.png', dpi=f.dpi)
            plt.close()

        # ------------------------------------------------------------------------------------------------------------------
        for bl in block_lengths:
            f = plt.figure(figsize=(17,9))

            for i,m in enumerate(modes):
                for j,s in enumerate(sensors):

                    tag = met + '_anom_' + s + '_' + m + '_bl_' + bl
                    plt.subplot(3, 3, 3 * i + (j + 1))

                    title = s if i == 0 else ''
                    plot_cb = True if (i == 2) else False

                    plot_ease_img(res, tag, cbrange=cbrange, title = title, plot_cb = plot_cb, cmap = cmap)

                    if j ==0:
                        plt.text(-0.5e6,1.7e6, m, fontsize=16, rotation=90)

            plt.savefig(r'D:\work\validation_good_practice\confidence_invervals\plots' + '\\' + met + '_anom_bl_' + bl + '.png', dpi=f.dpi)
            plt.close()

def plot_ci_test2():

    res = pd.read_csv(r'D:\work\validation_good_practice\confidence_invervals\result.csv',index_col=0)

    # block_lengths = ['1', '5', '10', '15', '25', '50', 'opt']
    block_lengths = ['1', '10', '25', 'opt']
    sensors = ['ASCAT','AMSR2','MERRA2']

    metrics = ['r2', 'ubrmse']

    for met in metrics:

        cbrange = [0, 1] if met == 'r2' else [0, 0.25]
        cmap = 'YlGn' if met == 'r2' else 'YlOrRd'

        # ------------------------------------------------------------------------------------------------------------------
        for freq in ['abs', 'anom']:

            f = plt.figure(figsize=(20,8))

            for i,s in enumerate(sensors):

                for j, bl in enumerate(block_lengths):

                    tagu = met + '_' + freq + '_' + s + '_ci_u_bl_' + bl
                    tagl = met + '_' + freq + '_' + s + '_ci_l_bl_' + bl
                    tagci = met + '_' + freq + '_' + s + '_ci_bl_' + bl

                    res[tagci] = res[tagu] - res[tagl]

                    plt.subplot(3, len(block_lengths), len(block_lengths) * i + (j + 1))

                    title = 'bl_' + bl if i == 0 else ''
                    plot_cb = True if (i == 2) else False

                    if j == 0:
                        plt.text(-0.5e6, 2.2e6, s, fontsize=16, rotation=90)

                    plot_ease_img(res, tagci, cbrange=cbrange, title = title, plot_cb = plot_cb, cmap = cmap)

            plt.savefig(r'D:\work\validation_good_practice\confidence_invervals\plots' + '\\CI_' + met + '_' + freq + '.png', dpi=f.dpi)
            plt.close()


def plot_ci_test3():

    res = pd.read_csv(r'D:\work\validation_good_practice\confidence_invervals\result.csv',index_col=0)

    sensors = ['ASCAT','AMSR2','MERRA2']
    metrics = ['r2', 'ubrmse']

    for met in metrics:

        cbrange = [0, 0.4] if met == 'r2' else [0, 0.25]
        cmap = 'YlGn' if met == 'r2' else 'YlOrRd'

        # ------------------------------------------------------------------------------------------------------------------
        f = plt.figure(figsize=(12,12))

        for i,s in enumerate(sensors):
            for j,freq in enumerate(['abs', 'anom']):

                tagu = met + '_' + freq + '_' + s + '_ci_u_bl_50'
                tagl = met + '_' + freq + '_' + s + '_ci_l_bl_50'
                tagci50 = met + '_' + freq + '_' + s + '_ci_bl_50'
                res[tagci50] = res[tagu] - res[tagl]

                tagu = met + '_' + freq + '_' + s + '_ci_u_bl_1'
                tagl = met + '_' + freq + '_' + s + '_ci_l_bl_1'
                tagci1 = met + '_' + freq + '_' + s + '_ci_bl_1'
                res[tagci1] = res[tagu] - res[tagl]

                tagcidiff = met + '_' + freq + '_' + s + '_ci_50_minus_ci_1'

                res[tagcidiff] = res[tagci50] - res[tagci1]

                plt.subplot(3, 2, 2 * i + (j + 1))

                title = 'CI_diff_' + freq if i == 0 else ''
                plot_cb = True if (i == 2) else False

                if j == 0:
                    plt.text(-0.5e6, 2.2e6, s, fontsize=16, rotation=90)

                plot_ease_img(res, tagcidiff, cbrange=cbrange, title = title, plot_cb = plot_cb, cmap = cmap)

        plt.savefig(r'D:\work\validation_good_practice\confidence_invervals\plots' + '\\CI_diff_' + met + '.png', dpi=f.dpi)
        plt.close()


if __name__=='__main__':
    plot_ci_test3()


