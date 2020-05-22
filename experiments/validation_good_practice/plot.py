

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
                  fontsize=12):

    grid = EASE2()
    lons,lats = np.meshgrid(grid.ease_lons, grid.ease_lats)

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
        # cb = m.colorbar(im, "bottom", size="7%", pad=0.05, ticks=[0,1,2])
        # cb.ax.set_xticklabels(['ASC','no sig. diff', 'AMS'])
        cb = m.colorbar(im, "bottom", size="7%", pad=0.05)
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(fontsize)

    if title != '':
        plt.title(title,fontsize=fontsize)

def plot_matches():

    res = pd.read_csv(r"D:\work\validation_good_practice\confidence_invervals\result.csv",index_col=0)

    plt.figure(figsize=(14,8))

    plt.subplot(221)
    plot_ease_img(res, 'n_abs', cbrange=[0,800], title='# matches (abs)', cmap='jet')

    plt.subplot(222)
    plot_ease_img(res, 'n_anom', cbrange=[0,800], title='# matches (anom)', cmap='jet')

    plt.subplot(223)
    plot_ease_img(res, 'dt_opt', cbrange=[0,24], title='matching window center (abs)', cmap='jet')

    plt.subplot(224)
    plot_ease_img(res, 'dt_opt_anom', cbrange=[0,24], title='matching window center (anom)', cmap='jet')

    plt.tight_layout()
    plt.show()

def plot_blocklength():

    res = pd.read_csv(r"D:\work\validation_good_practice\confidence_invervals\result.csv",index_col=0)

    plt.figure(figsize=(14,5))

    plt.subplot(121)
    plot_ease_img(res, 'bl_opt', cbrange=[0,100], title='Blocklength (abs)', cmap='jet')

    plt.subplot(122)
    plot_ease_img(res, 'bl_opt_anom', cbrange=[0,16], title='Blocklength (anom)', cmap='jet')

    plt.tight_layout()
    plt.show()

def plot_ci_l_50_u():

    res = pd.read_csv(r'D:\work\validation_good_practice\confidence_invervals\result.csv',index_col=0)

    block_lengths = ['1', '10', '25', '50', 'opt']
    block_lengths_anom = ['1', '5', '15', '25', 'opt']

    sensors = ['ASCAT','AMSR2','MERRA2']
    modes = ['ci_l', 'p50', 'ci_u']
    # metrics = ['r2', 'ubrmse']
    metrics = ['ubrmse',]

    for met in metrics:

        cbrange = [0, 1] if met == 'r2' else [0, 0.08]
        # cmap = 'YlGn' if met == 'r2' else 'YlOrRd'
        cmap = 'RdYlGn' if met == 'r2' else 'RdYlGn_r'
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
        for bl in block_lengths_anom:
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

def plot_ci_width():

    res = pd.read_csv(r"D:\work\validation_good_practice\confidence_invervals\result.csv",index_col=0)

    sensors = ['ASCAT','AMSR2','MERRA2']
    block_lengths = ['1', '25', '50', 'opt']
    block_lengths_anom = ['1', '15', '25', 'opt']

    metrics = ['r2', 'ubrmse']

    for met in metrics:

        cbrange = [0, 1] if met == 'r2' else [0, 0.25]
        cmap = 'YlGn' if met == 'r2' else 'YlOrRd'

        # ------------------------------------------------------------------------------------------------------------------
        for freq in ['abs', 'anom']:

            f = plt.figure(figsize=(20,8))

            for i,s in enumerate(sensors):

                lengths = block_lengths if freq == 'abs' else block_lengths_anom
                for j, bl in enumerate(lengths):

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

def plot_ci_width_hist():

    res = pd.read_csv(r'D:\work\validation_good_practice\confidence_invervals\result.csv',index_col=0)

    sensors = ['ASCAT','AMSR2','MERRA2']
    block_lengths = ['1', '10', '25', 'opt']
    block_lengths_anom = ['1', '5', '15', 'opt']


    metrics = ['r2', 'ubrmse']

    for met in metrics:

        cbrange = [0, 1] if met == 'r2' else [0, 0.1]
        fontsize = 14

        # ------------------------------------------------------------------------------------------------------------------
        for freq in ['abs', 'anom']:

            f = plt.figure(figsize=(20,8))

            for i,s in enumerate(sensors):

                lengths = block_lengths if freq == 'abs' else block_lengths_anom
                for j, bl in enumerate(lengths):

                    tagu = met + '_' + freq + '_' + s + '_ci_u_bl_' + bl
                    tagl = met + '_' + freq + '_' + s + '_ci_l_bl_' + bl
                    tagci = met + '_' + freq + '_' + s + '_ci_bl_' + bl

                    res[tagci] = res[tagu] - res[tagl]

                    plt.subplot(3, len(block_lengths), len(block_lengths) * i + (j + 1))

                    if j == 0:
                        plt.text(-0.5e6, 2.2e6, s, fontsize=16, rotation=90)

                    h = res[tagci].hist(bins=20, range=cbrange, density=True)
                    for t in h.get_xticklabels():
                        t.set_fontsize(fontsize-2)
                    for t in h.get_yticklabels():
                        t.set_fontsize(fontsize-2)
                    if i == 0:
                        plt.title('bl_' + bl, fontsize=fontsize)
                    if j == 0:
                        plt.ylabel(s, fontsize=fontsize)

            plt.savefig(r'D:\work\validation_good_practice\confidence_invervals\plots' + '\\CI_hist_' + met + '_' + freq + '.png', dpi=f.dpi)
            plt.close()


def plot_ci_width_bl_dependence():

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

                bl = '50' if freq == 'abs' else '25'

                tagu = met + '_' + freq + '_' + s + '_ci_u_bl_' + bl
                tagl = met + '_' + freq + '_' + s + '_ci_l_bl_' + bl
                tagci50 = met + '_' + freq + '_' + s + '_ci_bl_' + bl
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


def plot_significant_differences():

    res = pd.read_csv(r"D:\work\validation_good_practice\confidence_invervals\result.csv",index_col=0)

    block_lengths = ['1', '10', '25', 'opt']
    block_lengths_anom = ['1', '5', '15', 'opt']

    for met in ['r2', 'ubrmse']:

        f = plt.figure(figsize=(13,4))

        for i, freq in enumerate(['abs', 'anom']):

            lengths = block_lengths if freq == 'abs' else block_lengths_anom
            for j, bl in enumerate(lengths):

                tagu_asc = met + '_' + freq + '_ASCAT_ci_u_bl_' + bl
                tagl_asc = met + '_' + freq + '_ASCAT_ci_l_bl_' + bl

                tagu_ams = met + '_' + freq + '_AMSR2_ci_u_bl_' + bl
                tagl_ams = met + '_' + freq + '_AMSR2_ci_l_bl_' + bl

                res['sig'] = np.nan

                if met == 'r2':
                    res.loc[res[tagu_ams] <= res[tagl_asc],'sig'] = 0
                    res.loc[(res[tagu_ams] > res[tagl_asc])&(res[tagu_asc] > res[tagl_ams]),'sig'] = 1
                    res.loc[res[tagu_asc] <= res[tagl_ams],'sig'] = 2
                else:
                    res.loc[res[tagu_ams] <= res[tagl_asc],'sig'] = 2
                    res.loc[(res[tagu_ams] > res[tagl_asc])&(res[tagu_asc] > res[tagl_ams]),'sig'] = 1
                    res.loc[res[tagu_asc] <= res[tagl_ams],'sig'] = 1

                plt.subplot(2, 4, 4 * i + (j + 1))
                plt.xticks([0,1,2],[0,1,2])

                title = 'bl_' + bl if i == 0 else ''
                plot_cb = True if (i == 1) else False

                if j == 0:
                    plt.text(-0.7e6, 2.1e6, freq, fontsize=16, rotation=90)

                plot_ease_img(res, 'sig', cbrange=[0, 2], title=title, plot_cb=plot_cb, cmap='brg')

        plt.savefig(r'D:\work\validation_good_practice\confidence_invervals\significant_differences' + '\\' + met + '.png', dpi=f.dpi)
        plt.close()

# def validation_protocol_graphic:
    # def calc_anom(Ser, longterm=False, window_size=35):
    #
    #     xSer = Ser.dropna().copy()
    #     if len(xSer) == 0:
    #         return xSer
    #
    #     doys = xSer.index.dayofyear.values
    #     doys[xSer.index.is_leap_year & (doys > 59)] -= 1
    #     climSer = pd.Series(index=xSer.index, name=xSer.name)
    #
    #     if longterm is True:
    #         climSer[:] = calc_clim(xSer, window_size=window_size)[doys]
    #     else:
    #         years = xSer.index.year
    #         for yr in np.unique(years):
    #             clim = calc_clim(xSer[years == yr], window_size=window_size)
    #             climSer[years == yr] = clim[doys[years == yr]].values
    #
    #     return xSer - climSer, climSer

    # grid = pd.read_csv('/data_sets/EASE2_grid/grid_lut.csv', index_col=0)
    #
    #
    # lon = -101.139783
    # lat = 38.322267
    #
    # gpi = grid.index[np.argmin((grid.ease2_lon.values - lon) ** 2 + (grid.ease2_lat.values - lat) ** 2)]

    # sensors = ['SMAP','ISMN',]
    # io = reader()
    # data = io.read(83065)

    # data['SMAP'] = calc_anom(data['SMAP'], window_size=5)[1]
    # data /= 2
    # data['SMAP'] += 0.07
    # data['ISMN'] /= 1.5
    # data['ISMN_scl'] = (data['ISMN'] - data['ISMN'].mean()) / data['ISMN'].std() * data['SMAP'].std() + data['SMAP'].mean()
    # plt.figure(figsize=(15,5))
    # plt.plot(data.index, data['SMAP'], color='green', linewidth=3)
    # plt.plot(data.index, data['ISMN'], color='red', linewidth=3)
    # plt.plot(data.index, data['ISMN_scl'], ':', color='red', linewidth=4)
    # plt.show()

    # data['SMAP'] = calc_anom(data['SMAP'], window_size=3)[1]
    # data /= 2
    # data['SMAP'] += 0.07
    #
    # data['SMAP_anom'], data['SMAP_clim'] = calc_anom(data['SMAP'], window_size=40)
    #
    # plt.figure(figsize=(15,10))
    #
    # plt.subplot(2,1,1)
    # plt.plot(data.index, data['SMAP'], color='green', linewidth=4)
    # # plt.axhline(color='black', linestyle='--', linewidth=1)
    # # plt.ylim((-0.05, 0.23))
    #
    #
    # plt.subplot(2,1,2)
    # plt.plot(data.index, data['SMAP_clim'], color='green', linewidth=4)
    # plt.plot(data.index, data['SMAP_anom'], color='green', linewidth=4)
    # # plt.axhline(color='black', linestyle='--', linewidth=1)
    # # plt.ylim((-0.05, 0.23))
    #
    # plt.tight_layout()
    # plt.show()
    #


if __name__=='__main__':
    # plot_ci_l_50_u()
    # plot_ci_width()
    # plot_ci_width_hist()
    # plot_ci_width_bl_dependence()
    # plot_significant_differences()
    plot_blocklength()
