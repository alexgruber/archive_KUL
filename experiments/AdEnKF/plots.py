
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D

from myprojects.readers.mswep import MSWEP_io

def plot_figure(img, lons, lats,
                llcrnrlat=24,
                urcrnrlat=51,
                llcrnrlon=-128,
                urcrnrlon=-64,
                cbrange=(0,1),
                cmap='jet'):

    m = Basemap(projection='mill',
                    llcrnrlat=llcrnrlat,
                    urcrnrlat=urcrnrlat,
                    llcrnrlon=llcrnrlon,
                    urcrnrlon=urcrnrlon,
                    resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    m.colorbar(im, "bottom", size="7%", pad="4%")


def plot_result():

    res = pd.read_csv(r"D:\work\API\AdEnKF\result.csv", index_col=0)

    io = MSWEP_io()
    lats = io.ds['lat'][:]
    lons = io.ds['lon'][:]
    lons, lats = np.meshgrid(lons, lats)

    rows = res['row'].values.astype('int')
    cols = res['col'].values.astype('int')


    figsize = (17, 10)
    plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')


    param = 'Q'
    cbrange = [0, 10]
    cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows,cols)
    img[ind] = np.sqrt(res[param].values)
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(221)
    plot_figure(img_masked,lons,lats,cbrange=cbrange,cmap=cmap)
    plt.title('sqrt('+param+')', fontsize=20)


    param = 'R'
    cbrange = [0, 16]
    cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] = np.sqrt(res[param].values)
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(222)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap)
    plt.title('sqrt('+param+')', fontsize=20)


    param = 'checkvar'
    cbrange = [0.6, 1.4]
    cmap = 'RdYlBu'
    img = np.full(lats.shape, np.nan)
    ind = (rows,cols)
    img[ind] = res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(223)
    plot_figure(img_masked,lons,lats,cbrange=cbrange,cmap=cmap)
    plt.title('innov_var', fontsize=20)


    plt.tight_layout()
    plt.show()


def plot_syn_result():

    res = pd.read_csv(r"D:\work\API\MadEnKF\synthetic_experiment\result.csv", index_col=0)

    plt.figure(figsize=(18, 9))

    iterations = [5, 7, 9, 11, 13, 15]

    ensembles = [10, 30, 50, 70]
    legend = ['%i' % ens for ens in ensembles]

    offsets = [-0.3,-0.1,0.1,0.3]
    cols = ['lightblue', 'lightgreen', 'coral', 'brown']
    fontsize = 10


    ax = plt.subplot(221)
    plt.grid(color='k', linestyle='--', linewidth=0.25)
    data = list()
    ticks = list()
    pos = list()
    colors = list()
    for i, iters in enumerate(iterations):
        ticks.append(iters)
        for col, offs, ens in zip(cols, offsets, ensembles):
            ind = (res['n_ens'] == ens) & (res['n_iter'] == iters)
            tmp_data = res.loc[ind,'R_est'] / res.loc[ind,'R_true']
            data.append(tmp_data)
            pos.append(i + 1 + offs)
            colors.append(col)
    box = ax.boxplot(data, whis=[5, 95], showfliers=False, positions=pos, widths=0.1, patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set(color='black', linewidth=2)
        patch.set_facecolor(color)
    for patch in box['medians']:
        patch.set(color='black', linewidth=2)
    for patch in box['whiskers']:
        patch.set(color='black', linewidth=1)
    plt.figlegend((box['boxes'][0:4]), legend, 'upper left', fontsize=fontsize)
    plt.xticks(np.arange(len(iterations)) + 1, ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0.5, len(ticks) + 0.5)
    plt.ylim(0.0, 3.0)
    for i in np.arange(len(iterations)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    ax.set_title('R (est) / R (true)', fontsize=fontsize+2)

    ax = plt.subplot(222)
    plt.grid(color='k', linestyle='--', linewidth=0.25)
    data = list()
    ticks = list()
    pos = list()
    colors = list()
    for i, iters in enumerate(iterations):
        ticks.append(iters)
        for col, offs, ens in zip(cols, offsets, ensembles):
            ind = (res['n_ens'] == ens) & (res['n_iter'] == iters)
            tmp_data = res.loc[ind,'Q_est'] / res.loc[ind,'Q_true']
            data.append(tmp_data)
            pos.append(i + 1 + offs)
            colors.append(col)
    box = ax.boxplot(data, whis=[5, 95], showfliers=False, positions=pos, widths=0.1, patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set(color='black', linewidth=2)
        patch.set_facecolor(color)
    for patch in box['medians']:
        patch.set(color='black', linewidth=2)
    for patch in box['whiskers']:
        patch.set(color='black', linewidth=1)
    plt.figlegend((box['boxes'][0:4]), legend, 'upper left', fontsize=fontsize)
    plt.xticks(np.arange(len(iterations)) + 1, ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0.5, len(ticks) + 0.5)
    plt.ylim(0.0, 3.0)
    for i in np.arange(len(iterations)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    ax.set_title('Q (est) / Q (true)', fontsize=fontsize+2)


    ax = plt.subplot(223)
    plt.grid(color='k', linestyle='--', linewidth=0.25)
    data = list()
    ticks = list()
    pos = list()
    colors = list()
    for i, iters in enumerate(iterations):
        ticks.append(iters)
        for col, offs, ens in zip(cols, offsets, ensembles):
            ind = (res['n_ens'] == ens) & (res['n_iter'] == iters)
            tmp_data = res.loc[ind,'H_est'] * res.loc[ind,'H_true']
            data.append(tmp_data)
            pos.append(i + 1 + offs)
            colors.append(col)
    box = ax.boxplot(data, whis=[5, 95], showfliers=False, positions=pos, widths=0.1, patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set(color='black', linewidth=2)
        patch.set_facecolor(color)
    for patch in box['medians']:
        patch.set(color='black', linewidth=2)
    for patch in box['whiskers']:
        patch.set(color='black', linewidth=1)
    plt.figlegend((box['boxes'][0:4]), legend, 'upper left', fontsize=fontsize)
    plt.xticks(np.arange(len(iterations)) + 1, ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0.5, len(ticks) + 0.5)
    plt.ylim(0.0, 2.0)
    for i in np.arange(len(iterations)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    ax.set_title('H (est) / H (true)', fontsize=fontsize+2)


    ax = plt.subplot(224)
    plt.grid(color='k', linestyle='--', linewidth=0.25)
    data = list()
    ticks = list()
    pos = list()
    colors = list()
    for i, iters in enumerate(iterations):
        ticks.append(iters)
        for col, offs, ens in zip(cols, offsets, ensembles):
            ind = (res['n_ens']==ens)&(res['n_iter']==iters)
            tmp_data = res.loc[ind,'P_ana_true']
            data.append(tmp_data)
            pos.append(i + 1 + offs)
            colors.append(col)
    box = ax.boxplot(data, whis=[5, 95], showfliers=False, positions=pos, widths=0.1, patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set(color='black', linewidth=2)
        patch.set_facecolor(color)
    for patch in box['medians']:
        patch.set(color='black', linewidth=2)
    for patch in box['whiskers']:
        patch.set(color='black', linewidth=1)
    plt.figlegend((box['boxes'][0:4]), legend, 'upper left', fontsize=fontsize)
    plt.xticks(np.arange(len(iterations)) + 1, ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0.5, len(ticks) + 0.5)
    plt.ylim(0.0, 60.0)
    for i in np.arange(len(iterations)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    ax.set_title('Analysis RMSE', fontsize=fontsize+2)


    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    plot_syn_result()