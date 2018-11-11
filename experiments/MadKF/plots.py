
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

    med = np.percentile(img[np.where(~np.isnan(img))],50)
    x, y = m(-78.5,25)
    plt.text(x,y,'median = %.2f' % med ,fontsize=16)

def plot_insitu_eval_results():

    res = pd.read_csv(r"D:\work\MadKF\API\CONUS\ismn_eval\result.csv", index_col=0)

    res = res[(res.network == 'SCAN') | (res.network == 'USCRN')]

    cols = ['corr_ol', 'corr_avg', 'corr_rmsd', 'corr_kf', 'corr_madkf']
    ylim = [0.0, 0.7]

    # cols = ['rmse_ol','rmse_kf', 'rmse_avg', 'rmse_rmsd', 'rmse_madkf']
    # ylim = [0, 0.04]

    res = res[res.n_all>50][cols]
    cols = ['OL', 'const. err', 'rmsd', 'TC', 'MadKF']

    print len(res)

    tmp_data = [x[1].values for x in res.iteritems()]
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111)

    ax.yaxis.grid(True, linestyle='--', which='major', color='lightgrey',
                   alpha=0.7)

    box = ax.boxplot(tmp_data, whis=[5, 95], showfliers=False, widths=0.2, patch_artist=True, boxprops=dict(alpha=.7))
    # res.boxplot(showfliers=False, whis=[5,95])

    for patch in box['boxes']:
        patch.set(color='black', linewidth=2)
        patch.set_facecolor('orange')
    for patch in box['medians']:
        patch.set(color='black', linewidth=2)
    for patch in box['whiskers']:
        patch.set(color='black', linewidth=1)

    plt.ylim(ylim)
    plt.xticks(np.arange(len(cols))+1, cols, rotation=40, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Correlation', fontsize = 12)

    for i in np.arange(4)+1.5:
        plt.axvline(i, linewidth=1, linestyle='--', color='lightgrey', alpha=0.7)

    plt.title('SCAN + USCRN', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_result2():
    res = pd.read_csv(r"D:\work\MadKF\API\CONUS\result.csv", index_col=0)

    io = MSWEP_io(cellfiles = False)
    lats = io.ds['lat'][:]
    lons = io.ds['lon'][:]
    lons, lats = np.meshgrid(lons, lats)

    rows = res['row'].values.astype('int')
    cols = res['col'].values.astype('int')

    figsize = (17, 10)
    plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')

    # cbrange = [0,1]

    param = 'P_tc'
    # cbrange = cbrange
    cbrange = [0, 20]
    cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] = np.sqrt(res[param].values)
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(221)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap)
    plt.title('sqrt(P_TC)', fontsize=16)

    param = 'R_tc'
    # cbrange = cbrange
    cbrange = [0, 20]
    cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] =  np.sqrt(res[param].values) * res['H_tc'].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(222)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap)
    plt.title('sqrt(R_TC)', fontsize=16)

    param = 'P_madkf'
    cbrange = [0, 20]
    # cbrange = cbrange
    cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] = np.sqrt(res[param].values)
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(223)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap)
    plt.title('sqrt(P_madkf)', fontsize=16)

    param = 'R_madkf'
    cbrange = [0, 20]
    cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] = np.sqrt(res[param].values) * res['H_madkf'].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(224)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap)
    plt.title('sqrt(r_madkf)', fontsize=20)

    plt.tight_layout()
    plt.show()


def plot_result():

    res = pd.read_csv(r"D:\work\MadKF\API\CONUS\result.csv", index_col=0)

    io = MSWEP_io(cellfiles = False)
    lats = io.ds['lat'][:]
    lons = io.ds['lon'][:]
    lons, lats = np.meshgrid(lons, lats)

    rows = res['row'].values.astype('int')
    cols = res['col'].values.astype('int')

    figsize = (17, 10)
    plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')

    # n_obs = 1461.

    cmap = 'seismic_r'
    cbrange = [0, 2]

    param = 'checkvar1_2d'
    cbrange = cbrange
    # cbrange = [0, 150]
    cmap = cmap
    # cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows,cols)
    img[ind] = res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(221)
    plot_figure(img_masked,lons,lats,cbrange=cbrange,cmap=cmap)
    plt.title('Innovation variance KF (MOD - ASC)', fontsize=20)


    param = 'checkvar2_2d'
    cbrange = cbrange
    # cbrange = [0, 150]
    cmap = cmap
    # cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] = res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(222)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap)
    plt.title('Innovation variance KF (MOD - SMO)', fontsize=20)

    param = 'checkvar3_2d'
    cbrange = cbrange
    # cbrange = [0, 150]
    cmap = cmap
    # cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] = res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(223)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap)
    plt.title('Innovation variance KF (4*MOD - ASC - SMO)', fontsize=20)

    param = 'checkvar_madkf'
    cbrange = cbrange
    # cbrange = [0, 150]
    cmap = cmap
    # cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows,cols)
    img[ind] = res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(224)
    plot_figure(img_masked,lons,lats,cbrange=cbrange,cmap=cmap)
    plt.title('Innovation variance MadKF', fontsize=20)


    plt.tight_layout()
    plt.show()

def plot_syn_result2():

    res = pd.read_csv(r"D:\work\MadKF\API\synthetic_experiment\result.csv", index_col=0)

    ticks = ['R(RMSD) / R(true)', 'R(MadKF) / R(true)',
             'P(true) / P(true)', 'P(RMSD) / P(true)', 'P(MadKF) / P(true)',
             'innov_var(true)', 'innov_var(RMSD)', 'innov_var(MadKF)',
             'corr(OL)', 'corr(obs)', 'corr(true)', 'corr(RMSD)', 'corr(MadKF)']

    colors = ['lightblue', 'lightblue',
              'lightgreen', 'lightgreen', 'lightgreen',
              'coral', 'coral', 'coral',
              'brown', 'brown', 'brown', 'brown', 'brown']

    data = list()

    data.append(res.loc[(res['n_ens']==50)&(res['n_iter']==5),'R_est_rmsd'] / res.loc[(res['n_ens']==50)&(res['n_iter']==5),'R_true'])
    data.append(res.loc[(res['n_ens']==50)&(res['n_iter']==13),'R_est_madkf'] / res.loc[(res['n_ens']==50)&(res['n_iter']==13),'R_true'])

    data.append(res.loc[(res['n_ens']==50)&(res['n_iter']==5),'P_ana_est_enkf_true'] / res.loc[(res['n_ens']==50)&(res['n_iter']==5),'P_ana_true_enkf_true'])
    data.append(res.loc[(res['n_ens']==50)&(res['n_iter']==5),'P_ana_est_enkf_rmsd'] / res.loc[(res['n_ens']==50)&(res['n_iter']==5),'P_ana_true_enkf_true'])
    data.append(res.loc[(res['n_ens']==50)&(res['n_iter']==13),'P_ana_est_madkf'] / res.loc[(res['n_ens']==50)&(res['n_iter']==13),'P_ana_true_madkf'])

    data.append(res.loc[(res['n_ens']==50)&(res['n_iter']==5),'checkvar_enkf_true'])
    data.append(res.loc[(res['n_ens']==50)&(res['n_iter']==5),'checkvar_enkf_rmsd'])
    data.append(res.loc[(res['n_ens']==50)&(res['n_iter']==13),'checkvar_madkf'])

    data.append(res.loc[(res['n_ens']==50)&(res['n_iter']==5),'corr_OL'])
    data.append(res.loc[(res['n_ens']==50)&(res['n_iter']==5),'corr_obs'])
    data.append(res.loc[(res['n_ens']==50)&(res['n_iter']==5),'corr_ana_enkf_true'])
    data.append(res.loc[(res['n_ens']==50)&(res['n_iter']==5),'corr_ana_enkf_rmsd'])
    data.append(res.loc[(res['n_ens']==50)&(res['n_iter']==13),'corr_ana_madkf'])

    pos = np.arange(1,len(data)+1)

    plt.figure(figsize=(15, 7))
    fontsize = 14

    ax = plt.subplot(111)
    plt.grid(color='k', linestyle='--', linewidth=0.25)

    box = ax.boxplot(data, whis=[5, 95], showfliers=False, positions=pos, widths=0.3, patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set(color='black', linewidth=2)
        patch.set_facecolor(color)
    for patch in box['medians']:
        patch.set(color='black', linewidth=2)
    for patch in box['whiskers']:
        patch.set(color='black', linewidth=1)

    plt.xticks(np.arange(len(ticks))+1, ticks, fontsize=fontsize, rotation=60)
    plt.yticks(fontsize=fontsize)

    plt.xlim(0.5, len(ticks)+0.5)
    plt.ylim(0.6, 1.4)

    ax.set_title('n_ens = 50, n_iter = 13', fontsize=fontsize + 2)

    plt.axvline(2.5, linewidth=1.5, color='k')
    plt.axvline(5.5, linewidth=1.5, color='k')
    plt.axvline(8.5, linewidth=1.5, color='k')
    plt.axhline(1, linewidth=1.5, linestyle='--', color='k')

    plt.tight_layout()
    plt.show()

def plot_syn_result():

    res = pd.read_csv(r"D:\work\MadKF\API\synthetic_experiment\result.csv", index_col=0)

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
            tmp_data = res.loc[ind,'R_est_madkf'] / res.loc[ind,'R_true']
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
            tmp_data = res.loc[ind,'Q_est_madkf'] / res.loc[ind,'Q_true']
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
            tmp_data = res.loc[ind,'H_est_madkf'] / res.loc[ind,'H_true']
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
            # tmp_data = res.loc[ind,'P_ana_true']
            tmp_data = res.loc[ind,'checkvar_madkf']
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
    plt.ylim(0.8, 1.6)
    for i in np.arange(len(iterations)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    ax.set_title('Normalized innovation variance', fontsize=fontsize+2)


    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    plot_result()
    # plot_result2()
    # plot_syn_result()
    # plot_syn_result2()
    # plot_insitu_eval_results()