
import os

os.environ["PROJ_LIB"] = "/Users/u0116961/miniconda2/pkgs/proj4-5.2.0-h1de35cc_1001/share/proj"
import platform
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use("TkAgg")

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
                cmap='jet',
                fontsize=24,
                plot_cb=True):

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

    if plot_cb is True:
        cb = m.colorbar(im, "bottom", size="7%", pad="4%")
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)

    # med = np.percentile(np.abs(img[np.where(~np.isnan(img))]),50)
    # x, y = m(-78.5,25)
    # plt.text(x,y,'median = %.2f' % med ,fontsize=16)

def plot_ismn_station_locations():

    nl = pd.DataFrame.from_csv(r"D:\work\MadKF\CONUS\ismn_eval\result.csv")

    print('SCAN: %i' % nl[nl.network=='SCAN'].shape[0])
    print('USCRN: %i' % nl[nl.network=='USCRN'].shape[0])

    lats_scan = nl[nl.network=='SCAN']['lat'].values
    lons_scan = nl[nl.network=='SCAN']['lon'].values

    lats_crn = nl[nl.network=='USCRN']['lat'].values
    lons_crn = nl[nl.network=='USCRN']['lon'].values

    llcrnrlat=24
    urcrnrlat=51
    llcrnrlon=-128
    urcrnrlon=-64

    figsize = (20, 10)

    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')

    m = Basemap(projection='mill',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
            llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,resolution='c',)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()


    xs, ys = m(lons_scan,lats_scan)
    xc, yc = m(lons_crn,lats_crn)

    plt.plot(xs,ys,'or',markersize = 13, markeredgecolor='black', label='SCAN')
    plt.plot(xc,yc,'ob',markersize = 13, markeredgecolor='black', label='USCRN')

    plt.legend(fontsize=26,loc=3)

    fname = r'D:\work\MadKF\CONUS\ismn_eval\station_locations.png'
    plt.tight_layout()
    plt.savefig(fname, dpi = f.dpi)
    plt.close()


def plot_insitu_eval_ci():

    res80 = pd.read_csv('/work/MadKF/CONUS/ismn_eval/result_ci80_niv_test_pr_constrained.csv', index_col=0).sort_values(by=['r_madkf_m'])
    res80 = res80[(res80.network == 'SCAN') | (res80.network == 'USCRN')]
    res80 = res80[res80.n_all > 50]

    res95 = pd.read_csv('/work/MadKF/CONUS/ismn_eval/result_ci95_niv_test_pr_constrained.csv', index_col=0).sort_values(by=['r_madkf_m'])
    res95 = res95[(res95.network == 'SCAN') | (res95.network == 'USCRN')]
    res95 = res95[res95.n_all > 50]

    res = res95[['r_madkf_m','r_kf_m','r_ol_m','r_avg_m']].copy()

    res.loc[res80[res80.r_madkf_l > res80.r_ol_u].index,'madkf_ol_80'] = 1
    res.loc[res80[res80.r_madkf_l > res80.r_avg_u].index,'madkf_avg_80'] = 1
    res.loc[res80[res80.r_madkf_l > res80.r_kf_u].index,'madkf_kf_80'] = 1

    res.loc[res80[res80.r_ol_l > res80.r_madkf_u].index,'ol_madkf_80'] = 1
    res.loc[res80[res80.r_avg_l > res80.r_madkf_u].index,'avg_madkf_80'] = 1
    res.loc[res80[res80.r_kf_l > res80.r_madkf_u].index,'kf_madkf_80'] = 1

    res.loc[res95[res95.r_madkf_l > res95.r_ol_u].index,'madkf_ol_95'] = 1
    res.loc[res95[res95.r_madkf_l > res95.r_avg_u].index,'madkf_avg_95'] = 1
    res.loc[res95[res95.r_madkf_l > res95.r_kf_u].index,'madkf_kf_95'] = 1

    res.loc[res95[res95.r_ol_l > res95.r_madkf_u].index,'ol_madkf_95'] = 1
    res.loc[res95[res95.r_avg_l > res95.r_madkf_u].index,'avg_madkf_95'] = 1
    res.loc[res95[res95.r_kf_l > res95.r_madkf_u].index,'kf_madkf_95'] = 1

    x = np.arange(len(res))
    res.index = x

    # print('OL > AVG: ', len(np.where(res.r_ol_l > res.r_avg_u)[0]) / len(res) * 100)
    # print('OL > TCA: ', len(np.where(res.r_ol_l > res.r_kf_u)[0]) / len(res) * 100)
    # print('OL > MadKF: ', len(np.where(res.r_ol_l > res.r_madkf_u)[0]) / len(res) * 100)
    #
    # print('AVG > OL: ', len(np.where(res.r_avg_l > res.r_ol_u)[0]) / len(res) * 100)
    # print('AVG > TCA: ', len(np.where(res.r_avg_l > res.r_kf_u)[0]) / len(res) * 100)
    # print('AVG > MadKF: ', len(np.where(res.r_avg_l > res.r_madkf_u)[0]) / len(res) * 100)
    #
    # print('TCA > OL: ', len(np.where(res.r_kf_l > res.r_ol_u)[0]) / len(res) * 100)
    # print('TCA > AVG: ', len(np.where(res.r_kf_l > res.r_avg_u)[0]) / len(res) * 100)
    # print('TCA > MadKF: ', len(np.where(res.r_kf_l > res.r_madkf_u)[0]) / len(res) * 100)
    #
    # print('MadKF > OL: ', len(np.where(res.r_madkf_l > res.r_ol_u)[0]) / len(res) * 100)
    # print('MadKF > AVG: ', len(np.where(res.r_madkf_l > res.r_avg_u)[0]) / len(res) * 100)
    # print('MadKF > TCA: ', len(np.where(res.r_madkf_l > res.r_kf_u)[0]) / len(res) * 100)
    #
    # print(np.percentile(res['n_all'],[5,25,50,75,95]))

    barsize = 8
    fontsize = 22
    legfontsize = fontsize - 2
    markersize = 7

    plt.subplots(3,1, sharex=True, figsize=(20, 8))

    plt.subplot(311)

    ind = np.where(~np.isnan(res.ol_madkf_80))[0]
    for i in ind:
        plt.plot((i,i),(-0.1,1),'-',linewidth=barsize, color='peachpuff')
    ind = np.where(~np.isnan(res.madkf_ol_80))[0]
    for i in ind:
        plt.plot((i,i),(-0.1,1),'-',linewidth=barsize, color='palegreen')

    ind = np.where(~np.isnan(res.ol_madkf_95))[0]
    for i in ind:
        plt.plot((i,i),(-0.1,1),'-',linewidth=barsize, color='salmon')
    ind = np.where(~np.isnan(res.madkf_ol_95))[0]
    for i in ind:
        plt.plot((i,i),(-0.1,1),'-',linewidth=barsize, color='limegreen')

    p1, = plt.plot(x, res['r_ol_m'], 'o', markersize=markersize, color='blue', label = 'open-loop')
    p2, = plt.plot(x, res['r_madkf_m'], '-', linewidth=4, color='black', label='MadKF')

    plt.legend(handles=[p1,p2], loc='lower right', fontsize=legfontsize)
    plt.yticks([0, 0.3, 0.6, 0.9], fontsize=fontsize)
    plt.ylim([-0.1, 1.])
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.xlim([-3, len(res) + 3])

    plt.title('MadKF versus API open-loop', fontsize=fontsize)

    plt.subplot(312)

    ind = np.where(~np.isnan(res.avg_madkf_80))[0]
    for i in ind:
        plt.plot((i, i), (-0.1, 1), '-', linewidth=barsize, color='peachpuff')
    ind = np.where(~np.isnan(res.madkf_avg_80))[0]
    for i in ind:
        plt.plot((i, i), (-0.1, 1), '-', linewidth=barsize, color='palegreen')

    ind = np.where(~np.isnan(res.avg_madkf_95))[0]
    for i in ind:
        plt.plot((i, i), (-0.1, 1), '-', linewidth=barsize, color='salmon')
    ind = np.where(~np.isnan(res.madkf_avg_95))[0]
    for i in ind:
        plt.plot((i, i), (-0.1, 1), '-', linewidth=barsize, color='limegreen')

    p1, = plt.plot(x, res['r_avg_m'], 'o', markersize=markersize, color='blue', label='Avg. TCA')
    p2, = plt.plot(x, res['r_madkf_m'], '-', linewidth=4, color='black', label='MadKF')

    plt.legend(handles=[p1,p2], loc='lower right', fontsize=legfontsize)
    plt.yticks([0,0.3,0.6,0.9],fontsize=fontsize)
    plt.xlim([-3, len(res) + 3])
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylim([-0.1, 1.])

    plt.ylabel('R$^2$ [-]', fontsize=fontsize)

    plt.title('MadKF versus spatially-averaged TCA-based EnKF', fontsize=fontsize)

    plt.subplot(313)

    ind = np.where(~np.isnan(res.kf_madkf_80))[0]
    for i in ind:
        plt.plot((i, i), (-0.1,1), '-', linewidth=barsize, color='peachpuff')
    ind = np.where(~np.isnan(res.madkf_kf_80))[0]
    for i in ind:
        plt.plot((i, i), (-0.1,1), '-', linewidth=barsize, color='palegreen')

    ind = np.where(~np.isnan(res.kf_madkf_95))[0]
    for i in ind:
        plt.plot((i, i), (-0.1,1), '-', linewidth=barsize, color='salmon')
    ind = np.where(~np.isnan(res.madkf_kf_95))[0]
    for i in ind:
        plt.plot((i, i), (-0.1,1), '-', linewidth=barsize, color='limegreen')

    p1, = plt.plot(x, res['r_kf_m'], 'o', markersize=markersize, color='blue', label='TCA')
    p2, = plt.plot(x, res['r_madkf_m'], '-', linewidth=4, color='black', label='MadKF')

    plt.legend(handles=[p1,p2], loc='lower right', fontsize=legfontsize)
    plt.yticks([0, 0.3, 0.6, 0.9], fontsize=fontsize)
    plt.ylim([-0.1, 1.])
    plt.xticks(fontsize=fontsize)
    # plt.tick_params(axis='x',which='both',bottom=False,top=False, labelbottom=False)
    plt.xlim([-3,len(res)+3])

    plt.subplots_adjust(wspace=0, hspace=0.3)
    plt.title('MadKF versus spatially-distributed TCA-based EnKF', fontsize=fontsize)

    plt.xlabel('R$^2$ rank (MadKF)', fontsize=fontsize)

    path = '/publications/2018_ag_madkf/images/'
    plt.savefig(path + "ismn_eval.png", bbox_inches='tight')
    plt.close()
    # plt.tight_layout()
    # plt.show()


def plot_insitu_eval_results():

    res = pd.read_csv('/work/MadKF/CONUS/ismn_eval/result.csv', index_col=0)

    res = res[(res.network == 'SCAN') | (res.network == 'USCRN')]

    # cols = ['r_ol', 'r_avg', 'r_kf', 'r_madkf']
    cols = ['r_ol_l', 'r_avg_l', 'r_kf_l', 'r_madkf_l']
    # cols = ['corr_ol', 'corr_avg', 'corr_kf', 'corr_madkf']
    # cols = ['R_innov_avg', 'R_innov_kf', 'R_innov_madkf']
    ylim = [0.0, 1.0]

    # cols = ['rmse_ol','rmse_kf', 'rmse_avg', 'rmse_rmsd', 'rmse_madkf']
    # ylim = [0, 0.04]

    res = res[res.n_all>50][cols]
    cols = ['OL', 'Avg. TC', 'TCA', 'MadKF']
    # cols = ['const. err', 'TCA', 'MadKF']

    print(len(res))

    tmp_data = [x[1].values for x in res.iteritems()]
    plt.figure(figsize=(8,5))
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
    plt.xticks(np.arange(len(cols))+1, cols, rotation=0, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('R$^2$', fontsize = 16)

    for i in np.arange(4)+1.5:
        plt.axvline(i, linewidth=1, linestyle='--', color='lightgrey', alpha=0.7)

    plt.title('SCAN + USCRN', fontsize=16)

    plt.tight_layout()
    plt.show()

def plot_result_Q_R_H():
    res = pd.read_csv("/work/MadKF/CONUS/result.csv", index_col=0)

    io = MSWEP_io(cellfiles = False)
    lats = io.ds['lat'][:]
    lons = io.ds['lon'][:]
    lons, lats = np.meshgrid(lons, lats)

    rows = res['row'].values.astype('int')
    cols = res['col'].values.astype('int')

    figsize = (26, 9.5)
    plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')

    cbrange_r = [0, 220]
    cbrange_q = [0, 73]
    cbrange_h = [-0.5, 2.5]

    fontsize = 24

    param = 'Q_tc'
    cbrange = cbrange_q
    # cbrange = [0, 16**2]
    cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] = res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(231)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap, plot_cb=False)
    plt.title('Q [mm$^2$]', fontsize=fontsize)
    plt.ylabel('TCA', fontsize=fontsize, rotation=90)

    param = 'R_tc'
    cbrange = cbrange_r
    # cbrange = [0, 16**2]
    cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] =  res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(232)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap, plot_cb=False)
    plt.title('R [%sat$^2$]', fontsize=fontsize)

    param = 'H_tc'
    cbrange = cbrange_h
    cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] =  res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(233)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap, plot_cb=False)
    plt.title('H [-]', fontsize=fontsize)

    param = 'Q_madkf'
    cbrange = cbrange_q
    cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] = res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(234)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap)
    plt.ylabel('MadKF', fontsize=fontsize, rotation=90)

    param = 'R_madkf'
    cbrange = cbrange_r
    cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] = res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(235)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap)

    param = 'H_madkf'
    cbrange = cbrange_h
    cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] = res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(236)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap)

    plt.subplots_adjust(wspace=0.05, hspace=0)

    path = '/publications/2018_ag_madkf/images/'
    plt.savefig(path + "test.png", bbox_inches='tight')
    plt.close()

    # plt.tight_layout()
    # plt.show()


def plot_result_diag():

    res = pd.read_csv(r"D:\work\MadKF\CONUS\result.csv", index_col=0)

    io = MSWEP_io(cellfiles = False)
    lats = io.ds['lat'][:]
    lons = io.ds['lon'][:]
    lons, lats = np.meshgrid(lons, lats)

    rows = res['row'].values.astype('int')
    cols = res['col'].values.astype('int')

    figsize = (17, 15)
    plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')

    # n_obs = 1461.

    fontsize = 24

    cmap = 'seismic_r'

    param = 'checkvar_avg'
    cbrange = [-1, 3]
    cmap = cmap
    # cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] = res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(321)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap)
    plt.title('Normalized Innovation Variance Averaged TCA', fontsize=20)

    param = 'R_innov_avg'
    cbrange = [-0.6, 0.6]
    cmap = cmap
    # cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] = res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(322)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap)
    plt.title('Innovation auto-correlation', fontsize=fontsize)

    param = 'checkvar_tc'
    cbrange = [-1, 3]
    cmap = cmap
    # cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows,cols)
    img[ind] = res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(323)
    plot_figure(img_masked,lons,lats,cbrange=cbrange,cmap=cmap)
    plt.title('TCA', fontsize=fontsize)


    param = 'R_innov_tc'
    cbrange = [-0.6, 0.6]
    cmap = cmap
    # cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] = res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(324)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap)
    plt.title(' ', fontsize=fontsize)

    param = 'checkvar_madkf'
    cbrange = [-1, 3]
    cmap = cmap
    # cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows, cols)
    img[ind] = res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(325)
    plot_figure(img_masked, lons, lats, cbrange=cbrange, cmap=cmap)
    plt.title('MadKF', fontsize=fontsize)

    param = 'R_innov_madkf'
    cbrange = [-0.6, 0.6]
    cmap = cmap
    # cmap = 'jet'
    img = np.full(lats.shape, np.nan)
    ind = (rows,cols)
    img[ind] = res[param].values
    img_masked = np.ma.masked_invalid(img)
    plt.subplot(326)
    plot_figure(img_masked,lons,lats,cbrange=cbrange,cmap=cmap)
    plt.title(' ', fontsize=fontsize)

    plt.tight_layout()

    plt.savefig(r'D:\work\MadKF\CONUS\diagnostics.png')
    plt.close()
    # plt.show()

def plot_syn_result3():
    res1 = pd.read_csv(r"/work/MadKF/synthetic_experiment/no_c_corr/result.csv", index_col=0)
    res2 = pd.read_csv(r"/work/MadKF/synthetic_experiment/result.csv", index_col=0)
    res = pd.concat((res1, res2))

    plt.figure(figsize=(18, 9))

    iterations = [1, 3, 6, 9, 12, 15]

    ensembles = [10, 30, 50, 70]
    legend = ['%i' % ens for ens in ensembles]

    offsets = [-0.3, -0.1, 0.1, 0.3]
    cols = ['lightblue', 'lightgreen', 'coral', 'brown']
    fontsize = 16

    ax = plt.subplot2grid((2,4),(0,0),colspan=2)
    plt.grid(color='k', linestyle='--', linewidth=0.25)
    data = list()
    ticks = list()
    pos = list()
    colors = list()
    for i, iters in enumerate(iterations):
        if i == 0:
            ticks.append('0')
        else:
            ticks.append(iters)
        for col, offs, ens in zip(cols, offsets, ensembles):
            ind = (res['n_ens'] == ens) & (res['n_iter'] == iters)
            tmp_data = res.loc[ind, 'R_innov_madkf']
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
    # plt.figlegend((box['boxes'][0:4]), legend, 'upper right', fontsize=fontsize)
    plt.xticks(np.arange(len(iterations)) + 1, ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0.5, len(ticks) + 0.5)
    # plt.ylim(0.2, 1.8)
    plt.ylim(-0.2, 0.4)
    for i in np.arange(len(iterations)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    ax.set_title('Innovation Auto-Correlation', fontsize=fontsize + 2)

    ax = plt.subplot2grid((2,4),(0,2),colspan=2)
    plt.grid(color='k', linestyle='--', linewidth=0.25)
    data = list()
    ticks = list()
    pos = list()
    colors = list()
    for i, iters in enumerate(iterations):
        if i == 0:
            ticks.append('0')
        else:
            ticks.append(iters)
        for col, offs, ens in zip(cols, offsets, ensembles):
            ind = (res['n_ens'] == ens) & (res['n_iter'] == iters)
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
    plt.figlegend((box['boxes'][0:4]), legend, 'upper right', fontsize=fontsize)
    plt.xticks(np.arange(len(iterations)) + 1, ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0.5, len(ticks) + 0.5)
    plt.ylim(0.2, 1.8)
    for i in np.arange(len(iterations)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    ax.set_title('Normalized Innovation Variance', fontsize=fontsize+2)

    ax = plt.subplot2grid((2,4),(1,1),colspan=2)
    plt.grid(color='k', linestyle='--', linewidth=0.25)
    data = list()
    ticks = list()
    pos = list()
    colors = list()
    for i, iters in enumerate(iterations):
        if i == 0:
            ticks.append('0')
        else:
            ticks.append(iters)
        for col, offs, ens in zip(cols, offsets, ensembles):
            ind = (res['n_ens'] == ens) & (res['n_iter'] == iters)
            tmp_data = res.loc[ind, 'corr_ana_madkf']**2
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
    # plt.figlegend((box['boxes'][0:4]), legend, 'upper left', fontsize=fontsize)
    plt.xticks(np.arange(len(iterations)) + 1, ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0.5, len(ticks) + 0.5)
    plt.ylim(0.5, 1.0)
    for i in np.arange(len(iterations)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    ax.set_title('R$^2$', fontsize=fontsize + 2)
    plt.axhline(np.median(res['corr_OL']**2), linestyle='--', color='black', linewidth=1.5)

    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.tight_layout()
    plt.show()
    # path = '/publications/2018_ag_madkf/images/'
    # plt.savefig(path + "diagnostics_synthetic.png", bbox_inches='tight')
    # plt.close()

def plot_syn_result2():

    res = pd.read_csv("/work/MadKF/synthetic_experiment/result.csv", index_col=0)

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
    # plt.ylim(0.6, 1.4)

    ax.set_title('n_ens = 50, n_iter = 13', fontsize=fontsize + 2)

    plt.axvline(2.5, linewidth=1.5, color='k')
    plt.axvline(5.5, linewidth=1.5, color='k')
    plt.axvline(8.5, linewidth=1.5, color='k')
    plt.axhline(1, linewidth=1.5, linestyle='--', color='k')

    plt.tight_layout()
    plt.show()

def plot_syn_result():

    res1 = pd.read_csv(r"D:\work\MadKF\synthetic_experiment\result.csv", index_col=0)
    res2 = pd.read_csv(r"D:\work\MadKF\synthetic_experiment\no_c_corr\result.csv", index_col=0)
    res = pd.concat((res1,res2))

    plt.figure(figsize=(18, 9))

    iterations = [1, 3, 6, 9, 12, 15]

    ensembles = [10, 30, 50, 70]
    legend = ['%i' % ens for ens in ensembles]

    offsets = [-0.3,-0.1,0.1,0.3]
    cols = ['lightblue', 'lightgreen', 'coral', 'brown']
    fontsize = 16

    ax = plt.subplot(221)
    plt.grid(color='k', linestyle='--', linewidth=0.25)
    data = list()
    ticks = list()
    pos = list()
    colors = list()
    for i, iters in enumerate(iterations):
        if i == 0:
            ticks.append('0')
        else:
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
    # plt.figlegend((box['boxes'][0:4]), legend, 'upper right', fontsize=fontsize)
    plt.xticks(np.arange(len(iterations)) + 1, ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0.5, len(ticks) + 0.5)
    plt.ylim(0.0, 5.0)
    for i in np.arange(len(iterations)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    ax.set_title('Q (est.) / Q (true)', fontsize=fontsize+2)
    plt.axhline(1, linestyle='--', color='black', linewidth=1)

    ax = plt.subplot(222)
    plt.grid(color='k', linestyle='--', linewidth=0.25)
    data = list()
    ticks = list()
    pos = list()
    colors = list()
    for i, iters in enumerate(iterations):
        if i == 0:
            ticks.append('0')
        else:
            ticks.append(iters)
        for col, offs, ens in zip(cols, offsets, ensembles):
            ind = (res['n_ens'] == ens) & (res['n_iter'] == iters)
            tmp_data = res.loc[ind, 'R_est_madkf'] / res.loc[ind, 'R_true']
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
    plt.figlegend((box['boxes'][0:4]), legend, 'upper right', fontsize=fontsize)
    plt.xticks(np.arange(len(iterations)) + 1, ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0.5, len(ticks) + 0.5)
    plt.ylim(0.0, 5.0)
    for i in np.arange(len(iterations)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    ax.set_title('R (est.) / R (true)', fontsize=fontsize + 2)
    plt.axhline(1, linestyle='--', color='black', linewidth=1)

    ax = plt.subplot(223)
    plt.grid(color='k', linestyle='--', linewidth=0.25)
    data = list()
    ticks = list()
    pos = list()
    colors = list()
    for i, iters in enumerate(iterations):
        if i == 0:
            ticks.append('0')
        else:
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
    # plt.figlegend((box['boxes'][0:4]), legend, 'upper left', fontsize=fontsize)
    plt.xticks(np.arange(len(iterations)) + 1, ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0.5, len(ticks) + 0.5)
    plt.ylim(-0.5, 2.5)
    for i in np.arange(len(iterations)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    ax.set_title('H (est.) / H (true)', fontsize=fontsize+2)
    plt.axhline(1, linestyle='--', color='black', linewidth=1)


    ax = plt.subplot(224)
    plt.grid(color='k', linestyle='--', linewidth=0.25)
    data = list()
    ticks = list()
    pos = list()
    colors = list()
    for i, iters in enumerate(iterations):
        if i == 0:
            ticks.append('0')
        else:
            ticks.append(iters)
        for col, offs, ens in zip(cols, offsets, ensembles):
            ind = (res['n_ens']==ens)&(res['n_iter']==iters)
            # tmp_data = res.loc[ind,'checkvar_madkf']
            tmp_data = res.loc[ind,'R_innov_madkf']
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
    plt.figlegend((box['boxes'][0:4]), legend, 'upper right', fontsize=fontsize)
    plt.xticks(np.arange(len(iterations)) + 1, ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0.5, len(ticks) + 0.5)
    # plt.ylim(0.2, 1.8)
    plt.ylim(-0.2,0.4)
    for i in np.arange(len(iterations)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    # ax.set_title('Normalized Innovation Variance', fontsize=fontsize+2)
    ax.set_title('Innovation Auto-Correlation', fontsize=fontsize+2)
    plt.axhline(1, linestyle='--', color='black', linewidth=1)

    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    # plot_result_diag()
    # plot_result_Q_R_H()
    # plot_syn_result()
    # plot_syn_result2()
    # plot_syn_result3()
    # plot_insitu_eval_results()
    plot_insitu_eval_ci()
    # plot_ismn_station_locations()