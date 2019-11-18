
import numpy as np
import pandas as pd

from pathlib import Path

from netCDF4 import Dataset

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def plot_image(img, lats, lons,
                llcrnrlat=24,
                urcrnrlat=51,
                llcrnrlon=-128,
                urcrnrlon=-64,
                cbrange=(-20,20),
                cmap='jet',
                title='',
                fontsize=14):

    # img = np.full(lons.shape, np.nan)
    # img[:] = data
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
    # cb = m.colorbar(im, "bottom", size="7%", pad="5%")
    # for t in cb.ax.get_xticklabels():
    #     t.set_fontsize(fontsize)
    # for t in cb.ax.get_yticklabels():
    #     t.set_fontsize(fontsize)

    plt.title(title,fontsize=fontsize)

    x, y = m(-78.5, 27.5)
    plt.text(x, y, 'm', fontsize=fontsize - 3)
    x, y = m(-75, 27.5)
    plt.text(x, y, '= %.2f' % np.ma.median(img_masked), fontsize=fontsize - 3)

    x, y = m(-78, 25)
    plt.text(x, y, 's', fontsize=fontsize - 3)
    x, y = m(-75, 25)
    plt.text(x, y, '= %.2f' % np.ma.std(img_masked), fontsize=fontsize - 3)


def plot_filter_diagnostics(root, iteration):

    fname = root / 'filter_diagnostics.nc'

    fontsize = 14

    # iters = [0, iteration-2, iteration-1, iteration]
    iters = [0, 1, 2, 3]

    with Dataset(fname) as ds:

        lons = ds.variables['lon'][:]
        lats = ds.variables['lat'][:]
        lons, lats = np.meshgrid(lons, lats)

        # variables = ['norm_innov_mean',]
        variables = ['norm_innov_mean','norm_innov_var']
        cbranges = [[-1,1], [-3,5]]
        steps = [0.5, 2]

        for var, cbrange, step in zip(variables,cbranges,steps):

            f = plt.figure(figsize=(19,10))

            for spc in np.arange(4):
                for it_tit, it in zip(iters,np.arange(4)):

                    n = spc * 4 + it + 1

                    tit = 'Iter %i / Spc %i' % (it_tit, spc)

                    plt.subplot(4,4,n)

                    data = ds.variables[var][:,:,it,spc]

                    plot_image(data, lats, lons,
                               cmap='jet',
                               cbrange=cbrange,
                               fontsize = fontsize,
                               title=tit)

            f.subplots_adjust(hspace=0, wspace=0.05, bottom=0.05)

            pos1 = f.axes[-3].get_position()
            pos2 = f.axes[-2].get_position()

            x1 = (pos1.x0 + pos1.x1)/2
            x2 = (pos2.x0 + pos2.x1)/2

            im1 = f.axes[0].collections[-1]

            ticks = np.arange(cbrange[0], cbrange[1]+1, step)

            cbar_ax = f.add_axes([x1, 0.04, x2-x1, 0.02])
            cbar = f.colorbar(im1, orientation='horizontal', cax=cbar_ax, ticks=ticks)
            for t in cbar.ax.get_xticklabels():
                t.set_fontsize(fontsize)

            fout = root / 'plots' / (var + '.png')
            f.savefig(fout, dpi=300, bbox_inches='tight')
            plt.close()
            # plt.tight_layout()
            # plt.show()


def plot_ismn_statistics(root):

    res = pd.read_csv(root / 'insitu_TCA.csv')
    res.index = res.network
    res2 = pd.read_csv(root / 'insitu.csv')
    res2.index = res2.network

    modes = ['absolute','shortterm', 'longterm']
    networks  = ['SCAN', 'USCRN']


    variables = ['sm_surface','sm_rootzone','sm_profile']
    var_labels = ['ssm', 'rzsm', 'prsm']

    # variables = ['sm_surface',]
    # var_labels = ['ssm',]

    runs = ['noDA', 'DA_const_err','DA_madkf']
    offsets = [-0.2, 0.0, 0.2]
    cols = ['lightblue', 'lightgreen', 'coral']
    fontsize = 12

    # all networks + selection
    for i_net in np.arange(2):

        if i_net == 0:
            title = 'all networks'
        else:
            title = ', '.join(networks)
            res = res.loc[networks,:]
            res2 = res2.loc[networks,:]

        for var, var_label in zip(variables, var_labels):

            plt.figure(figsize=(15,10))

            # for i,mode in enumerate(modes):

            titles = ['ubRMSD (' + var_label + ') '+ title,
                      'ubRMSE (' + var_label + ') ' + title,
                      'Pearson R (' + var_label + ') '+ title,
                      'TCA R2 (' + var_label + ') '+ title]

            ylims = [[0.0, 0.1],
                     [0.0, 0.1],
                     [0.0, 1.0],
                     [0.0, 1.0]]

            valss = [[[res2['ubrmsd_' + run + '_' + mode + '_' + var].values for run in runs] for mode in modes],
                     [[res['ubRMSE_model_' + run + '_' + mode + '_' + var].values for run in runs] for mode in modes],
                     [[res2['corr_' + run + '_' + mode + '_' + var].values ** 2 for run in runs] for mode in modes],
                     [[res['R2_model_' + run + '_' + mode + '_' + var].values for run in runs] for mode in modes]]

            for n, (vals, tit, ylim) in enumerate(zip(valss, titles, ylims)):

                ax = plt.subplot(2,2,n+1)

                plt.grid(color='k', linestyle='--', linewidth=0.25)

                data = list()
                ticks = list()
                pos = list()
                colors = list()

                for i, (val, mode) in enumerate(zip(vals,modes)):

                    ticks.append(mode)
                    for col,offs, v in zip(cols,offsets,val):
                        tmp_data = v
                        tmp_data = tmp_data[~np.isnan(tmp_data)]
                        data.append(tmp_data)
                        pos.append(i+1 + offs)
                        colors.append(col)

                box = ax.boxplot(data, whis=[5,95], showfliers=False, positions=pos, widths=0.1, patch_artist=True)
                for patch, color in zip(box['boxes'], colors):
                    patch.set(color='black', linewidth=2)
                    patch.set_facecolor(color)
                for patch in box['medians']:
                    patch.set(color='black', linewidth=2)
                for patch in box['whiskers']:
                    patch.set(color='black', linewidth=1)
                plt.xticks(np.arange(len(modes))+1, ticks,fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.xlim(0.5,len(ticks)+0.5)
                plt.ylim(ylim)
                for k in np.arange(len(modes)):
                    plt.axvline(k+0.5, linewidth=1, color='k')
                if n == 1:
                    plt.figlegend((box['boxes'][0:4]),runs,'upper right',fontsize=fontsize)
                ax.set_title(tit ,fontsize=fontsize)

            plt.tight_layout()

            fname = root / 'plots' / (title + '_' + var + '.png')
            plt.savefig(fname)
            plt.close()

            # plt.show()


def plot_ismn_statistics_v2():


    root = Path('/work/MadKF/CLSM/')

    res_5 = pd.read_csv(root / 'iter_5' / 'validation' / 'insitu_TCA.csv')
    res_5.index = res_5.network
    res2_5 = pd.read_csv(root / 'iter_5' / 'validation' / 'insitu.csv')
    res2_5.index = res2_5.network

    res_51 = pd.read_csv(root / 'iter_51' / 'validation' / 'insitu_TCA.csv')
    res_51.index = res_51.network
    res2_51 = pd.read_csv(root / 'iter_51' / 'validation' / 'insitu.csv')
    res2_51.index = res2_51.network

    res_52 = pd.read_csv(root / 'iter_52' / 'validation' / 'insitu_TCA.csv')
    res_52.index = res_52.network
    res2_52 = pd.read_csv(root / 'iter_52' / 'validation' / 'insitu.csv')
    res2_52.index = res2_52.network

    modes = ['absolute','shortterm', 'longterm']
    networks  = ['SCAN', 'USCRN']

    variables = ['sm_surface','sm_rootzone','sm_profile']
    var_labels = ['ssm', 'rzsm', 'prsm']

    runs = ['v5', 'v51','v52']
    offsets = [-0.2, 0.0, 0.2]
    cols = ['lightblue', 'lightgreen', 'coral']
    fontsize = 12

    # all networks + selection
    for i_net in np.arange(2):

        if i_net == 0:
            title = 'all networks'
        else:
            title = ', '.join(networks)
            res_5 = res_5.loc[networks,:]
            res_51 = res_51.loc[networks,:]
            res_52 = res_52.loc[networks,:]
            res2_5 = res2_5.loc[networks,:]
            res2_51 = res2_51.loc[networks,:]
            res2_52 = res2_52.loc[networks,:]

        ress = [res_5, res_51, res_52]
        ress2 = [res2_5, res2_51, res2_52]

        for var, var_label in zip(variables, var_labels):

            plt.figure(figsize=(15,10))

            # for i,mode in enumerate(modes):

            titles = ['ubRMSD (' + var_label + ') '+ title,
                      'ubRMSE (' + var_label + ') ' + title,
                      'Pearson R (' + var_label + ') '+ title,
                      'TCA R2 (' + var_label + ') '+ title]

            ylims = [[0.0, 0.1],
                     [0.0, 0.1],
                     [0.0, 1.0],
                     [0.0, 1.0]]

            valss = [[[res2['ubrmsd_DA_madkf_' + mode + '_' + var].values for res2 in ress2] for mode in modes],
                     [[res['ubRMSE_model_DA_madkf_' + mode + '_' + var].values for res in ress] for mode in modes],
                     [[res2['corr_DA_madkf_' + mode + '_' + var].values ** 2 for res2 in ress2] for mode in modes],
                     [[res['R2_model_DA_madkf_' + mode + '_' + var].values for res in ress] for mode in modes]]

            for n, (vals, tit, ylim) in enumerate(zip(valss, titles, ylims)):

                ax = plt.subplot(2,2,n+1)

                plt.grid(color='k', linestyle='--', linewidth=0.25)

                data = list()
                ticks = list()
                pos = list()
                colors = list()

                for i, (val, mode) in enumerate(zip(vals,modes)):

                    ticks.append(mode)
                    for col,offs, v in zip(cols,offsets,val):
                        tmp_data = v
                        tmp_data = tmp_data[~np.isnan(tmp_data)]
                        data.append(tmp_data)
                        pos.append(i+1 + offs)
                        colors.append(col)

                box = ax.boxplot(data, whis=[5,95], showfliers=False, positions=pos, widths=0.1, patch_artist=True)
                for patch, color in zip(box['boxes'], colors):
                    patch.set(color='black', linewidth=2)
                    patch.set_facecolor(color)
                for patch in box['medians']:
                    patch.set(color='black', linewidth=2)
                for patch in box['whiskers']:
                    patch.set(color='black', linewidth=1)
                plt.xticks(np.arange(len(modes))+1, ticks,fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                plt.xlim(0.5,len(ticks)+0.5)
                plt.ylim(ylim)
                for k in np.arange(len(modes)):
                    plt.axvline(k+0.5, linewidth=1, color='k')
                if n == 1:
                    plt.figlegend((box['boxes'][0:4]),runs,'upper right',fontsize=fontsize)
                ax.set_title(tit ,fontsize=fontsize)

            plt.tight_layout()

            fname = '/work/MadKF/CLSM/version_comparison/' + (title + '_' + var + '.png')
            plt.savefig(fname)
            plt.close()

            # plt.show()


if __name__=='__main__':

    iteration = 52
    root = Path('/work/MadKF/CLSM/iter_%i/validation' % iteration)

    if not (root / 'plots').exists():
        Path.mkdir((root / 'plots'), parents=True)

    # plot_ismn_statistics(root)
    plot_ismn_statistics_v2()
    # plot_filter_diagnostics(root, iteration)