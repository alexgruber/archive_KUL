
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import date

from netCDF4 import Dataset

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from pyldas.interface import LDAS_io
from pyldas.templates import template_error_Tb40

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


def plot_ismn_statistics_v3(root):

    res = pd.read_csv(root / 'insitu_TCA.csv', index_col=0)
    res.index = res.network
    res.drop('network', axis='columns', inplace=True)
    res2 = pd.read_csv(root / 'insitu.csv', index_col=0)
    res2.index = res2.network
    res2.drop('network', axis='columns', inplace=True)

    variables = ['sm_surface', 'sm_rootzone']
    var_labels = ['surface', 'root-zone']

    runs = ['noDA', 'DA_const_err','DA_madkf']
    run_labels = ['Open Loop', 'EnKF (const. err.)','MadKF']
    offsets = [-0.2, 0.0, 0.2]
    cols = ['lightblue', 'lightgreen', 'coral']
    fontsize = 16

    # networks  = ['SCAN', 'USCRN']
    # title = ', '.join(networks)
    # res = res.loc[res.index.isin(networks),:]
    # res2 = res2.loc[res2.index.isin(networks),:]

    f = plt.figure(figsize=(14,8))

    titles = ['ubRMSD', 'ubRMSE', 'Pearson R$^2$ ', 'TCA R$^2$']

    ylims = [[0.0, 0.08],
             [0.0, 0.08],
             [0.0, 1.0],
             [0.0, 1.0]]

    valss = [[[res2['ubrmsd_' + run + '_absolute_' + var].values for run in runs] for var in variables],
             [[res['ubRMSE_model_' + run + '_absolute_' + var].values for run in runs] for var in variables],
             [[res2['corr_' + run + '_absolute_' + var].values ** 2 for run in runs] for var in variables],
             [[res['R2_model_' + run + '_absolute_' + var].values for run in runs] for var in variables]]

    for n, (vals, tit, ylim) in enumerate(zip(valss, titles, ylims)):

        ax = plt.subplot(2,2,n+1)

        plt.grid(color='k', linestyle='--', linewidth=0.25)

        data = list()
        ticks = list()
        pos = list()
        colors = list()

        for i, (val, var_label) in enumerate(zip(vals,var_labels)):

            ticks.append(var_label)
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
        plt.xticks(np.arange(len(var_labels))+1, ticks,fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlim(0.5,len(ticks)+0.5)
        plt.ylim(ylim)
        for k in np.arange(len(var_labels)):
            plt.axvline(k+0.5, linewidth=1, color='k')
        if n == 1:
            plt.figlegend((box['boxes'][0:4]),run_labels,'upper right',fontsize=fontsize-4)
        ax.set_title(tit ,fontsize=fontsize)

    fout = root / 'plots' / 'ismn_stats.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

    # plt.show()


def plot_improvement_vs_uncertainty_update(iteration):

    io = LDAS_io()

    root = Path('/work/MadKF/CLSM/iter_%i' % iteration)

    res = pd.read_csv(root / 'validation' / 'insitu_TCA.csv')
    res.index = res.network
    tilenum = np.vectorize(io.grid.colrow2tilenum)(res.ease_col.values,res.ease_row.values)

    root = Path('/work/MadKF/CLSM/iter_%i' % 531)
    fA = root / 'absolute' / 'error_files' / 'gapfilled' / 'SMOS_fit_Tb_A.bin'
    fD = root / 'absolute' / 'error_files' / 'gapfilled' / 'SMOS_fit_Tb_D.bin'

    dtype, hdr, length = template_error_Tb40()
    imgA = io.read_fortran_binary(fA, dtype, hdr=hdr, length=length)
    imgD = io.read_fortran_binary(fD, dtype, hdr=hdr, length=length)
    imgA.index += 1
    imgD.index += 1

    pol = 'h'
    orb = 'dsc'

    # if (orb == 'asc') & (pol == 'h'):
    #     perts = imgA.loc[tilenum,'err_Tbh'].values
    # elif (orb == 'asc') & (pol == 'v'):
    #     perts = imgA.loc[tilenum,'err_Tbv'].values
    # elif (orb == 'dsc') & (pol == 'h'):
    #     perts = imgD.loc[tilenum,'err_Tbh'].values
    # elif (orb == 'dsc') & (pol == 'v'):
    #     perts = imgD.loc[tilenum,'err_Tbv'].values

    perts = (imgA.loc[tilenum, 'err_Tbh'].values +
            imgA.loc[tilenum,'err_Tbv'].values +
            imgD.loc[tilenum,'err_Tbh'].values +
            imgD.loc[tilenum, 'err_Tbv'].values) / 4

    fontsize=14

    f = plt.figure(figsize=(20,12))

    for i,var in enumerate(['sm_surface', 'sm_rootzone']):
        for j,mode in enumerate(['absolute','shortterm','longterm']):

            tag1 = 'R2_model_DA_madkf_' + mode + '_' + var
            tag2 = 'R2_model_DA_const_err_' + mode + '_' + var
            dR2 = (res[tag1] - res[tag2]).values

            ind = np.where(~np.isnan(dR2))
            fit = np.polyfit(perts[ind],dR2[ind],1)


            ax = plt.subplot(2,3,j+1 + i*3)
            plt.axhline(color='black', linestyle='--', linewidth=1)
            plt.plot(perts, dR2, 'o', color='orange', markeredgecolor='black', markeredgewidth=0.5, markersize=6)
            plt.plot(np.arange(12), fit[0] * np.arange(12) + fit[1], '--', color='black', linewidth=3)

            plt.xlim(0,11)
            plt.ylim(-1,1)

            if i==0:
                plt.title(mode, fontsize=fontsize)
                ax.tick_params(labelbottom=False)
                # labels = [item.get_text() for item in ax.get_xticklabels()]
                # empty_string_labels = [''] * len(labels)
                # ax.set_xticklabels(empty_string_labels)
            else:
                plt.xticks(fontsize=fontsize-2)


            if j==0:
                plt.ylabel(var, fontsize=fontsize)
                plt.yticks(fontsize=fontsize-2)
            else:
                ax.tick_params(labelleft=False)
                # labels = [item.get_text() for item in ax.get_yticklabels()]
                # empty_string_labels = [''] * len(labels)
                # ax.set_xticklabels(empty_string_labels)

    fout = root / 'validation' / 'plots' / 'gain_vs_err.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

    # plt.tight_layout()
    # plt.show()

def plot_ts(lon, lat):

    experiments = ['US_M36_SMAP_TB_MadKF_DA_it34', 'US_M36_SMOS40_TB_MadKF_DA_it614', 'US_M36_SMOS40_TB_MadKF_DA_it615', 'US_M36_SMOS40_TB_MadKF_DA_it613']

    f = plt.figure(figsize=(18,10))

    for i, exp in enumerate(experiments):

        if 'SMAP' in exp:
            ol = 'US_M36_SMAP_TB_OL_noScl'
        else:
            ol = 'US_M36_SMOS40_TB_OL_noScl'

        ds_ol = LDAS_io('ObsFcstAna', ol)
        ds_da = LDAS_io('ObsFcstAna', exp)

        ts_fcst = ds_ol.read_ts('obs_fcst', lon, lat)
        ts_obs = ds_da.read_ts('obs_obs', lon, lat)
        ts_ana = ds_da.read_ts('obs_ana', lon, lat)

        spc = 1
            # if spc == 1:
            #     spc_tit = 'H pol. / Asc.'
            # elif spc == 2:
            #     spc_tit = 'H pol. / Dsc.'
            # elif spc == 3:
            #     spc_tit = 'V pol. / Asc.'
            # else:
            #     spc_tit = 'V pol. / Dsc.'

        df = pd.concat((ts_fcst[spc], ts_obs[spc], ts_ana[spc]), axis='columns').dropna()
        df.columns = ['Fcst', 'Obs', 'Ana']
        df['time'] = df.index

        ax = plt.subplot(4, 1, i+1)
        g = sns.lineplot(x='time', y='Tb', hue='Variable', data=df.melt('time', df.columns[0:-1], 'Variable', 'Tb'))
        plt.legend(loc='upper right')
        if spc != 4:
            g.set(xticklabels=[])
        ax.set_xlabel('')
        ax.set_xlim([date(2010,1,1), date(2020,1,1)])
        ax.set_ylim([170,280])
        # ax.set_ylabel('')
        plt.title(exp)

    plt.tight_layout()
    plt.show()

if __name__=='__main__':

    iteration = 3
    # root = Path(f'~/Documents/work/MadKF/CLSM/iter_{iteration}/validation').expanduser()
    root = Path(f'/Users/u0116961/Documents/work/MadKF/CLSM/SMAP/validation/iter_{iteration}')

    if not (root / 'plots').exists():
        Path.mkdir((root / 'plots'), parents=True)


    # exp = 'US_M36_SMAP_TB_DA_scl_SMOSSMAP_short'
    # exp = 'US_M36_SMOS40_TB_MadKF_DA_it614'
    # exp = 'US_M36_SMAP_TB_MadKF_DA_it34'

    # lat, lon = 37.573933, -96.840000 # Kansas
    # lat, lon = 44.434550, -99.703901 # South Dakota
    # lat, lon = 41.203456192, -102.249755859 # Nebraska

    # plot_ts(lon, lat)

    # plot_ismn_statistics(root)
    # plot_ismn_statistics_v2()
    # plot_ismn_statistics_v3(root)
    plot_filter_diagnostics(root, iteration)

    # plot_improvement_vs_uncertainty_update(iteration)

