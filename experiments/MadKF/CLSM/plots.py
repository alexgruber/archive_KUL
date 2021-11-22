
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import date

from netCDF4 import Dataset

import seaborn as sns
sns.set_context('talk', font_scale=0.8)

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import colorcet as cc

from pyldas.interface import LDAS_io
from pyldas.templates import template_error_Tb40

from myprojects.readers.ascat import HSAF_io
from myprojects.readers.insitu import ISMN_io
from myprojects.timeseries import calc_anom

from pytesmo.temporal_matching import df_match

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

    x, y = m(-79, 27.5)
    plt.text(x, y, 'mean', fontsize=fontsize - 3)
    x, y = m(-74, 27.5)
    plt.text(x, y, '= %.2f' % np.ma.median(img_masked), fontsize=fontsize - 3)

    x, y = m(-79, 25)
    plt.text(x, y, 'std.', fontsize=fontsize - 3)
    x, y = m(-74, 25)
    plt.text(x, y, '= %.2f' % np.ma.std(img_masked), fontsize=fontsize - 3)

def plot_filter_diagnostics_violins(root):

    fname = root / 'validation' / 'filter_diagnostics.nc'

    fontsize = 14

    names = ['OL_Pcorr', 'OL_noPcorr' ] + \
            [f'DA_{pc}_{err}' for pc in ['Pcorr', 'noPcorr'] for err in ['4K', 'abs', 'anom_lt', 'anom_lst', 'anom_st']]
    species = [1,2,3,4]

    sns.set_context('talk', font_scale=0.8)

    params = ['innov_autocorr', 'norm_innov_var']
    ylims = [[-0.1, 1.1],[-1,6]]

    with Dataset(fname) as ds:

        for param, ylim in zip(params,ylims):

            data = ds[param][:,:,:,:].flatten()

            df = pd.DataFrame({'data': data,
                               'names': np.tile(names,int(len(data)/len(names))),
                               'species': np.repeat(species,int(len(data)/len(species)))})

            # f = plt.figure(figsize=(15,10))
            g = sns.catplot(x='names', y='data', data=df, row='species', kind='violin', gridsize=300,height=2.5, aspect=7)

            # g.set_titles('{col_name}')
            g.set_ylabels('')
            g.set_xlabels('')

            [ax[0].set(ylim=ylim) for ax in g.axes]

            [ax[0].axvline(1.5, color='black', linestyle='-', linewidth=1.5) for ax in g.axes]
            [ax[0].axvline(6.5, color='black', linestyle='-', linewidth=1.5) for ax in g.axes]
            if param == 'innov_autocorr':
                [ax[0].axhline(0.4, color='black', linestyle='--', linewidth=1) for ax in g.axes]
            else:
                [ax[0].axhline(1, color='black', linestyle='--', linewidth=1) for ax in g.axes]

            g.set_xticklabels(rotation=15)

            fout = root / 'plots' / f'diagnostics' / f'violin_{param}.png'
            g.savefig(fout, dpi=300, bbox_inches='tight')
            plt.close()

        # plt.show()

def plot_filter_diagnostics(root):

    fname = root / 'validation' / 'filter_diagnostics.nc'

    fontsize = 14

    runs = ['OL_Pcorr', 'OL_noPcorr', 'DA_Pcorr_4K', 'DA_noPcorr_4K'] + \
            [f'DA_{pc}_{err}' for pc in ['Pcorr', 'noPcorr'] for err in ['abs', 'anom_lt', 'anom_lst', 'anom_st']]

    # ref = 'DA_Pcorr_4K'
    ref = None

    iters = np.arange(len(runs))

    with Dataset(fname) as ds:

        lons = ds.variables['lon'][:]
        lats = ds.variables['lat'][:]
        lons, lats = np.meshgrid(lons, lats)

        # variables = ['innov_autocorr',]
        # cbranges = [[-0.1,0.1]]
        # steps = [0.05]
        # cmaps = [cc.cm.bjy]
        variables = ['innov_autocorr_abs','norm_innov_mean_abs','norm_innov_var_abs']
        cbranges = [[0,0.7], [-0.5,0.5], [-2,4]]
        steps = [0.2, 0.25, 1]
        cmaps = ['viridis' ,cc.cm.bjy,  cc.cm.bjy]

        for var, cbrange, cmap, step in zip(variables, cbranges, cmaps, steps):
            for spc in np.arange(4):
                f = plt.figure(figsize=(23, 9))

                for i, (it_tit, it) in enumerate(zip(runs,iters)):

                    plt.subplot(3, 4, i+1)
                    if ref is not None:
                        refit = iters[np.array(runs)==ref][0]
                        data = ds.variables[var][:,:,it,spc] - ds.variables[var][:,:,refit,spc]
                    else:
                        data = ds.variables[var][:,:,it,spc]
                    plot_image(data, lats, lons,
                               cmap=cmap,
                               cbrange=cbrange,
                               fontsize = fontsize,
                               title=it_tit)

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
                if ref is not None:
                    fout = root / 'plots' / f'diagnostics' / f'rel_{var}_spc{spc+1}.png'
                else:
                    fout = root / 'plots' / f'diagnostics' / f'{var}_spc{spc+1}.png'
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

def find_suspicious_stations(root):

    res = pd.read_csv(root / 'insitu_TCA.csv', index_col=0)
    res.index = res.network + '_' + res.station
    # res.drop(['network','station'], axis='columns', inplace=True)

    res2 = pd.read_csv(root / 'insitu.csv', index_col=0)
    res2.index = res2.network + '_' + res2.station
    res2.drop(['network','station','lat','lon', 'ease_col','ease_row'], axis='columns', inplace=True)
    res2 = res2[~res2.index.duplicated(keep='first')].reindex(res.index)

    df = res[['network','station','ease_col','ease_row']].copy()

    variables = ['sm_surface','sm_rootzone']
    modes = ['absolute','longterm','shortterm']

    for var in variables:
        for mode in modes:
            df[f'diff_pearsonr2_{mode}_{var}'] = res2[f'corr_DA_4K_obserr_{mode}_{var}']**2 - res2[f'corr_open_loop_{mode}_{var}']**2
            df[f'diff_tcar2_{mode}_{var}'] = res[f'R2_model_DA_4K_obserr_{mode}_{var}'] - res[f'R2_model_open_loop_{mode}_{var}']

    df = df[(df.network == 'SCAN') | (df.network == 'USCRN')].dropna()
    df.to_csv('/Users/u0116961/Documents/work/MadKF/CLSM/suspicious_stations/station_list_r_diff.csv', float_format='%0.4f')

def lonlat2gpi(lon,lat,gpi_list):

    rdiff = np.sqrt((gpi_list.lon - lon)**2 + (gpi_list.lat - lat)**2)
    return gpi_list.iloc[np.where((rdiff - rdiff.min()) < 0.0001)[0][0]].name

def plot_suspicious_stations(root):

    statlist = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/suspicious_stations/station_list_r_diff.csv', index_col=0)

    rmsd_root = 'US_M36_SMAP_TB_DA_SM_PROXY_'
    rmsd_exps = list(np.sort([x.name.split(rmsd_root)[1] for x in Path('/Users/u0116961/data_sets/LDASsa_runs').glob('*SM_PROXY*')]))

    ds_ol = LDAS_io('xhourly', 'US_M36_SMAP_TB_OL_scaled_4K_obserr').timeseries
    ds_da = LDAS_io('xhourly', 'US_M36_SMAP_TB_DA_scaled_4K_obserr').timeseries

    ts_ana = LDAS_io('ObsFcstAna', 'US_M36_SMAP_TB_DA_scaled_4K_obserr').timeseries['obs_obs']
    t_ana = pd.DatetimeIndex(ts_ana.time.values).sort_values()

    ascat = HSAF_io()
    gpi_list = pd.read_csv(ascat.root / 'warp5_grid' / 'pointlist_warp_conus.csv', index_col=0)

    ismn = ISMN_io()

    variables = ['sm_surface', 'sm_rootzone']
    modes = ['absolute', 'longterm', 'shortterm']

    ismn.list.index = ismn.list.network + '_' + ismn.list.station
    ismn.list.reindex(statlist.index)
    ismn.list = ismn.list.reindex(statlist.index)

    for i, (meta, ts_insitu) in enumerate(ismn.iter_stations(surface_only=False)):
        if 'tmp_res' in locals():
            if (meta.network in tmp_res) & (meta.station in tmp_res):
                print(f'Skipping {i}')
                continue

        try:
            res = pd.DataFrame(meta.copy()).transpose()
            col = meta.ease_col
            row = meta.ease_row

            gpi = lonlat2gpi(meta.lon, meta.lat, gpi_list)

            ts_ascat = ascat.read(gpi) / 100 * 0.6
            if ts_ascat is None:
                continue

            for mode in modes:
                for var in variables:

                    tmp = statlist[(statlist.network==meta.network)&(statlist.station==meta.station)]
                    dpr = tmp[f'diff_pearsonr2_{mode}_{var}'].values[0]
                    dtr = tmp[f'diff_tcar2_{mode}_{var}'].values[0]

                    if not ((dtr < 0) & (dpr > 0)):
                        continue

                    if mode == 'absolute':
                        ts_asc = ts_ascat.dropna()
                    else:
                        ts_asc = calc_anom(ts_ascat, longterm=(mode == 'longterm')).dropna()
                    ts_asc.name = 'ascat'
                    ts_asc = pd.DataFrame(ts_asc)

                    if mode == 'absolute':
                        ts_ins = ts_insitu[var].dropna()
                    else:
                        ts_ins = calc_anom(ts_insitu[var], longterm=(mode == 'longterm')).dropna()
                    ts_ins.name = 'insitu'
                    ts_ins = pd.DataFrame(ts_ins)

                    ind = (ds_ol['snow_mass'].isel(lat=row, lon=col).values == 0) & \
                          (ds_ol['soil_temp_layer1'].isel(lat=row, lon=col).values > 277.15)
                    ts_ol = ds_ol[var].isel(lat=row, lon=col).to_series().loc[ind]
                    ts_ol.index += pd.to_timedelta('2 hours')
                    ind_obs = np.bitwise_or.reduce(~np.isnan(ts_ana[:, :, row, col].values), 1)
                    if mode == 'absolute':
                        ts_ol = ts_ol.reindex(t_ana[ind_obs]).dropna()
                    else:
                        ts_ol = calc_anom(ts_ol.reindex(t_ana[ind_obs]), longterm=(mode == 'longterm')).dropna()
                    ts_ol.name = 'open_loop'
                    ts_ol = pd.DataFrame(ts_ol)

                    ind = (ds_da['snow_mass'].isel(lat=row, lon=col).values == 0) & \
                          (ds_da['soil_temp_layer1'].isel(lat=row, lon=col).values > 277.15)
                    ts_da = ds_da[var].isel(lat=row, lon=col).to_series().loc[ind]
                    ts_da.index += pd.to_timedelta('2 hours')
                    ind_obs = np.bitwise_or.reduce(~np.isnan(ts_ana[:, :, row, col].values), 1)
                    if mode == 'absolute':
                        ts_da = ts_da.reindex(t_ana[ind_obs]).dropna()
                    else:
                        ts_da = calc_anom(ts_da.reindex(t_ana[ind_obs]), longterm=(mode == 'longterm')).dropna()
                    ts_da.name = 'DA_4K'
                    ts_da = pd.DataFrame(ts_da)

                    matched = df_match(ts_ol, ts_da, ts_asc, ts_ins, window=0.5)
                    data = ts_ol.join(matched[0]['DA_4K']).join(matched[1]['ascat']).join(matched[2]['insitu']).dropna()

                    dpr_triplets = data.corr()['DA_4K']['insitu'] - data.corr()['open_loop']['insitu']
                    if dpr_triplets < 0:
                        continue

                    f = plt.figure(figsize=(15, 5))
                    sns.lineplot(data=data[['open_loop', 'DA_4K', 'insitu']], dashes=False, linewidth=1.5, axes=plt.gca())
                    plt.title(f'{meta.network} / {meta.station} ({var}): d(Pearson R2) = {dpr_triplets:.3f} , d(TCA R2) = {dtr:.3f}')

                    fbase = Path('/Users/u0116961/Documents/work/MadKF/CLSM/suspicious_stations/timeseries')
                    fname = fbase / f'{mode}_{var}_{meta.network}_{meta.station}.png'
                    f.savefig(fname, dpi=300, bbox_inches='tight')
                    plt.close()

        except:
            continue

def plot_ismn_statistics(root):

    res = pd.read_csv(root / 'validation' / 'insitu_TCA.csv', index_col=0)
    res.index = res.network
    res.drop('network', axis='columns', inplace=True)
    # res2 = pd.read_csv(root / 'insitu.csv', index_col=0)
    # res2.index = res2.network
    # res2.drop('network', axis='columns', inplace=True)

    variables = ['sm_surface', 'sm_rootzone']
    var_labels = ['surface', 'root-zone']

    pc = 'Pcorr'

    runs = [f'OL_{pc}'] + [f'DA_{pc}_{err}' for err in ['4K', 'abs', 'anom_lt', 'anom_lst', 'anom_st']]
    run_labels = runs

    n_runs = len(runs)
    offsets = np.linspace(-0.5 + 1/(n_runs+1), 0.5 - 1/(n_runs+1),n_runs)
    cols = ['darkred', 'coral',
            'darkgreen', 'forestgreen', 'limegreen', 'lightgreen',
            'navy', 'mediumblue', 'slateblue', 'lightblue',
            'rebeccapurple', 'blueviolet', 'mediumorchid', 'plum'][0:n_runs]

    width = (offsets[1]- offsets[0]) / 2.5
    ss = offsets[1] - offsets[0]

    # offsets = [-0.2, 0.0, 0.2]
    # cols = ['lightblue', 'lightgreen', 'coral']
    fontsize = 16

    nets = ''

    networks  = ['SCAN', 'USCRN']
    res = res.loc[res.index.isin(networks),:]
    nets = '_SCANUSCRN'


    # titles = ['ubRMSD', 'ubRMSE', 'Pearson R$^2$ ', 'TCA R$^2$']
    titles = ['ubRMSD (DA-ISMN)', 'R$^2$ (DA-ISMN)', 'ubRMSE ', 'TCA R$^2$']

    ylims = [[0.00, 0.1],
             [0.0, 0.8],
             [0.0, 0.05],
             [0.2, 1.0]]

    modes = ['abs', 'anom_lt', 'anom_st', 'anom_lst']

    for mode in modes:

        f = plt.figure(figsize=(20,8))

        valss = [[[res[f'ubRMSD_model_insitu_{run}_{mode}_{var}'].values for run in runs] for var in variables],
                 [[res[f'R2_model_insitu_{run}_{mode}_{var}'].values ** 2 for run in runs] for var in variables],
                 [[res[f'ubRMSE_model_{run}_{mode}_{var}'].values for run in runs] for var in variables],
                 [[res[f'R2_model_{run}_{mode}_{var}'].values for run in runs] for var in variables]]

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

            box = ax.boxplot(data, whis=[5,95], showfliers=False, positions=pos, widths=width, patch_artist=True)
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
                plt.axvline(k+0.5, linewidth=1.5, color='k')
                plt.axvline(k+1 - 2*ss, linewidth=1, color='k', linestyle='--')
                plt.axvline(k+1 - 1*ss, linewidth=1, color='k', linestyle='--')
                # plt.axvline(k+1 + 0*ss, linewidth=1, color='k', linestyle='--')
            if n == 1:
                plt.figlegend((box['boxes'][0:n_runs]),run_labels,'upper right',fontsize=fontsize-4)
            ax.set_title(tit ,fontsize=fontsize)

        outdir = root / 'plots' / f'ismn_eval_{pc}'
        if not outdir.exists():
            Path.mkdir(outdir, parents=True)

        f.savefig(outdir / f'ismn_stats_{mode}{nets}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # plt.tight_layout()
    # plt.show()

def plot_ismn_results_map(root):

    # res = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/validation/insitu_TCA.csv', index_col=0)
    res = pd.read_csv(root / 'validation' / 'insitu_TCA.csv', index_col=0)
    res.index = res.network
    res.drop('network', axis='columns', inplace=True)

    # networks  = ['SCAN', 'USCRN']
    # res = res.loc[res.index.isin(networks),:]

    variables = ['sm_surface', 'sm_rootzone']
    var_labels = ['surface', 'root-zone']

    lats = res['latitude'].values
    lons = res['longitude'].values

    # runs = [f'OL_Pcorr', 'OL_noPcorr'] + [f'DA_{pc}_{err}' for pc in ['Pcorr', 'noPcorr'] for err in ['4K', 'abs', 'anom_lt', 'anom_lst', 'anom_st']]
    runs = [f'OL_Pcorr'] + [f'DA_Pcorr_{err}' for err in ['4K', 'abs', 'anom_lt', 'anom_lst', 'anom_st']]
    run_labels = runs
    n_runs = len(runs)

    fontsize = 16


    titles = ['ubRMSE', 'R$^2$ (DA-ASCAT)', 'R$^2$ (DA-ISMN) ', 'TCA R$^2$']

    ylims = [[0.0, 0.05],
             [0.0, 0.8],
             [0.0, 0.8],
             [0.2, 1.0]]

    modes = ['abs', 'anom_lt', 'anom_st', 'anom_lst']

    for mode in modes:

        f = plt.figure(figsize=(24,10))

        for i, run in enumerate(runs[1::]):

            ax = plt.subplot(2,3,i+1)

            llcrnrlat = 24
            urcrnrlat = 51
            llcrnrlon = -128
            urcrnrlon = -64

            m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                        llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, resolution='c', )
            m.drawcoastlines()
            m.drawcountries()
            m.drawstates()

            xs, ys = m(lons, lats)

            c = res[f'R2_model_{run}_{mode}_sm_surface'].values - res[f'R2_model_DA_Pcorr_4K_{mode}_sm_surface'].values
            # c = res[f'R2_model_{run}_{mode}_sm_surface'].values - res[f'R2_model_OL_Pcorr_{mode}_sm_surface'].values

            sc = plt.scatter(xs, ys, c=c, s=40, label='R2', vmin=-0.2, vmax=0.2, cmap=cc.cm.bjy)
            plt.title(run)

            cb = m.colorbar(sc, "bottom", size="7%", pad="5%")

            x, y = m(-79, 25)
            plt.text(x, y, 'm. = %.3f' % np.nanmedian(c), fontsize=fontsize - 2)

        fout = root / 'plots' / 'ismn_eval_Pcorr' / f'skill_gain_ref_4K_{mode}.png'
        # fout = root / 'plots' / 'ismn_eval_Pcorr' / f'skill_gain_ref_OL_{mode}.png'
        f.savefig(fout, dpi=300, bbox_inches='tight')
        plt.close()

    # plt.tight_layout()
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

    root = Path(f'/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas')

    # if not (root / 'plots').exists():
    #     Path.mkdir((root / 'plots'), parents=True)


    # exp = 'US_M36_SMAP_TB_DA_scl_SMOSSMAP_short'
    # exp = 'US_M36_SMOS40_TB_MadKF_DA_it614'
    # exp = 'US_M36_SMAP_TB_MadKF_DA_it34'

    # lat, lon = 37.573933, -96.840000 # Kansas
    # lat, lon = 44.434550, -99.703901 # South Dakota
    # lat, lon = 41.203456192, -102.249755859 # Nebraska

    # plot_ts(lon, lat)

    # plot_ismn_statistics(root)
    # plot_ismn_statistics_v2()

    plot_ismn_statistics(root)

    # find_suspicious_stations(root)
    # plot_suspicious_stations(root)

    # plot_ismn_results_map(root)
    # plot_filter_diagnostics(root)
    # plot_filter_diagnostics_violins(root)

    # plot_improvement_vs_uncertainty_update(iteration)

