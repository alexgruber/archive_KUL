
import os
os.environ["PROJ_LIB"] = '/Users/u0116961/opt/miniconda3/pkgs/proj4-5.2.0-h6de7cb9_1006/share/proj'

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from pathlib import Path

from pyldas.interface import LDAS_io

from pyldas.templates import template_error_Tb40

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.ndimage import gaussian_filter

from myprojects.timeseries import calc_anom

def tca(df):
    cov = df.dropna().cov().values
    ind = (0, 1, 2, 0, 1, 2)
    err_var = np.array([np.abs(cov[i, i] - (cov[i, ind[i + 1]] * cov[i, ind[i + 2]]) / cov[ind[i + 1], ind[i + 2]])
                        for i in np.arange(3)])
    return err_var


def calc_tb_mse(root, iteration, anomaly=False, longterm=False):

    # exp_ol = f'US_M36_SMOS40_TB_MadKF_OL_it{iteration}'
    # exp_da = f'US_M36_SMOS40_TB_MadKF_DA_it{iteration}'

    exp_ol = 'US_M36_SMAP_TB_OL_noScl'
    exp_da = 'US_M36_SMAP_TB_DA_scl_SMOSSMAP_short'

    param = 'ObsFcstAna'

    io_ol = LDAS_io(param, exp_ol)
    io_da = LDAS_io(param, exp_da)

    res = pd.DataFrame(index=io_ol.grid.tilecoord.index.values,
                       columns=['col', 'row'] + [f'mse_obs_spc{spc}' for spc in [1,2,3,4]] \
                                              + [f'mse_fcst_spc{spc}' for spc in [1,2,3,4]] \
                                              + [f'mse_ana_spc{spc}' for spc in [1,2,3,4]])

    ids = io_ol.grid.tilecoord['tile_id'].values
    res.loc[:, 'col'], res.loc[:, 'row'] = np.vectorize(io_ol.grid.tileid2colrow)(ids, local_cs=False)

    for cnt, (idx, val) in enumerate(io_ol.grid.tilecoord.iterrows()):
        print(f'{cnt} / {len(res)}')

        col, row = io_ol.grid.tileid2colrow(val.tile_id)

        for spc in [1, 2, 3, 4]:
            ts_fcst = io_ol.read_ts('obs_fcst', col, row, species=spc, lonlat=False).dropna()
            ts_obs = io_da.read_ts('obs_obs', col, row, species=spc, lonlat=False).dropna()
            ts_ana = io_da.read_ts('obs_ana', col, row, species=spc, lonlat=False).dropna()
            if len(ts_ana) == 0:
                continue
            ts_fcst = ts_fcst.reindex(ts_ana.index)

            if anomaly is True:
                ts_fcst = calc_anom(ts_fcst, longterm=longterm, window_size=45).dropna()
                ts_obs = calc_anom(ts_obs, longterm=longterm, window_size=45).dropna()
                ts_ana = calc_anom(ts_ana, longterm=longterm, window_size=45).dropna()

            df = pd.concat((ts_obs,ts_fcst,ts_ana),axis=1).dropna()
            tc_res = tca(df)

            res.loc[idx,f'mse_obs_spc{spc}'] = tc_res[0]
            res.loc[idx,f'mse_fcst_spc{spc}'] = tc_res[1]
            res.loc[idx,f'mse_ana_spc{spc}'] = tc_res[2]

    fname = root / 'result_files' / 'mse.csv'

    res.to_csv(fname, float_format='%0.8f')

def calc_ens_var(root, iteration):

    exp_ol = 'US_M36_SMAP_TB_OL_noScl'
    exp_da = 'US_M36_SMAP_TB_DA_scl_SMOSSMAP_short'

    # exp_ol = f'US_M36_SMOS40_TB_MadKF_OL_it{iteration}'
    # exp_da = f'US_M36_SMOS40_TB_MadKF_DA_it{iteration}'

    # param = 'ObsFcstAnaEns'
    param = 'ObsFcstAna'

    io_ol = LDAS_io(param, exp_ol)
    io_da = LDAS_io(param, exp_da)

    res = pd.DataFrame(index=io_ol.grid.tilecoord.index.values,
                       columns=['col', 'row'] + [f'obs_var_spc{spc}' for spc in [1,2,3,4]] \
                                              + [f'fcst_var_spc{spc}' for spc in [1,2,3,4]] \
                                              + [f'ana_var_spc{spc}' for spc in [1,2,3,4]])

    ids = io_ol.grid.tilecoord['tile_id'].values
    res.loc[:, 'col'], res.loc[:, 'row'] = np.vectorize(io_ol.grid.tileid2colrow)(ids, local_cs=False)

    for cnt, (idx, val) in enumerate(io_ol.grid.tilecoord.iterrows()):
        print('%i / %i' % (cnt, len(res)))

        col, row = io_ol.grid.tileid2colrow(val.tile_id)

        for spc in [1, 2, 3, 4]:
            # ts_fcst = io_ol.read_ts('obs_fcst', col, row, species=spc, lonlat=False).dropna()
            # ts_obs = io_da.read_ts('obs_obs', col, row, species=spc, lonlat=False).dropna()
            # ts_ana = io_da.read_ts('obs_ana', col, row, species=spc, lonlat=False).dropna()
            # if len(ts_ana) == 0:
            #     continue
            # ts_fcst = ts_fcst.reindex(ts_ana.index)
            # ts_obs = ts_obs.reindex(ts_ana.index)
            # res.loc[idx,f'obs_var_spc{spc}'] = ts_obs.var(axis='columns').mean()
            # res.loc[idx,f'fcst_var_spc{spc}'] = ts_fcst.var(axis='columns').mean()
            # res.loc[idx,f'ana_var_spc{spc}'] = ts_ana.var(axis='columns').mean()

            res.loc[idx,f'obs_var_spc{spc}'] = np.nanmean(io_da.timeseries['obs_obsvar'][:,spc-1,row,col].values)
            res.loc[idx,f'fcst_var_spc{spc}'] = np.nanmean(io_ol.timeseries['obs_fcstvar'][:,spc-1,row,col].values)
            res.loc[idx,f'ana_var_spc{spc}'] = np.nanmean(io_da.timeseries['obs_anavar'][:,spc-1,row,col].values)

    fname = root / 'result_files' / 'ens_var.csv'

    res.to_csv(fname, float_format='%0.8f')


def smooth_parameters(root):

    for file in ['ens_var', 'ens_cov', 'mse']:
    # for file in ['mse',]:

        infile = root / 'result_files' / (file + '.csv')
        outfile = root / 'result_files' / 'smoothed' / (file + '.csv')

        if not outfile.parent.exists():
            Path.mkdir(outfile.parent, parents=True)

        res = pd.read_csv(infile,index_col=0)
        tags = res.columns.drop(['col','row']).values

        res = fill_gaps(res, tags, smooth=True)

        res.to_csv(outfile, float_format='%.6f')



def calc_ens_cov(root, iteration):

    exp_ol = f'US_M36_SMOS40_TB_MadKF_OL_it{iteration}'
    exp_da = f'US_M36_SMOS40_TB_MadKF_DA_it{iteration}'

    param = 'ObsFcstAnaEns'

    io_ol = LDAS_io(param, exp_ol)
    io_da = LDAS_io(param, exp_da)

    res = pd.DataFrame(index=io_ol.grid.tilecoord.index.values, columns=['col', 'row'])

    ids = io_ol.grid.tilecoord['tile_id'].values
    res.loc[:, 'col'], res.loc[:, 'row'] = np.vectorize(io_ol.grid.tileid2colrow)(ids, local_cs=False)

    for spc in [1,2,3,4]:

        for cnt, (idx,val) in enumerate(io_ol.grid.tilecoord.iterrows()):
            print('spc %i: %i / %i' % (spc, cnt, len(res)))

            col, row = io_ol.grid.tileid2colrow(val.tile_id)

            ts_ol = io_ol.read_ts('obs_fcst', col, row, species=spc, lonlat=False).dropna()
            ts_obs = io_da.read_ts('obs_obs', col, row, species=spc, lonlat=False).dropna()
            ts_ana = io_da.read_ts('obs_ana', col, row, species=spc, lonlat=False).dropna()
            if len(ts_ana) == 0:
                continue
            ts_ol = ts_ol.reindex(ts_ana.index)
            ts_obs = ts_obs.reindex(ts_ana.index)

            tmp_c_ol_obs = np.full(len(ts_ol),np.nan)
            tmp_c_ol_ana = np.full(len(ts_ol),np.nan)
            tmp_c_obs_ana = np.full(len(ts_ol),np.nan)

            for i in np.arange(len(ts_ol)):
                try:
                    tmp_c_ol_obs[i] = np.cov(ts_ol.iloc[i,:],ts_obs.iloc[i,:])[0,1]
                    tmp_c_ol_ana[i] = np.cov(ts_ol.iloc[i,:],ts_ana.iloc[i,:])[0,1]
                    tmp_c_obs_ana[i] = np.cov(ts_obs.iloc[i,:],ts_ana.iloc[i,:])[0,1]
                except:
                    pass

            res.loc[idx,f'c_ol_obs_spc{spc}'] = np.nanmean(tmp_c_ol_obs)
            res.loc[idx,f'c_ol_ana_spc{spc}'] = np.nanmean(tmp_c_ol_ana)
            res.loc[idx,f'c_obs_ana_spc{spc}'] = np.nanmean(tmp_c_obs_ana)

    fname = root / 'result_files' / 'ens_cov.csv'

    res.to_csv(fname, float_format='%0.8f')

def plot_ease_img2(data, tag,
                  llcrnrlat=24,
                  urcrnrlat=51,
                  llcrnrlon=-128,
                  urcrnrlon=-64,
                  cbrange=(-20, 20),
                  cmap='jet',
                  title='',
                  fontsize=16):

    io = LDAS_io()

    tc = io.grid.tilecoord

    lons, lats = np.meshgrid(io.grid.ease_lons, io.grid.ease_lats)

    img = np.full(lons.shape, np.nan)

    ind_lat = tc.loc[data.index.values, 'j_indg']
    ind_lon = tc.loc[data.index.values, 'i_indg']

    img[ind_lat, ind_lon] = data[tag]
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

    if title != '':
        plt.title(title,fontsize=fontsize)

    x, y = m(-79, 25)
    plt.text(x, y, 'mean = %.3f' % np.ma.mean(img_masked), fontsize=fontsize - 2)


def plot_ease_img(data,tag,
                  llcrnrlat=24,
                  urcrnrlat=51,
                  llcrnrlon=-128,
                  urcrnrlon=-64,
                  cbrange=(-20,20),
                  cmap='jet',
                  plot_cb =True,
                  title='',
                  fontsize=12,
                  sqrt=False):

    grid = LDAS_io().grid
    lons,lats = np.meshgrid(grid.ease_lons, grid.ease_lats)

    ind_lat = data['row'].values.astype('int')
    ind_lon = data['col'].values.astype('int')
    img = np.full(lons.shape, np.nan)

    img[ind_lat,ind_lon] = np.abs(data[tag]) if sqrt is False else np.sqrt(np.abs(data[tag]))
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
            t.set_fontsize(fontsize-2)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(fontsize-2)

    if title != '':
        plt.title(title,fontsize=fontsize)

    x, y = m(-79, 25)
    plt.text(x, y, 'mean = %.2f' % np.ma.mean(img_masked), fontsize=fontsize - 4)

def plot_ens_cov(root, last_it, mode='', sub=''):

    fontsize = 16

    fname = root / 'result_files' / 'ens_cov.csv'
    res = pd.read_csv(fname, index_col=0).dropna()

    # for corrected in [True]:
    for corrected in [False, True]:

        if corrected:
            fname = Path(f'{str(root).split("_")[0]}_{last_it}') / mode / 'result_files' / sub / 'mse.csv'
            res0 = pd.read_csv(fname, index_col=0).dropna()

            fname = root / 'result_files' / 'ens_var.csv'
            res1 = pd.read_csv(fname, index_col=0).dropna()

            fname_out = root / 'plots' / 'ens_cov_corrected.png'
        else:
            fname_out = root / 'plots' / 'ens_cov_uncorrected.png'

        plt.figure(figsize=(25,11))

        cmap='jet'

        cbrange1 = [-1, 1]
        cbrange2 = [0, 16]

        for spc in [1,2,3,4]:

            if spc == 1:
                spc_tit = 'H pol. / Asc.'
            elif spc == 2:
                spc_tit = 'H pol. / Dsc.'
            elif spc == 3:
                spc_tit = 'V pol. / Asc.'
            else:
                spc_tit = 'V pol. / Dsc.'

            if corrected:
                scl_obs =  res0['mse_obs_spc%i' % spc] / res1['obs_var_spc%i' % spc]
                scl_fcst =  res0['mse_fcst_spc%i' % spc] / res1['fcst_var_spc%i' % spc]
                scl_ana =  res0['mse_ana_spc%i' % spc] / res1['ana_var_spc%i' % spc]

                res['c_ol_obs_spc%i' % spc] *= np.sqrt(scl_fcst*scl_obs)
                res['c_ol_ana_spc%i' % spc] *= np.sqrt(scl_fcst*scl_ana)
                res['c_obs_ana_spc%i' % spc] *= np.sqrt(scl_obs*scl_ana)

            plt.subplot(3,4,spc)
            plot_ease_img(res, 'c_ol_obs_spc%i'%spc, cbrange=cbrange1, title=spc_tit, fontsize=fontsize, cmap=cmap)
            if spc == 1:
                plt.ylabel('OL - Obs.', fontsize=fontsize)

            plt.subplot(3,4,spc+4)
            plot_ease_img(res, 'c_ol_ana_spc%i'%spc, cbrange=cbrange1, title='', fontsize=fontsize, cmap=cmap)
            if spc == 1:
                plt.ylabel('OL - Ana.', fontsize=fontsize)

            plt.subplot(3,4,spc+8)
            plot_ease_img(res, 'c_obs_ana_spc%i'%spc, cbrange=cbrange2, title='', fontsize=fontsize, cmap=cmap)
            if spc == 1:
                plt.ylabel('Obs. - Ana.', fontsize=fontsize)

        # plt.tight_layout()
        # plt.show()

        plt.savefig(fname_out, dpi=plt.gcf().dpi, bbox_inches='tight')
        plt.close()


def correct_mse(root, last_it, mode='', sub=''):

    xroot = Path(f'{str(root).split("_")[0]}_{last_it}') / mode / 'result_files' / sub
    if not (fname := (xroot  / 'mse_corrected.csv')).exists():
        print(f'WARNING: mse_corrected.csv does not exist for iteration {last_it}... Using mse.csv instead.')
        fname = xroot  / 'mse.csv'
    res0 = pd.read_csv(fname, index_col=0)

    xroot = root / 'result_files'
    res = pd.read_csv(xroot / 'mse.csv', index_col=0)
    res1 = pd.read_csv(xroot / 'ens_var.csv', index_col=0)
    cov = pd.read_csv(xroot / 'ens_cov.csv', index_col=0)

    fname_out = xroot / 'mse_corrected.csv'

    for spc in [1, 2, 3, 4]:
        scl_obs = res0[f'mse_obs_spc{spc}'] / res1[f'obs_var_spc{spc}']
        scl_fcst = res0[f'mse_fcst_spc{spc}'] / res1[f'fcst_var_spc{spc}']
        scl_ana = res0[f'mse_ana_spc{spc}'] / res1[f'ana_var_spc{spc}']

        cov[f'c_ol_obs_spc{spc}'] *= np.sqrt(scl_fcst * scl_obs)
        cov[f'c_ol_ana_spc{spc}'] *= np.sqrt(scl_fcst * scl_ana)
        cov[f'c_obs_ana_spc{spc}'] *= np.sqrt(scl_obs * scl_ana)

        res[f'mse_obs_spc{spc}'] = np.abs(res[f'mse_obs_spc{spc}'] - cov[f'c_ol_ana_spc{spc}'] + cov[f'c_obs_ana_spc{spc}'])
        res[f'mse_fcst_spc{spc}'] = np.abs(res[f'mse_fcst_spc{spc}'] + cov[f'c_ol_ana_spc{spc}'] - cov[f'c_obs_ana_spc{spc}'])

    res.to_csv(fname_out)


def plot_mse(root, sub=''):

    fontsize = 16

    for corrected in [False, True]:
    # for corrected in [False,]:

        if corrected:
            fname_in = root / 'result_files' / sub / 'mse_corrected.csv'
            fname_out = root / 'plots' / sub / 'mse_corrected.png'
        else:
            fname_in = root / 'result_files' / sub / 'mse.csv'
            fname_out = root / 'plots' / sub / 'mse_uncorrected.png'

        res = pd.read_csv(fname_in, index_col=0)

        plt.figure(figsize=(25,11))
        cbrange = [0, 12]
        cmap='jet'

        for spc in [1, 2, 3, 4]:

            if spc == 1:
                spc_tit = 'H pol. / Asc.'
            elif spc == 2:
                spc_tit = 'H pol. / Dsc.'
            elif spc == 3:
                spc_tit = 'V pol. / Asc.'
            else:
                spc_tit = 'V pol. / Dsc.'

            plt.subplot(3,4,spc)
            plot_ease_img(res, 'mse_obs_spc%i'%spc, cbrange=cbrange, title=spc_tit, cmap=cmap, sqrt=True, fontsize=fontsize)
            if spc == 1:
                plt.ylabel('ubRMSE Observation', fontsize=fontsize)

            plt.subplot(3,4,spc+4)
            plot_ease_img(res, 'mse_fcst_spc%i'%spc, cbrange=cbrange, title='', cmap=cmap, sqrt=True, fontsize=fontsize)
            if spc == 1:
                plt.ylabel('ubRMSE Open Loop', fontsize=fontsize)

            plt.subplot(3,4,spc+8)
            plot_ease_img(res, 'mse_ana_spc%i'%spc, cbrange=cbrange, title='', cmap=cmap, sqrt=True, fontsize=fontsize)
            if spc == 1:
                plt.ylabel('ubRMSE Analysis', fontsize=fontsize)

        # plt.tight_layout()
        # plt.show()

        plt.savefig(fname_out, dpi=plt.gcf().dpi, bbox_inches='tight')
        plt.close()


def plot_ens_var(root, smoothed=False):

    sub = 'smoothed' if smoothed else ''

    fname = root / 'result_files' / sub / 'ens_var.csv'
    res = pd.read_csv(fname, index_col=0).dropna()

    plt.figure(figsize=(24,11))

    cmap='jet'

    cbrange = [0,40]
    plt.subplot(341)
    plot_ease_img(res, 'obs_var_spc1', cbrange=cbrange, title='Obs. Ens. Var. Spc1', cmap=cmap)
    plt.subplot(342)
    plot_ease_img(res, 'obs_var_spc2', cbrange=cbrange, title='Obs. Ens. Var. Spc2', cmap=cmap)
    plt.subplot(343)
    plot_ease_img(res, 'obs_var_spc3', cbrange=cbrange, title='Obs. Ens. Var. Spc3', cmap=cmap)
    plt.subplot(344)
    plot_ease_img(res, 'obs_var_spc4', cbrange=cbrange, title='Obs. Ens. Var. Spc4', cmap=cmap)

    cbrange = [0,40]
    plt.subplot(345)
    plot_ease_img(res, 'fcst_var_spc1', cbrange=cbrange, title='Fcst. Ens. Var. Spc1', cmap=cmap)
    plt.subplot(346)
    plot_ease_img(res, 'fcst_var_spc2', cbrange=cbrange, title='Fcst. Ens. Var. Spc2', cmap=cmap)
    plt.subplot(347)
    plot_ease_img(res, 'fcst_var_spc3', cbrange=cbrange, title='Fcst. Ens. Var. Spc3', cmap=cmap)
    plt.subplot(348)
    plot_ease_img(res, 'fcst_var_spc4', cbrange=cbrange, title='Fcst. Ens. Var. Spc4', cmap=cmap)

    cbrange = [0,10]
    plt.subplot(3,4,9)
    plot_ease_img(res, 'ana_var_spc1', cbrange=cbrange, title='Ana. Ens. Var. Spc1', cmap=cmap)
    plt.subplot(3,4,10)
    plot_ease_img(res, 'ana_var_spc2', cbrange=cbrange, title='Ana. Ens. Var. Spc2', cmap=cmap)
    plt.subplot(3,4,11)
    plot_ease_img(res, 'ana_var_spc3', cbrange=cbrange, title='Ana. Ens. Var. Spc3', cmap=cmap)
    plt.subplot(3,4,12)
    plot_ease_img(res, 'ana_var_spc4', cbrange=cbrange, title='Ana. Ens. Var. Spc4', cmap=cmap)

    # plt.tight_layout()
    # plt.show()

    plt.savefig(root / 'plots' / sub / 'ens_var.png', dpi=plt.gcf().dpi, bbox_inches='tight')
    plt.close()


def plot_obs_pert(root, smoothed=False):

    sub = 'smoothed' if smoothed else ''

    fname = root / 'result_files' / sub / 'ens_var.csv'
    ensvar = pd.read_csv(fname, index_col=0)

    fname = root / 'result_files' / sub / 'mse_corrected.csv'
    mse = pd.read_csv(fname, index_col=0)

    res = ensvar[['col','row']]

    for spc in np.arange(1,5):
        res['obs_var_spc%i'%spc] = ensvar['fcst_var_spc%i'%spc] * mse['mse_obs_spc%i'%spc] / mse['mse_fcst_spc%i'%spc]

        res.loc[(res['obs_var_spc%i' % spc] < 1), 'obs_var_spc%i' % spc] = 1
        res.loc[(res['obs_var_spc%i' % spc] > 100), 'obs_var_spc%i' % spc] = 100

        res.loc[:, 'obs_var_spc%i' % spc] **= 0.5

        # res.loc[np.isnan(res['obs_var_spc%i'%spc]),'obs_var_spc%i'%spc] = res['obs_var_spc%i'%spc].median()

    cmap = 'jet'
    cbrange = [0,10]

    plt.figure(figsize=(14,8))

    plt.subplot(221)
    plot_ease_img(res, 'obs_var_spc1', cbrange=cbrange, title='Species 1', cmap=cmap)

    plt.subplot(222)
    plot_ease_img(res, 'obs_var_spc2', cbrange=cbrange, title='Species 2', cmap=cmap)

    plt.subplot(223)
    plot_ease_img(res, 'obs_var_spc3', cbrange=cbrange, title='Species 3', cmap=cmap)

    plt.subplot(224)
    plot_ease_img(res, 'obs_var_spc4', cbrange=cbrange, title='Species 4', cmap=cmap)

    # plt.tight_layout()
    # plt.show()

    plt.savefig(root / 'plots' / sub / 'obs_pert.png', dpi=plt.gcf().dpi, bbox_inches='tight')
    plt.close()


def fill_gaps(xres, tags, smooth=False):

    res = xres.copy()

    grid = LDAS_io().grid
    lons, lats = np.meshgrid(grid.ease_lons, grid.ease_lats)

    ind_lat = res['row'].values.astype('int')
    ind_lon = res['col'].values.astype('int')

    imp = IterativeImputer(max_iter=10, random_state=0)
    for tag in np.atleast_1d(tags):
        img = np.full(lons.shape, np.nan)
        img[ind_lat, ind_lon] = res[tag]

        # find all non-zero values
        idx = np.where(~np.isnan(img))
        vmin, vmax = np.percentile(img[idx], [2.5, 97.5])
        img[img<vmin] = vmin
        img[img>vmax] = vmax

        # calculate fitting parameters
        imp.set_params(min_value=vmin, max_value=vmax)
        imp.fit(img)

        # Define an anchor pixel to infer fitted image dimensions
        tmp_img = img.copy()
        tmp_img[idx[0][100],idx[1][100]] = 1000000

        # transform image with and without anchor pixel
        tmp_img_fitted = imp.transform(tmp_img)
        img_fitted = imp.transform(img)

        # # Get indices of fitted image
        idx_anchor = np.where(tmp_img_fitted == 1000000)[1][0]
        start = idx[1][100] - idx_anchor
        end = start + img_fitted.shape[1]

        # write output
        img[:,start:end] = img_fitted

        if smooth:
            img = gaussian_filter(img, sigma=0.6, truncate=1)

        res.loc[:, tag] = img[ind_lat, ind_lon]
        res.loc[:, tag] = res.loc[:, tag].replace(np.nan, res.loc[:, tag].median())

    return res

def write_spatial_errors(root, iteration, gapfilled=True, smooth=False):

    froot = root / 'error_files'
    fbase = 'SMOS_fit_Tb_'

    exp = 'US_M36_SMAP_TB_DA_scl_SMOSSMAP_short'
    # exp = 'US_M36_SMAP_TB_MadKF_DA_it%i' % iteration
    io = LDAS_io('ObsFcstAna', exp)

    fname = root / 'result_files' / 'ens_var.csv'
    ensvar = pd.read_csv(fname, index_col=0)

    # fname = root / 'result_files' / 'mse_corrected.csv'
    fname = root / 'result_files' / 'mse.csv'                  #TODO: SHOULD BE MSE_CORRECTED!!
    mse = pd.read_csv(fname, index_col=0)

    obs_err = ensvar[['col','row']]
    obs_err.loc[:, 'tile_id'] = io.grid.tilecoord.loc[obs_err.index, 'tile_id'].values

    for spc in np.arange(1,5):
        obs_err.loc[:,'obs_var_spc%i'%spc] = ensvar['fcst_var_spc%i'%spc] * mse['mse_obs_spc%i'%spc] / mse['mse_fcst_spc%i'%spc]
        obs_err.loc[(obs_err['obs_var_spc%i' % spc] < 1), 'obs_var_spc%i' % spc] = 1
        obs_err.loc[(obs_err['obs_var_spc%i' % spc] > 1600), 'obs_var_spc%i' % spc] = 1600
        # obs_err.loc[np.isnan(obs_err['obs_var_spc%i'%spc]),'obs_var_spc%i'%spc] = obs_err['obs_var_spc%i'%spc].median()
        obs_err.loc[:, 'obs_var_spc%i' % spc] **= 0.5

        if gapfilled:
            obs_err.loc[:, 'obs_var_spc%i' % spc] = fill_gaps(obs_err, 'obs_var_spc%i' % spc, smooth=smooth)['obs_var_spc%i' % spc]

    dtype = template_error_Tb40()[0]

    angles = np.array([40.,])
    orbits = ['A', 'D']

    template = pd.DataFrame(columns=dtype.names).astype('float32')
    template['lon'] = io.grid.tilecoord['com_lon'].values.astype('float32')
    template['lat'] = io.grid.tilecoord['com_lat'].values.astype('float32')
    template.index += 1

    modes = np.array([0, 0])
    sdate = np.array([2015, 4, 1, 0, 0])
    edate = np.array([2020, 4, 1, 0, 0])
    lengths = np.array([len(template), len(angles)])  # tiles, incidence angles, whatever

    # ----- write output files -----
    for orb in orbits:
        # !!! inconsistent with the definition in the obs_paramfile (species) !!!
        modes[0] = 1 if orb == 'A' else 0

        res = template.copy()

        spc = 0 if orb == 'A' else 1
        res.loc[:,'err_Tbh'] = obs_err.loc[res.index,'obs_var_spc%i'%(spc+1)].values

        spc = 2 if orb == 'A' else 3
        res.loc[:,'err_Tbv'] = obs_err.loc[res.index,'obs_var_spc%i'%(spc+1)].values

        fname = froot / (fbase + orb + '.bin')

        fid = open(fname, 'wb')
        io.write_fortran_block(fid, modes)
        io.write_fortran_block(fid, sdate)
        io.write_fortran_block(fid, edate)
        io.write_fortran_block(fid, lengths)
        io.write_fortran_block(fid, angles)

        for f in res.columns.values:
            io.write_fortran_block(fid, res[f].values)
        fid.close()

def plot_P_R_check(root, last_it):

    xroot = Path(f'{str(root).split("_")[0]}_{last_it}')
    fname = xroot / 'result_files' / 'mse.csv'
    res0 = pd.read_csv(fname, index_col=0).dropna()

    fname = root / 'result_files' / 'ens_var.csv'
    res1 = pd.read_csv(fname, index_col=0).dropna()

    plt.figure(figsize=(24,11))

    cmap='jet'

    for spc in [1,2,3,4]:

        cbrange = [0,15]
        res0['ratio_mse_spc%i'%spc] = res0['mse_fcst_spc%i'%spc] / res0['mse_obs_spc%i'%spc]
        plt.subplot(2,4,spc+4)
        plot_ease_img(res0, 'ratio_mse_spc%i'%spc, cbrange=cbrange, title='P / R (mse; spc%i)'%spc, cmap=cmap)

        cbrange = [0,15]
        res1['ratio_ensvar_spc%i'%spc] = res1['fcst_var_spc%i'%spc] / res1['obs_var_spc%i'%spc]
        plt.subplot(2,4,spc)
        plot_ease_img(res1, 'ratio_ensvar_spc%i'%spc, cbrange=cbrange, title='P / R (ensvar; spc%i)'%spc, cmap=cmap)

    plt.tight_layout()

    # plt.show()
    plt.savefig(root / 'plots' / 'P_R_ratio.png', dpi=plt.gcf().dpi)
    plt.close()


def plot_P_R_scl(root, last_it):

    xroot = Path(f'{str(root).split("_")[0]}_{last_it}')
    fname = xroot / 'result_files' / 'mse.csv'
    res0 = pd.read_csv(fname, index_col=0).dropna()

    fname = root / 'result_files' / 'ens_var.csv'
    res1 = pd.read_csv(fname, index_col=0).dropna()

    plt.figure(figsize=(24,11))

    cmap='jet'

    for spc in [1,2,3,4]:

        cbrange = [0,2]

        res0['var_scl_fcst%i'%spc] = res1['fcst_var_spc%i'%spc] / res0['mse_fcst_spc%i'%spc]
        plt.subplot(3,4,spc)
        plot_ease_img(res0, 'var_scl_fcst%i'%spc, cbrange=cbrange, title='Fcst. scale factor (spc%i)'%spc, cmap=cmap)

        cbrange = [0,2]
        res0['var_scl_obs%i'%spc] = res1['obs_var_spc%i'%spc] / res0['mse_obs_spc%i'%spc]
        plt.subplot(3,4,spc+4)
        plot_ease_img(res0, 'var_scl_obs%i'%spc, cbrange=cbrange, title='Obs. scale factor (spc%i)'%spc, cmap=cmap)

        res0['cov_scl%i'%spc] = np.sqrt((res1['fcst_var_spc%i'%spc] / res0['mse_fcst_spc%i'%spc]) * (res1['obs_var_spc%i'%spc] / res0['mse_obs_spc%i'%spc]))
        plt.subplot(3,4,spc+8)
        plot_ease_img(res0, 'cov_scl%i'%spc, cbrange=cbrange, title='Cov. scale factor (spc%i)'%spc, cmap=cmap)

    plt.tight_layout()

    # plt.show()
    plt.savefig(root / 'plots' / 'P_R_scalefactor.png', dpi=plt.gcf().dpi)
    plt.close()



def plot_perturbations(root):

    # sub = 'smoothed' if smoothed else 'gapfilled' if gapfilled else 'raw'
    sub = ''

    fA = root / 'error_files' / sub / 'SMOS_fit_Tb_A.bin'
    fD = root / 'error_files' / sub / 'SMOS_fit_Tb_D.bin'

    dtype, hdr, length = template_error_Tb40()

    io = LDAS_io()

    imgA = io.read_fortran_binary(fA, dtype, hdr=hdr, length=length)
    imgD = io.read_fortran_binary(fD, dtype, hdr=hdr, length=length)

    imgA.index += 1
    imgD.index += 1

    cbrange = [0,5]

    plt.figure(figsize=(20, 10))

    plt.subplot(221)
    plot_ease_img2(imgA,'err_Tbh', cbrange=cbrange, title='H-pol (Asc.)')
    plt.subplot(222)
    plot_ease_img2(imgA,'err_Tbv', cbrange=cbrange, title='V-pol (Asc.)')
    plt.subplot(223)
    plot_ease_img2(imgD,'err_Tbh', cbrange=cbrange, title='H-pol (Dsc.)')
    plt.subplot(224)
    plot_ease_img2(imgD,'err_Tbv', cbrange=cbrange, title='V-pol (Dsc.)')

    # plt.tight_layout()
    # plt.show()

    plt.savefig(root / 'plots' / 'perturbations.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__=='__main__':

    curr_it = 13
    anomaly = False
    longterm = False

    # last_it = 611

    root = Path(f'~/Documents/work/MadKF/CLSM/SMAP/iter_{curr_it}').expanduser()

    if not (root / 'result_files').exists():
        Path.mkdir(root / 'result_files', parents=True)
    if not (root / 'plots' ).exists():
        Path.mkdir(root / 'plots', parents=True)
    if not (root / 'error_files' ).exists():
        Path.mkdir(root / 'error_files' , parents=True)

    # calc_tb_mse(root, curr_it, anomaly=anomaly, longterm=longterm)

    # calc_ens_var(root, curr_it)
    # calc_ens_cov(root, curr_it)
    # smooth_parameters(root)

    # correct_mse(root, last_it, mode=mode, sub=sub)
    # plot_mse(root)

    # plot_ens_var(root)
    # plot_ens_cov(root, last_it, mode=mode, sub=sub)
    # plot_obs_pert(root)

    # plot_P_R_check(curr_it, last_it)
    # plot_P_R_scl(curr_it, last_it)

    write_spatial_errors(root, curr_it)
    plot_perturbations(root)

    # list(map(plot_perturbations, ('532', '533')))


