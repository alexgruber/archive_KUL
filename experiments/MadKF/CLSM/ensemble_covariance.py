

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from pathlib import Path

from pyldas.interface import LDAS_io

from pyldas.templates import template_error_Tb40

def tca(df):
    cov = df.dropna().cov().values
    ind = (0, 1, 2, 0, 1, 2)
    err_var = np.array([np.abs(cov[i, i] - (cov[i, ind[i + 1]] * cov[i, ind[i + 2]]) / cov[ind[i + 1], ind[i + 2]])
                        for i in np.arange(3)])
    return err_var


def calc_tb_mse(root, iteration):

    # exp_ol = 'US_M36_SMOS40_TB_ens_test_OL'
    # exp_da = 'US_M36_SMOS40_TB_MadKF_it%i' % iteration
    exp_ol = 'US_M36_SMOS40_TB_MadKF_OL_it%i' % iteration
    exp_da = 'US_M36_SMOS40_TB_MadKF_DA_it%i' % iteration

    param = 'ObsFcstAna'

    io_ol = LDAS_io(param, exp_ol)
    io_da = LDAS_io(param, exp_da)

    res = pd.DataFrame(index=io_ol.grid.tilecoord.index.values,
                       columns=['col', 'row'] + ['mse_obs_spc%i'%spc for spc in [1,2,3,4]] \
                                              + ['mse_fcst_spc%i'%spc for spc in [1,2,3,4]] \
                                              + ['mse_ana_spc%i'%spc for spc in [1,2,3,4]])

    ids = io_ol.grid.tilecoord['tile_id'].values
    res.loc[:, 'col'], res.loc[:, 'row'] = np.vectorize(io_ol.grid.tileid2colrow)(ids, local_cs=False)

    for cnt, (idx, val) in enumerate(io_ol.grid.tilecoord.iterrows()):
        print('%i / %i' % (cnt, len(res)))

        col, row = io_ol.grid.tileid2colrow(val.tile_id)

        for spc in [1, 2, 3, 4]:
            ts_fcst = io_ol.read_ts('obs_fcst', col, row, species=spc, lonlat=False).dropna()
            ts_obs = io_da.read_ts('obs_obs', col, row, species=spc, lonlat=False).dropna()
            ts_ana = io_da.read_ts('obs_ana', col, row, species=spc, lonlat=False).dropna()
            if len(ts_ana) == 0:
                continue
            ts_fcst = ts_fcst.loc[ts_ana.index]
            ts_obs = ts_obs.loc[ts_ana.index]

            tc_res = tca(pd.concat((ts_obs,ts_fcst,ts_ana),axis=1))

            res.loc[idx,'mse_obs_spc%i'%spc] = tc_res[0]
            res.loc[idx,'mse_fcst_spc%i'%spc] = tc_res[1]
            res.loc[idx,'mse_ana_spc%i'%spc] = tc_res[2]

    fname = root / 'result_files' / 'mse.csv'

    res.to_csv(fname, float_format='%0.8f')

def calc_ens_var(root, iteration):

    # exp_ol = 'US_M36_SMOS40_TB_ens_test_OL'
    # exp_da = 'US_M36_SMOS40_TB_MadKF_it%i' % iteration
    exp_ol = 'US_M36_SMOS40_TB_MadKF_OL_it%i' % iteration
    exp_da = 'US_M36_SMOS40_TB_MadKF_DA_it%i' % iteration

    param = 'ObsFcstAnaEns'
    # param = 'ObsFcstAna'

    io_ol = LDAS_io(param, exp_ol)
    io_da = LDAS_io(param, exp_da)

    res = pd.DataFrame(index=io_ol.grid.tilecoord.index.values,
                       columns=['col', 'row'] + ['obs_var_spc%i'%spc for spc in [1,2,3,4]] \
                                              + ['fcst_var_spc%i'%spc for spc in [1,2,3,4]] \
                                              + ['ana_var_spc%i'%spc for spc in [1,2,3,4]])

    ids = io_ol.grid.tilecoord['tile_id'].values
    res.loc[:, 'col'], res.loc[:, 'row'] = np.vectorize(io_ol.grid.tileid2colrow)(ids, local_cs=False)

    for cnt, (idx, val) in enumerate(io_ol.grid.tilecoord.iterrows()):
        print('%i / %i' % (cnt, len(res)))

        col, row = io_ol.grid.tileid2colrow(val.tile_id)

        for spc in [1, 2, 3, 4]:
            ts_fcst = io_ol.read_ts('obs_fcst', col, row, species=spc, lonlat=False).dropna()
            ts_obs = io_da.read_ts('obs_obs', col, row, species=spc, lonlat=False).dropna()
            ts_ana = io_da.read_ts('obs_ana', col, row, species=spc, lonlat=False).dropna()
            if len(ts_ana) == 0:
                continue
            ts_fcst = ts_fcst.loc[ts_ana.index]
            ts_obs = ts_obs.loc[ts_ana.index]

            res.loc[idx,'obs_var_spc%i'%spc] = ts_obs.var(axis='columns').mean()
            res.loc[idx,'fcst_var_spc%i'%spc] = ts_fcst.var(axis='columns').mean()
            res.loc[idx,'ana_var_spc%i'%spc] = ts_ana.var(axis='columns').mean()

            # res.loc[idx,'obs_var_spc%i'%spc] = np.nanmean(io_da.timeseries['obs_obsvar'][:,spc-1,row,col].values)
            # res.loc[idx,'fcst_var_spc%i'%spc] = np.nanmean(io_ol.timeseries['obs_fcstvar'][:,spc-1,row,col].values)
            # res.loc[idx,'ana_var_spc%i'%spc] = np.nanmean(io_da.timeseries['obs_anavar'][:,spc-1,row,col].values)

    fname = root / 'result_files' / 'ens_var.csv'

    res.to_csv(fname, float_format='%0.8f')


def calc_ens_cov(root, iteration):

    # exp_ol = 'US_M36_SMOS40_TB_ens_test_OL'
    # exp_da = 'US_M36_SMOS40_TB_MadKF_it%i' % iteration
    exp_ol = 'US_M36_SMOS40_TB_MadKF_OL_it%i' % iteration
    exp_da = 'US_M36_SMOS40_TB_MadKF_DA_it%i' % iteration

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
            ts_ol = ts_ol.loc[ts_ana.index,:]
            ts_obs = ts_obs.loc[ts_ana.index,:]

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

            res.loc[idx,'c_ol_obs_spc%i'%spc] = np.nanmean(tmp_c_ol_obs)
            res.loc[idx,'c_ol_ana_spc%i'%spc] = np.nanmean(tmp_c_ol_ana)
            res.loc[idx,'c_obs_ana_spc%i'%spc] = np.nanmean(tmp_c_obs_ana)

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

    img = np.empty(lons.shape, dtype='float32')
    img.fill(None)

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

    img[ind_lat,ind_lon] = data[tag] if sqrt is False else np.sqrt(data[tag])
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

    x, y = m(-79, 25)
    plt.text(x, y, 'mean = %.3f' % np.ma.mean(img_masked), fontsize=fontsize - 2)

def plot_ens_cov(root,iteration):

    fname = root / 'result_files' / 'ens_cov.csv'
    res = pd.read_csv(fname, index_col=0).dropna()

    for corrected in [False, True]:

        if corrected:
            root = Path('/work/MadKF/CLSM/iter_%i'%(iteration-1))
            fname = root / 'result_files' / 'mse.csv'
            res0 = pd.read_csv(fname, index_col=0).dropna()

            root = Path('/work/MadKF/CLSM/iter_%i'%iteration)
            fname = root / 'result_files' / 'ens_var.csv'
            res1 = pd.read_csv(fname, index_col=0).dropna()

        plt.figure(figsize=(24,11))

        cmap='jet'

        cbrange1 = [-1, 1]
        cbrange2 = [0, 10]

        for spc in [1,2,3,4]:

            if corrected:
                scl_obs =  res0['mse_obs_spc%i' % spc] / res1['obs_var_spc%i' % spc]
                scl_fcst =  res0['mse_fcst_spc%i' % spc] / res1['fcst_var_spc%i' % spc]
                scl_ana =  res0['mse_ana_spc%i' % spc] / res1['ana_var_spc%i' % spc]

                res['c_ol_obs_spc%i' % spc] *= np.sqrt(scl_fcst*scl_obs)
                res['c_ol_ana_spc%i' % spc] *= np.sqrt(scl_fcst*scl_ana)
                res['c_obs_ana_spc%i' % spc] *= np.sqrt(scl_obs*scl_ana)

            plt.subplot(3,4,spc)
            plot_ease_img(res, 'c_ol_obs_spc%i'%spc, cbrange=cbrange1, title='c_ol_obs_spc%i'%spc, cmap=cmap)

            plt.subplot(3,4,spc+4)
            plot_ease_img(res, 'c_ol_ana_spc%i'%spc, cbrange=cbrange1, title='c_ol_ana_spc%i'%spc, cmap=cmap)

            plt.subplot(3,4,spc+8)
            plot_ease_img(res, 'c_obs_ana_spc%i'%spc, cbrange=cbrange2, title='c_obs_ana_spc%i'%spc, cmap=cmap)

        if corrected:
            fname = root / 'plots' / 'ens_cov_corrected.png'
        else:
            fname = root / 'plots' / 'ens_cov_uncorrected.png'

        plt.tight_layout()
        # plt.show()
        plt.savefig(fname, dpi=plt.gcf().dpi)
        plt.close()


def plot_mse(root, iteration):

    fname = root / 'result_files' / 'mse.csv'
    res = pd.read_csv(fname, index_col=0)

    for corrected in [False, True]:

        if corrected:
            fname = root / 'result_files' / 'ens_cov.csv'
            cov = pd.read_csv(fname, index_col=0)

            root = Path('/work/MadKF/CLSM/iter_%i'%(iteration-1))
            fname = root / 'result_files' / 'mse.csv'
            res0 = pd.read_csv(fname, index_col=0)

            root = Path('/work/MadKF/CLSM/iter_%i'%iteration)
            fname = root / 'result_files' / 'ens_var.csv'
            res1 = pd.read_csv(fname, index_col=0)

            for spc in [1,2,3,4]:
                scl_obs =  res0['mse_obs_spc%i' % spc] / res1['obs_var_spc%i' % spc]
                scl_fcst =  res0['mse_fcst_spc%i' % spc] / res1['fcst_var_spc%i' % spc]
                scl_ana =  res0['mse_ana_spc%i' % spc] / res1['ana_var_spc%i' % spc]

                cov['c_ol_obs_spc%i' % spc] *= np.sqrt(scl_fcst*scl_obs)
                cov['c_ol_ana_spc%i' % spc] *= np.sqrt(scl_fcst*scl_ana)
                cov['c_obs_ana_spc%i' % spc] *= np.sqrt(scl_obs*scl_ana)

                res['mse_obs_spc%i' % spc] = res['mse_obs_spc%i' % spc] - cov['c_ol_ana_spc%i' % spc] + cov['c_obs_ana_spc%i' % spc]
                res['mse_fcst_spc%i' % spc] = res['mse_fcst_spc%i' % spc] + cov['c_ol_ana_spc%i' % spc] - cov['c_obs_ana_spc%i' % spc]

                fname = root / 'result_files' / 'mse_corrected.csv'
                res.to_csv(fname)

        plt.figure(figsize=(24,11))

        cmap='jet'

        cbrange = [0, 100]

        for spc in [1, 2, 3, 4]:
            plt.subplot(3,4,spc)
            plot_ease_img(res, 'mse_obs_spc%i'%spc, cbrange=cbrange, title='ubMSE obs (spc%i)'%spc, cmap=cmap)
            plt.subplot(3,4,spc+4)
            plot_ease_img(res, 'mse_fcst_spc%i'%spc, cbrange=cbrange, title='ubMSE fcst (spc%i)'%spc, cmap=cmap)
            plt.subplot(3,4,spc+8)
            plot_ease_img(res, 'mse_ana_spc%i'%spc, cbrange=cbrange, title='ubMSE ana (spc%i)'%spc, cmap=cmap)

        plt.tight_layout()

        if corrected:
            fname = root / 'plots' / 'mse_corrected.png'
        else:
            fname = root / 'plots' / 'mse_uncorrected.png'

        plt.tight_layout()

        # plt.show()
        plt.savefig(fname, dpi=plt.gcf().dpi)
        plt.close()

def plot_ens_var(root):

    fname = root / 'result_files' / 'ens_var.csv'
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

    plt.tight_layout()

    # plt.show()
    plt.savefig(root / 'plots' / 'ens_var.png', dpi=plt.gcf().dpi)
    plt.close()


def plot_obs_pert(root):

    fname = root / 'result_files' / 'ens_var.csv'
    ensvar = pd.read_csv(fname, index_col=0)

    fname = root / 'result_files' / 'mse.csv'
    mse = pd.read_csv(fname, index_col=0)

    res = ensvar[['col','row']]

    for spc in np.arange(1,5):
        res['obs_var_spc%i'%spc] = ensvar['fcst_var_spc%i'%spc] * mse['mse_obs_spc%i'%spc] / mse['mse_fcst_spc%i'%spc]

        res.loc[(res['obs_var_spc%i' % spc] < 1), 'obs_var_spc%i' % spc] = 1
        res.loc[(res['obs_var_spc%i' % spc] > 15), 'obs_var_spc%i' % spc] = 15

        res.loc[np.isnan(res['obs_var_spc%i'%spc]),'obs_var_spc%i'%spc] = res['obs_var_spc%i'%spc].median()

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

    plt.tight_layout()

    plt.show()
    # plt.savefig(root / 'plots' / 'obs_pert.png', dpi=plt.gcf().dpi)
    # plt.close()


def write_spatial_errors(root, iteration):

    froot = root / 'error_files'
    fbase = 'SMOS_fit_Tb_'

    # exp = 'US_M36_SMOS40_TB_MadKF_it%i' % iteration
    exp = 'US_M36_SMOS40_TB_MadKF_DA_it%i' % iteration
    io = LDAS_io('ObsFcstAna', exp)

    fname = root / 'result_files' / 'ens_var.csv'
    ensvar = pd.read_csv(fname, index_col=0)

    fname = root / 'result_files' / 'mse_corrected.csv'
    mse = pd.read_csv(fname, index_col=0)

    obs_err = ensvar[['col','row']]
    obs_err.loc[:, 'tile_id'] = io.grid.tilecoord.loc[obs_err.index, 'tile_id'].values

    for spc in np.arange(1,5):
        obs_err.loc[:,'obs_var_spc%i'%spc] = ensvar['fcst_var_spc%i'%spc] * mse['mse_obs_spc%i'%spc] / mse['mse_fcst_spc%i'%spc]
        obs_err.loc[(obs_err['obs_var_spc%i' % spc] < 1), 'obs_var_spc%i' % spc] = 1
        obs_err.loc[(obs_err['obs_var_spc%i' % spc] > 100), 'obs_var_spc%i' % spc] = 100
        obs_err.loc[np.isnan(obs_err['obs_var_spc%i'%spc]),'obs_var_spc%i'%spc] = obs_err['obs_var_spc%i'%spc].median()

    dtype = template_error_Tb40()[0]

    angles = np.array([40.,])
    orbits = ['A', 'D']

    template = pd.DataFrame(columns=dtype.names).astype('float32')
    template['lon'] = io.grid.tilecoord['com_lon'].values.astype('float32')
    template['lat'] = io.grid.tilecoord['com_lat'].values.astype('float32')
    template.index += 1

    modes = np.array([0, 0])
    sdate = np.array([2010, 1, 1, 0, 0])
    edate = np.array([2014, 1, 1, 0, 0])
    lengths = np.array([len(template), len(angles)])  # tiles, incidence angles, whatever

    # ----- write output files -----
    for orb in orbits:
        # !!! inconsistent with the definition in the obs_paramfile (species) !!!
        modes[0] = 1 if orb == 'A' else 0

        res = template.copy()

        spc = 0 if orb == 'A' else 1
        res.loc[:,'err_Tbh'] = np.sqrt(obs_err.loc[res.index,'obs_var_spc%i'%(spc+1)]).values

        spc = 2 if orb == 'A' else 3
        res.loc[:,'err_Tbv'] = np.sqrt(obs_err.loc[res.index,'obs_var_spc%i'%(spc+1)]).values

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

def plot_P_R_check(iteration):

    root = Path('/work/MadKF/CLSM/iter_%i'%(iteration-1))
    fname = root / 'result_files' / 'mse.csv'
    res0 = pd.read_csv(fname, index_col=0).dropna()

    root = Path('/work/MadKF/CLSM/iter_%i'%iteration)
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


def plot_P_R_scl(iteration):

    root = Path('/work/MadKF/CLSM/iter_%i'%(iteration-1))
    fname = root / 'result_files' / 'mse.csv'
    res0 = pd.read_csv(fname, index_col=0).dropna()

    root = Path('/work/MadKF/CLSM/iter_%i'%iteration)
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



def plot_perturbations(iteration):

    fA = '/work/MadKF/CLSM/iter_%i/error_files/SMOS_fit_Tb_A.bin' % iteration
    fD = '/work/MadKF/CLSM/iter_%i/error_files/SMOS_fit_Tb_D.bin' % iteration

    dtype, hdr, length = template_error_Tb40()

    io = LDAS_io()

    imgA = io.read_fortran_binary(fA, dtype, hdr=hdr, length=length)
    imgD = io.read_fortran_binary(fD, dtype, hdr=hdr, length=length)

    imgA.index += 1
    imgD.index += 1

    cbrange = [0,10]

    plt.figure(figsize=(20, 10))

    plt.subplot(221)
    plot_ease_img2(imgA,'err_Tbh', cbrange=cbrange, title='H-pol (Asc.)')
    plt.subplot(222)
    plot_ease_img2(imgA,'err_Tbv', cbrange=cbrange, title='V-pol (Asc.)')
    plt.subplot(223)
    plot_ease_img2(imgD,'err_Tbh', cbrange=cbrange, title='H-pol (Dsc.)')
    plt.subplot(224)
    plot_ease_img2(imgD,'err_Tbh', cbrange=cbrange, title='V-pol (Dsc.)')

    plt.tight_layout()
    # plt.show()

    plt.savefig(root / 'plots' / 'perturbations.png', dpi=plt.gcf().dpi)
    plt.close()

if __name__=='__main__':

    iteration = 4

    root = Path('/work/MadKF/CLSM/iter_%i'%iteration)

    if not (root / 'result_files').exists():
        Path.mkdir(root / 'result_files', parents=True)
    if not (root / 'plots').exists():
        Path.mkdir(root / 'plots')
    if not (root / 'error_files').exists():
        Path.mkdir(root / 'error_files')

    # calc_tb_mse(root, iteration)
    # calc_ens_var(root, iteration)
    # calc_ens_cov(root, iteration)

    plot_mse(root, iteration)
    plot_ens_cov(root, iteration)
    plot_ens_var(root)

    plot_P_R_check(iteration)
    plot_P_R_scl(iteration)

    write_spatial_errors(root, iteration)

    plot_perturbations(iteration)