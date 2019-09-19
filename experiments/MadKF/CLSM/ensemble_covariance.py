
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from pyldas.interface import LDAS_io

from pyldas.templates import template_error_Tb40

def tca(df):
    cov = df.dropna().cov().values
    ind = (0, 1, 2, 0, 1, 2)
    err_var = np.array([np.abs(cov[i, i] - (cov[i, ind[i + 1]] * cov[i, ind[i + 2]]) / cov[ind[i + 1], ind[i + 2]])
                        for i in np.arange(3)])
    return err_var


def calc_tb_rmsd():

    exp_ol = 'US_M36_SMOS40_TB_ens_test_OL'
    param = 'ObsFcstAna'
    io_ol = LDAS_io(param, exp_ol)

    res = pd.DataFrame(index=io_ol.grid.tilecoord.index.values,
                       columns=['col', 'row', 'rmsd_spc1', 'rmsd_spc2', 'rmsd_spc3', 'rmsd_spc4'])

    ids = io_ol.grid.tilecoord['tile_id'].values
    res.loc[:, 'col'], res.loc[:, 'row'] = np.vectorize(io_ol.grid.tileid2colrow)(ids, local_cs=False)

    for cnt, (idx, val) in enumerate(io_ol.grid.tilecoord.iterrows()):
        print('%i / %i' % (cnt, len(res)))

        col, row = io_ol.grid.tileid2colrow(val.tile_id)

        for spc in [1, 2, 3, 4]:
            ts_fcst = io_ol.read_ts('obs_fcst', col, row, species=spc, lonlat=False).dropna()
            ts_obs = io_ol.read_ts('obs_obs', col, row, species=spc, lonlat=False).dropna()

            res.loc[idx,'rmsd_spc%i'%spc] = np.mean(((ts_fcst - ts_fcst.mean()) - (ts_obs - ts_obs.mean()))**2)**0.5

    fname = '/work/MadKF/CLSM/rmsd.csv'

    res.to_csv(fname, float_format='%0.8f')


def calc_tb_rmse():

    exp_ol = 'US_M36_SMOS40_TB_ens_test_OL'
    exp_da = 'US_M36_SMOS40_TB_ens_test_DA'

    param = 'ObsFcstAna'

    io_ol = LDAS_io(param, exp_ol)
    io_da = LDAS_io(param, exp_da)

    res = pd.DataFrame(index=io_ol.grid.tilecoord.index.values,
                       columns=['col', 'row'] + ['rmse_obs_spc%i'%spc for spc in [1,2,3,4]] \
                                              + ['rmse_fcst_spc%i'%spc for spc in [1,2,3,4]] \
                                              + ['rmse_ana_spc%i'%spc for spc in [1,2,3,4]])

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

            res.loc[idx,'rmse_obs_spc%i'%spc] = tc_res[0]
            res.loc[idx,'rmse_fcst_spc%i'%spc] = tc_res[1]
            res.loc[idx,'rmse_ana_spc%i'%spc] = tc_res[2]

    fname = '/work/MadKF/CLSM/rmse.csv'

    res.to_csv(fname, float_format='%0.8f')

def calc_ens_var():

    exp_ol = 'US_M36_SMOS40_TB_ens_test_OL'
    exp_da = 'US_M36_SMOS40_TB_ens_test_DA'

    param = 'ObsFcstAnaEns'

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

    fname = '/work/MadKF/CLSM/ens_var.csv'

    res.to_csv(fname, float_format='%0.8f')


def calc_ens_cov():

    exp_ol = 'US_M36_SMOS40_TB_ens_test_OL'
    exp_da = 'US_M36_SMOS40_TB_ens_test_DA'

    param = 'ObsFcstAnaEns'

    io_ol = LDAS_io(param, exp_ol)
    io_da = LDAS_io(param, exp_da)

    for spc in [2,3,4]:

        res = pd.DataFrame(index=io_ol.grid.tilecoord.index.values, columns=['col','row','c_ol_obs','c_ol_ana','c_obs_ana'])

        for cnt, (idx,val) in enumerate(io_ol.grid.tilecoord.iterrows()):
            print('%i / %i' % (cnt, len(res)))

            res.loc[:, 'col'], res.loc[:, 'row'] = io_ol.grid.tileid2colrow(val.tile_id, local_cs=False)

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

            res.loc[idx,'c_ol_obs'] = np.nanmean(tmp_c_ol_obs)
            res.loc[idx,'c_ol_ana'] = np.nanmean(tmp_c_ol_ana)
            res.loc[idx,'c_obs_ana'] = np.nanmean(tmp_c_obs_ana)

        fname = '/work/MadKF/CLSM/test_spc%i.csv' % spc

        res.to_csv(fname, float_format='%0.8f')


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

def plot_ens_cov():

    fname = '/work/MadKF/CLSM/test_spc1.csv'

    res = pd.read_csv(fname, index_col=0).dropna()

    plt.figure(figsize=(8,12))

    cmap='jet'

    cbrange1 = [-0.15, 0.15]
    cbrange2 = [-1.2, 1.2]

    plt.subplot(311)
    plot_ease_img(res, 'c_ol_obs', cbrange=cbrange1, title='c_ol_obs', cmap=cmap)

    plt.subplot(312)
    plot_ease_img(res, 'c_ol_ana', cbrange=cbrange1, title='c_ol_ana', cmap=cmap)

    plt.subplot(313)
    plot_ease_img(res, 'c_obs_ana', cbrange=cbrange2, title='c_obs_ana', cmap=cmap)

    plt.tight_layout()
    plt.show()


def plot_rmsd():

    fname = '/work/MadKF/CLSM/rmsd.csv'

    res = pd.read_csv(fname, index_col=0).dropna()

    plt.figure(figsize=(14,8))

    cmap='jet'

    cbrange = [0, 20]

    plt.subplot(221)
    plot_ease_img(res, 'rmsd_spc1', cbrange=cbrange, title='species 1', cmap=cmap)

    plt.subplot(222)
    plot_ease_img(res, 'rmsd_spc2', cbrange=cbrange, title='species 2', cmap=cmap)

    plt.subplot(223)
    plot_ease_img(res, 'rmsd_spc3', cbrange=cbrange, title='species 3', cmap=cmap)

    plt.subplot(224)
    plot_ease_img(res, 'rmsd_spc4', cbrange=cbrange, title='species 4', cmap=cmap)

    plt.tight_layout()
    plt.show()


def plot_rmse():

    fname = '/work/MadKF/CLSM/rmse.csv'
    res = pd.read_csv(fname, index_col=0).dropna()

    # fname = '/work/MadKF/CLSM/test_spc1.csv'
    # covs = pd.read_csv(fname, index_col=0).dropna()
    # res['rmse_obs_spc1'] = np.sqrt( res['rmse_obs_spc1']**2 - covs['c_ol_ana'] + covs['c_obs_ana'] )
    # res['rmse_fcst_spc1'] = np.sqrt( res['rmse_fcst_spc1']**2 + covs['c_ol_ana'] - covs['c_obs_ana'] )

    plt.figure(figsize=(8,12))

    cmap='jet'

    cbrange = [0, 15]

    spc = 1

    plt.subplot(311)
    plot_ease_img(res, 'rmse_obs_spc%i'%spc, cbrange=cbrange, title='ubRMSE obs', cmap=cmap)

    plt.subplot(312)
    plot_ease_img(res, 'rmse_fcst_spc%i'%spc, cbrange=cbrange, title='ubRMSE fcst', cmap=cmap)

    plt.subplot(313)
    plot_ease_img(res, 'rmse_ana_spc%i'%spc, cbrange=cbrange, title='ubRMSE ana', cmap=cmap)

    plt.tight_layout()
    plt.show()

def plot_ens_var():

    fname = '/work/MadKF/CLSM/ens_var.csv'
    res = pd.read_csv(fname, index_col=0).dropna()

    plt.figure(figsize=(8,12))

    cmap='jet'

    spc = 1

    plt.subplot(311)
    plot_ease_img(res, 'obs_var_spc%i'%spc, cbrange=[0, 5], title='Observation Ens. Variance', cmap=cmap)

    plt.subplot(312)
    plot_ease_img(res, 'fcst_var_spc%i'%spc, cbrange=[0, 25], title='Forecast Ens. Variance', cmap=cmap)

    plt.subplot(313)
    plot_ease_img(res, 'ana_var_spc%i'%spc, cbrange=[0, 5], title='Analysis Ens. Variance', cmap=cmap)

    plt.tight_layout()
    plt.show()


def plot_obs_pert():

    fname = '/work/MadKF/CLSM/ens_var.csv'
    ensvar = pd.read_csv(fname, index_col=0)

    fname = '/work/MadKF/CLSM/mse.csv'
    mse = pd.read_csv(fname, index_col=0)

    res = ensvar[['col','row']]

    for spc in np.arange(1,5):
        res['obs_var_spc%i'%spc] = ensvar['fcst_var_spc%i'%spc] * mse['rmse_obs_spc%i'%spc] / mse['rmse_fcst_spc%i'%spc]

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


def write_spatial_errors():

    froot = '/work/MadKF/CLSM/error_files'
    fbase = 'SMOS_fit_Tb_'

    exp = 'US_M36_SMOS40_TB_ens_test_DA'
    io = LDAS_io('ObsFcstAna', exp)

    fname = '/work/MadKF/CLSM/result_files/ens_var.csv'
    ensvar = pd.read_csv(fname, index_col=0)

    fname = '/work/MadKF/CLSM/result_files/mse.csv'
    mse = pd.read_csv(fname, index_col=0)

    obs_err = ensvar[['col','row']]
    obs_err.loc[:, 'tile_id'] = io.grid.tilecoord.loc[obs_err.index, 'tile_id'].values

    for spc in np.arange(1,5):
        obs_err.loc[:,'obs_var_spc%i'%spc] = ensvar['fcst_var_spc%i'%spc] * mse['rmse_obs_spc%i'%spc] / mse['rmse_fcst_spc%i'%spc]
        obs_err.loc[(obs_err['obs_var_spc%i' % spc] < 1), 'obs_var_spc%i' % spc] = 1
        obs_err.loc[(obs_err['obs_var_spc%i' % spc] > 15), 'obs_var_spc%i' % spc] = 15
        obs_err.loc[np.isnan(obs_err['obs_var_spc%i'%spc]),'obs_var_spc%i'%spc] = obs_err['obs_var_spc%i'%spc].median()

    dtype = template_error_Tb40()[0]

    angles = np.array([40.,])
    orbits = ['A', 'D']

    template = pd.DataFrame(columns=dtype.names).astype('float32')
    template['lon'] = io.grid.tilecoord['com_lon'].values.astype('float32')
    template['lat'] = io.grid.tilecoord['com_lat'].values.astype('float32')

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

        fname = os.path.join(froot, fbase + orb + '.bin')

        fid = open(fname, 'wb')
        io.write_fortran_block(fid, modes)
        io.write_fortran_block(fid, sdate)
        io.write_fortran_block(fid, edate)
        io.write_fortran_block(fid, lengths)
        io.write_fortran_block(fid, angles)

        for f in res.columns.values:
            io.write_fortran_block(fid, res[f].values)
        fid.close()

if __name__=='__main__':
    # calc_tb_rmsd()
    # calc_tb_rmse()
    # calc_ens_cov()
    # calc_ens_var()

    # plot_ens_cov()
    # plot_rmsd()
    # plot_rmse()
    # plot_ens_var()

    write_spatial_errors()
