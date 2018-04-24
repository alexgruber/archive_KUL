

import os

import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm

from pyldas.grids import EASE2

from pyldas.interface import LDAS_io

from myprojects.timeseries import calc_anomaly


def plot_increments():

    cal = LDAS_io('incr','US_M36_SMOS_DA_calibrated_scaled')
    uncal = LDAS_io('incr','US_M36_SMOS_DA_nocal_scaled_pentadal')

    incr_var_cal = (cal.timeseries['srfexc'] + cal.timeseries['rzexc'] - cal.timeseries['catdef']).var(dim='time').values
    incr_var_uncal = (uncal.timeseries['srfexc'] + uncal.timeseries['rzexc'] - uncal.timeseries['catdef']).var(dim='time').values

    print np.nanmean(incr_var_cal), np.nanmean(incr_var_uncal)

    tc = LDAS_io().tilecoord
    tg = LDAS_io().tilegrids

    tc.i_indg -= tg.loc['domain', 'i_offg']  # col / lon
    tc.j_indg -= tg.loc['domain', 'j_offg']  # row / lat

    lons = np.unique(tc.com_lon.values)
    lats = np.unique(tc.com_lat.values)[::-1]

    lons, lats = np.meshgrid(lons, lats)

    llcrnrlat = 24
    urcrnrlat = 51
    llcrnrlon = -128
    urcrnrlon = -64

    figsize = (10, 10)
    cmap = 'jet'
    fontsize = 18
    cbrange = (0, 25)

    plt.figure(figsize=figsize)

    plt.subplot(211)
    plt_img = np.ma.masked_invalid(incr_var_cal)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon, resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('Increment variance (calibrated RTM)', fontsize=fontsize)

    plt.subplot(212)
    plt_img = np.ma.masked_invalid(incr_var_uncal)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('Increment variance (uncalibrated RTM)', fontsize=fontsize)


    plt.tight_layout()
    plt.show()

def increment_density_plot():

    cal = LDAS_io('incr','US_M36_SMOS_DA_calibrated_scaled')
    uncal = LDAS_io('incr','US_M36_SMOS_DA_nocal_scaled_pentadal')

    incr_var_cal = (cal.timeseries['srfexc'] + cal.timeseries['rzexc'] - cal.timeseries['catdef']).values.flatten()
    incr_var_cal = incr_var_cal[~np.isnan(incr_var_cal)]

    incr_var_uncal = (uncal.timeseries['srfexc'] + uncal.timeseries['rzexc'] - uncal.timeseries['catdef']).values.flatten()
    incr_var_uncal = incr_var_uncal[~np.isnan(incr_var_uncal)]

    ind = (incr_var_cal != 0) & (incr_var_uncal != 0)
    incr_var_cal = incr_var_cal[ind]
    incr_var_uncal = incr_var_uncal[ind]

    figsize = (10, 8)
    fontsize = 16
    plt.figure(figsize=figsize)
    plt.hist2d(incr_var_cal, incr_var_uncal, bins=[100, 100], range=[[-60, 60], [-60, 60]], cmap='jet', norm=LogNorm(), normed=True)
    plt.xlabel('Increment variance (calibrated RTM)',fontsize=fontsize)
    plt.ylabel('Increment variance (uncalibrated RTM)',fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=fontsize-2)

    plt.tight_layout()
    plt.show()


def plot_innovation_variance():

    ds = xr.open_dataset(r"D:\work\LDAS\2018-02_scaling\_new\diagnostics\filter_diagnostics.nc")

    innov_var_cal = ds['innov_var'][:, :, 1, :].mean(dim='species').values
    innov_var_uncal = ds['innov_var'][:, :, 3, :].mean(dim='species').values

    print np.nanmean(innov_var_cal), np.nanmean(innov_var_uncal)

    lons = ds.lon.values
    lats = ds.lat.values

    lons, lats = np.meshgrid(lons, lats)

    llcrnrlat = 24
    urcrnrlat = 51
    llcrnrlon = -128
    urcrnrlon = -64

    figsize = (10, 10)
    cmap = 'jet'
    fontsize = 18
    cbrange = (0, 120)

    plt.figure(figsize=figsize)

    plt.subplot(211)
    plt_img = np.ma.masked_invalid(innov_var_cal)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon, resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('Innovation variance (calibrated RTM)', fontsize=fontsize)

    plt.subplot(212)
    plt_img = np.ma.masked_invalid(innov_var_uncal)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('Innovation variance (uncalibrated RTM)', fontsize=fontsize)


    plt.tight_layout()
    plt.show()


def plot_normalized_innovation_variance():

    ds = xr.open_dataset(r"D:\work\LDAS\2018-02_scaling\_new\diagnostics\filter_diagnostics.nc")

    innov_var_cal = ds['norm_innov_var'][:, :, 1, :].mean(dim='species').values
    innov_var_uncal = ds['norm_innov_var'][:, :, 3, :].mean(dim='species').values

    print np.nanmean(innov_var_cal), np.nanmean(innov_var_uncal)

    lons = ds.lon.values
    lats = ds.lat.values

    lons, lats = np.meshgrid(lons, lats)

    llcrnrlat = 24
    urcrnrlat = 51
    llcrnrlon = -128
    urcrnrlon = -64

    figsize = (10, 10)
    cmap = 'jet'
    fontsize = 18
    cbrange = (0, 3)

    plt.figure(figsize=figsize)

    plt.subplot(211)
    plt_img = np.ma.masked_invalid(innov_var_cal)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon, resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('Normalized innovation variance (calibrated RTM)', fontsize=fontsize)

    plt.subplot(212)
    plt_img = np.ma.masked_invalid(innov_var_uncal)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, plt_img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('Normalized innovation variance (uncalibrated RTM)', fontsize=fontsize)


    plt.tight_layout()
    plt.show()


def plot_ismn_statistics():

    fname = r"D:\work\LDAS\2018-02_scaling\ismn_eval\no_da_cal_uncal_ma_harm\validation_masked.csv"

    res = pd.read_csv(fname, index_col=0)

    variables = ['sm_surface','sm_rootzone','sm_profile']
    runs = ['OL', 'DA_ref', 'DA_pent']
    legend = ['open loop', 'calibrated', 'uncalibrated']

    # networks = ['SCAN','USCRN']
    # res.index = res.network
    # res = res.loc[networks,:]

    r_title = 'Pearson R '
    ubrmsd_title = 'ubRMSD '

    plt.figure(figsize=(7, 8))

    offsets = [-0.2, 0, 0.2]
    cols = ['lightblue', 'lightgreen', 'coral']
    # offsets = [-0.3,-0.1,0.1,0.3]
    # cols = ['lightblue', 'lightgreen', 'coral', 'brown']
    fontsize = 12

    ax = plt.subplot(211)
    plt.grid(color='k', linestyle='--', linewidth=0.25)

    data = list()
    ticks = list()
    pos = list()
    colors = list()

    for i, var in enumerate(variables):
        ticks.append(var)
        for col, offs, run in zip(cols, offsets, runs):
            tmp_data = res['corr_' + run + '_ma_' + var].values
            tmp_data = tmp_data[~np.isnan(tmp_data)]
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
    plt.xticks(np.arange(len(variables)) + 1, ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0.5, len(ticks) + 0.5)
    plt.ylim(0.0, 1.0)
    for i in np.arange(len(variables)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    ax.set_title(r_title, fontsize=fontsize)

    # ---------------------------------------------------------------------------------------------------------
    ax = plt.subplot(212)
    plt.grid(color='k', linestyle='--', linewidth=0.25)

    data = list()
    ticks = list()
    pos = list()
    colors = list()

    for i, var in enumerate(variables):
        ticks.append(var)
        for col, offs, run in zip(cols, offsets, runs):
            tmp_data = res['ubrmsd_' + run + '_ma_' + var].values
            tmp_data = tmp_data[~np.isnan(tmp_data)]
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
    # plt.figlegend((box['boxes'][0:4]),runs,'upper left',fontsize=fontsize)
    plt.xticks(np.arange(len(variables)) + 1, ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(0.5, len(ticks) + 0.5)
    plt.ylim(0, 0.1)
    for i in np.arange(len(variables)):
        plt.axvline(i + 0.5, linewidth=1, color='k')
    ax.set_title(ubrmsd_title, fontsize=fontsize)

    plt.show()

def plot_timeseries():

    # Colorado
    # lat = 39.095962936
    # lon = -106.918945312

    # Nebraska
    # lat = 41.203456192
    # lon = -102.249755859

    # New Mexico
    # lat = 31.522361470
    # lon = -108.528442383

    # Oklahoma
    lat = 35.205233348
    lon = -97.910156250

    cal = LDAS_io('incr','US_M36_SMOS_DA_calibrated_scaled')
    uncal = LDAS_io('incr','US_M36_SMOS_DA_nocal_scaled_pentadal')

    incr_var_cal = (cal.timeseries['srfexc'] + cal.timeseries['rzexc'] - cal.timeseries['catdef']).var(dim='time').values
    incr_var_uncal = (uncal.timeseries['srfexc'] + uncal.timeseries['rzexc'] - uncal.timeseries['catdef']).var(dim='time').values

    col, row = LDAS_io().grid.lonlat2colrow(lon, lat, domain=True)

    title = 'increment variance (calibrated): %.2f        increment variance (uncalibrated): %.2f' % (incr_var_cal[row,col], incr_var_uncal[row,col])
    # title = ''

    fontsize = 12

    cal = LDAS_io('ObsFcstAna', 'US_M36_SMOS_DA_calibrated_scaled')
    uncal = LDAS_io('ObsFcstAna', 'US_M36_SMOS_DA_nocal_scaled_pentadal')
    orig = LDAS_io('ObsFcstAna', 'US_M36_SMOS_noDA_unscaled')

    ts_obs_cal = cal.read_ts('obs_obs', lon, lat, species=3, lonlat=True)
    ts_obs_cal.name = 'Tb obs (calibrated)'
    ts_obs_uncal = uncal.read_ts('obs_obs', lon, lat, species=3, lonlat=True)
    ts_obs_uncal.name = 'Tb obs (uncalibrated)'

    ts_obs_orig = orig.read_ts('obs_obs', lon, lat, species=3, lonlat=True)
    ts_obs_orig.name = 'Tb obs (uncalibrated, unscaled)'

    ts_fcst_cal = cal.read_ts('obs_fcst', lon, lat, species=3, lonlat=True)
    ts_fcst_cal.name = 'Tb fcst (calibrated)'
    ts_fcst_uncal = uncal.read_ts('obs_fcst', lon, lat, species=3, lonlat=True)
    ts_fcst_uncal.name = 'Tb fcst (uncalibrated)'

    df = pd.concat((ts_obs_cal, ts_obs_uncal, ts_obs_orig, ts_fcst_cal, ts_fcst_uncal),axis=1).dropna()

    plt.figure(figsize=(19,8))

    ax1 = plt.subplot(211)
    df.plot(ax=ax1, ylim=[140,300], xlim=['2010-01-01','2017-01-01'], fontsize=fontsize, style=['-','--',':','-','--'], linewidth=2)
    plt.xlabel('')
    plt.title(title, fontsize=fontsize+2)

    cols = df.columns.values
    for i,col in enumerate(df):
        df[col] = calc_anomaly(df[col], method='ma', longterm=True).values
        if i < 3:
            cols[i] = col[0:7] + 'anomaly ' + col[7::]
        else:
            cols[i] = col[0:7] + ' anomaly' + col[7::]
    df.columns = cols
    df.dropna(inplace=True)

    ax2 = plt.subplot(212, sharex=ax1)
    df.plot(ax=ax2, ylim=[-60,60], xlim=['2010-01-01','2017-01-01'], fontsize=fontsize, style=['-','--',':','-','--'], linewidth=2)
    plt.xlabel('')

    plt.tight_layout()
    plt.show()

def plot_ismn_locations():

    fname = r"D:\work\LDAS\2018-02_scaling\ismn_eval\no_da_cal_uncal_ma_harm\validation_masked.csv"

    res = pd.read_csv(fname, index_col=0)

    lats = res['lat'].values
    lons = res['lon'].values

    llcrnrlat = 24
    urcrnrlat = 51
    llcrnrlon = -128
    urcrnrlon = -64

    figsize = (10, 5)

    plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')

    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    x, y = m(lons, lats)
    plt.plot(x, y, 'o', markersize=7, markeredgecolor='k', markerfacecolor='coral')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    plt.title('ISMN station locations', fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_model_parameters():

    cal = LDAS_io(exp='US_M36_SMOS_DA_calibrated_scaled').read_params('RTMparam')
    uncal = LDAS_io(exp='US_M36_SMOS_DA_nocal_scaled_pentadal').read_params('RTMparam')

    tc = LDAS_io().tilecoord
    tg = LDAS_io().tilegrids

    tc.i_indg -= tg.loc['domain', 'i_offg']  # col / lon
    tc.j_indg -= tg.loc['domain', 'j_offg']  # row / lat

    lons = np.unique(tc.com_lon.values)
    lats = np.unique(tc.com_lat.values)[::-1]

    lons, lats = np.meshgrid(lons, lats)

    llcrnrlat = 24
    urcrnrlat = 51
    llcrnrlon = -128
    urcrnrlon = -64
    figsize = (14, 8)
    cbrange = (0, 1)
    cmap = 'jet'
    fontsize = 16

    plt.figure(figsize=figsize)

    plt.subplot(221)
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = cal['bh'].values
    img_masked = np.ma.masked_invalid(img)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('bh (calibrated)', fontsize=fontsize)

    plt.subplot(222)
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = cal['bv'].values
    img_masked = np.ma.masked_invalid(img)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('bv (calibrated)', fontsize=fontsize)

    plt.subplot(223)
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = uncal['bh'].values
    img_masked = np.ma.masked_invalid(img)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('bh (uncalibrated)', fontsize=fontsize)

    plt.subplot(224)
    img = np.full(lons.shape, np.nan)
    img[tc.j_indg.values, tc.i_indg.values] = uncal['bv'].values
    img_masked = np.ma.masked_invalid(img)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    plt.title('bv (uncalibrated)', fontsize=fontsize)

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    plot_ismn_statistics()