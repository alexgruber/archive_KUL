
import numpy as np
import pandas as pd

import platform
if platform.system() in ['Linux', 'Darwin']:
    import matplotlib
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.basemap import Basemap

import seaborn as sns
sns.set_context('talk', font_scale=0.8)

from pyldas.interface import GEOSldas_io, LDASsa_io

from myprojects.timeseries import calc_clim_harmonic, calc_clim_moving_average
from pytesmo.time_series.anomaly import calc_anomaly, calc_climatology

def plot_innov_ts_example():

    lat, lon = 41.70991616028507, -92.39133686043398

    io_smap = LDAS_io('ObsFcstAna', exp='US_M36_SMOS40_DA_cal_scaled')
    # io_smap = LDAS_io('ObsFcstAna', exp='US_M36_SMOS_DA_cal_scaled_yearly')

    idx_lon, idx_lat = io_smap.grid.lonlat2colrow(lon, lat, domain=True)
    print(idx_lon, idx_lat)

    ts = io_smap.timeseries.isel(lat=idx_lat, lon=idx_lon, species=0).to_dataframe()[['obs_fcst', 'obs_obs']].dropna()

    plt.figure(figsize=(21, 8))

    ax = plt.subplot(2, 1, 1)
    sns.lineplot(data=ts, dashes=False, ax=ax)
    plt.title(f'{lat:.2f} N, {lon:.2f} W')
    plt.xlabel('')
    plt.ylabel('Tb')

    # ---- cliamtology ----
    ts['obs_fcst_clim'] = calc_anomaly(ts['obs_fcst'], return_clim=True, climatology=calc_climatology(ts['obs_fcst']))[
        'climatology']
    ts['obs_obs_clim'] = calc_anomaly(ts['obs_obs'], return_clim=True, climatology=calc_climatology(ts['obs_obs']))[
        'climatology']

    ts['obs_fcst_seas'] = ts['obs_fcst'] - calc_anomaly(ts['obs_fcst'])
    ts['obs_obs_seas'] = ts['obs_obs'] - calc_anomaly(ts['obs_obs'])

    ax = plt.subplot(2, 1, 2)
    ts['climatology_scaled'] = ts['obs_obs'] - ts['obs_obs_clim'] + ts['obs_fcst_clim'] - ts['obs_fcst']
    ts['seasonality_scaled'] = ts['obs_obs'] - ts['obs_obs_seas'] + ts['obs_fcst_seas'] - ts['obs_fcst']
    sns.lineplot(data=ts[['climatology_scaled', 'seasonality_scaled']], dashes=False, ax=ax)
    plt.axhline(color='black', linewidth=1, linestyle='--')
    plt.xlabel('')
    plt.ylabel('O-F')

    plt.tight_layout()
    plt.show()

def plot_scaling_parameters():

    # fname = '/Users/u0116961/data_sets/LDASsa_runs/scaling_files/Thvf_TbSM_001_src_SMOSSMAP_trg_SMAP_2015_p19_2020_p19_W_9p_Nmin_20_A_p12.bin'
    fname = '/Users/u0116961/data_sets/GEOSldas_runs/_scaling_files_Pcorr_daily/y2020/Thvf_TbSM_001_src_SMAP_trg_SMAP_2015_p19_2021_p19_W_9p_Nmin_20_D_d340.bin'

    io = LDASsa_io('scaling')

    res = io.read_scaling_parameters(fname=fname)

    angle = 40

    res = res[['lon','lat','m_mod_H_%2i'%angle,'m_mod_V_%2i'%angle,'m_obs_H_%2i'%angle,'m_obs_V_%2i'%angle]]
    res.replace(-9999.,np.nan,inplace=True)

    lats = res['lat'].values
    lons = res['lon'].values

    llcrnrlat = 24
    urcrnrlat = 51
    llcrnrlon = -128
    urcrnrlon = -64

    figsize = (17,8)

    plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')

    ax = plt.subplot(221)
    m = Basemap(projection='mill',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    x,y = m(lons,lats)
    ax.scatter(x,y,s=10,c=res['m_obs_H_%2i'%angle].values,marker='o', cmap='jet',vmin=220,vmax=300)

    ax = plt.subplot(222)
    m = Basemap(projection='mill',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    x,y = m(lons,lats)
    ax.scatter(x,y,s=10,c=res['m_mod_H_%2i'%angle].values,marker='o', cmap='jet', vmin=220, vmax=300)

    ax = plt.subplot(223)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    x, y = m(lons, lats)
    ax.scatter(x, y, s=10, c=res['m_obs_V_%2i'%angle].values, marker='o', cmap='jet', vmin=220, vmax=300)

    ax = plt.subplot(224)
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    x, y = m(lons, lats)
    ax.scatter(x, y, s=10, c=res['m_mod_V_%2i'%angle].values, marker='o', cmap='jet', vmin=220, vmax=300)

    plt.tight_layout()
    plt.show()

def plot_Tb_clims():

    # ts = LDAS_io('ObsFcstAna', exp='NLv4_M36_US_SMAP_TB_OL').timeseries
    ts = LDAS_io('ObsFcstAna', exp='US_M36_SMAP_TB_OL_noScl', root='/Users/u0116961/data_sets/LDASsa_runs').timeseries
    data = pd.DataFrame(index=ts.time.values)
    data['obs'] = pd.to_numeric(ts['obs_obs'][:,0,45,30].values,errors='coerce')
    data['fcst'] = pd.to_numeric(ts['obs_fcst'][:,0,45,30].values,errors='coerce')

    data_plt = data.copy().resample('1d').first()
    data_plt.interpolate(inplace=True)

    plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3])

    ax = plt.subplot(gs[1])
    arr = data_plt['obs']
    yrs = np.unique(arr.index.year)
    df = pd.DataFrame(columns=yrs,index=np.arange(366)+1,dtype='float')
    for yr in yrs:
        df[yr].loc[arr[arr.index.year == yr].index.dayofyear] = arr[arr.index.year == yr].values
        df[yr] = df[yr]
    df.drop(366,axis='index',inplace=True)
    df.plot(ax=ax,linestyle='--',linewidth=0.5)
    arr = data['obs'].copy()
    clim = pd.DataFrame(index=np.arange(365)+1)
    for n in [0,1,2,3,10,]:
        clim[n] = calc_clim_harmonic(arr,n=n)
    clim['moving_average'] = calc_clim_moving_average(arr,window_size=45)
    clim.plot(linewidth=2,ax=ax)
    ax.set_xlim((0,365))
    ax.set_ylim((200,295))
    ax.legend(loc=2)

    ax = plt.subplot(gs[0])
    doys = arr.index.dayofyear.values
    modes = clim.columns.values
    rmse = pd.Series(index=modes)
    rmse[:] = 0.
    for mod in modes:
        n = 0
        for doy in np.arange(365)+1:
            n += len(arr[doys==doy])
            rmse[mod] += ((arr[doys==doy] - clim.loc[doy,mod])**2).sum()
        rmse[mod] = np.sqrt(rmse[mod]/n)
    ax = rmse.plot(ax=ax)
    ax.set_xticks(np.arange(len(modes)+1))
    ax.set_xticklabels(modes)

    ax = plt.subplot(gs[3])
    arr = data_plt['fcst']
    yrs = np.unique(arr.index.year)
    df = pd.DataFrame(columns=yrs,index=np.arange(366)+1,dtype='float')
    for yr in yrs:
        df[yr].loc[arr[arr.index.year == yr].index.dayofyear] = arr[arr.index.year == yr].values
        df[yr] = df[yr]
    df.drop(366,axis='index',inplace=True)
    df.plot(ax=ax,linestyle='--',linewidth=0.5)
    arr = data['fcst'].copy()
    clim = pd.DataFrame(index=np.arange(365)+1)
    for n in [0,1,2,3,10,]:
        clim[n] = calc_clim_harmonic(arr,n=n)
    clim['moving_average'] = calc_clim_moving_average(arr,window_size=45)
    clim.plot(linewidth=2,ax=ax)
    ax.set_xlim((0,365))
    ax.set_ylim((220,300))
    ax.legend(loc=2)

    ax = plt.subplot(gs[2])
    doys = arr.index.dayofyear.values
    modes = clim.columns.values
    rmse = pd.Series(index=modes)
    rmse[:] = 0.
    for mod in modes:
        n = 0
        for doy in np.arange(365)+1:
            n += len(arr[doys==doy])
            rmse[mod] += ((arr[doys==doy] - clim.loc[doy,mod])**2).sum()
        rmse[mod] = np.sqrt(rmse[mod]/n)
    ax = rmse.plot(ax=ax)
    ax.set_xticks(np.arange(len(modes)+1))
    ax.set_xticklabels(modes)

    plt.tight_layout()
    plt.show()

if __name__=='__main__':

    plot_scaling_parameters()
    # plot_Tb_clims()
    # plot_innov_ts_example()