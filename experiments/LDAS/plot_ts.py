
import numpy as np
import pandas as pd

import matplotlib

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk', font_scale=0.8)

from datetime import date

from pyldas.interface import LDAS_io

def plot_ts():

    lat, lon = 42.23745409478888, -117.08806967006959

    exp0 = 'US_M36_SMAP_TB_OL_scaled_4K_obserr'
    exp1 = 'US_M36_SMAP_TB_DA_SM_ERR_scl_clim_anom_lt_11'
    exp2 = 'US_M36_SMAP_TB_DA_SM_ERR_scl_seas_anom_st_1'

    xhr0 = LDAS_io('xhourly', exp=exp0)

    ofa1 = LDAS_io('ObsFcstAna', exp=exp1)
    xhr1 = LDAS_io('xhourly', exp=exp1)

    ofa2 = LDAS_io('ObsFcstAna', exp=exp2)
    xhr2 = LDAS_io('xhourly', exp=exp2)

    idx_lon, idx_lat = ofa1.grid.lonlat2colrow(lon, lat, domain=True)

    ts_sm0 = xhr0.timeseries.isel(lat=idx_lat, lon=idx_lon).to_dataframe()[['sm_rootzone',]].dropna()
    ts_sm0.columns = ['open_loop']

    ts_ofa1 = ofa1.timeseries.isel(lat=idx_lat, lon=idx_lon, species=1).to_dataframe()[['obs_ana', 'obs_fcst', 'obs_obs']].dropna()
    ts_sm1 = xhr1.timeseries.isel(lat=idx_lat, lon=idx_lon).to_dataframe()[['sm_rootzone',]].dropna()
    ts_sm1.columns = ['climatology-scaled']

    ts_ofa2 = ofa2.timeseries.isel(lat=idx_lat, lon=idx_lon, species=1).to_dataframe()[['obs_ana', 'obs_fcst', 'obs_obs']].dropna()
    ts_sm2 = xhr2.timeseries.isel(lat=idx_lat, lon=idx_lon).to_dataframe()[['sm_rootzone',]].dropna()
    ts_sm2.columns = ['seasonality-scaled']

    plt.figure(figsize=(21, 10))

    # ax = plt.subplot(4, 1, 1)
    # sns.lineplot(data=ts_ofa1, dashes=False, ax=ax)
    # plt.title(f'{lat:.2f} N, {lon:.2f} W')
    # plt.xlabel('')
    # ax.get_xaxis().set_ticks([])
    # # plt.ylabel('Tb')
    # plt.ylim(125,290)
    # plt.xlim(date(2015,3,1), date(2020,5,1))
    #
    # ax = plt.subplot(4, 1, 2)
    # sns.lineplot(data=ts_ofa2, dashes=False, ax=ax)
    # plt.title(f'{lat:.2f} N, {lon:.2f} W')
    # plt.xlabel('')
    # ax.get_xaxis().set_ticks([])
    # # plt.ylabel('Tb')
    # plt.ylim(125,290)
    # plt.xlim(date(2015,3,1), date(2020,5,1))

    ax = plt.subplot(2, 1, 1)
    ts_ofa1['innov (clim-scaled)'] = ts_ofa1['obs_obs'] - ts_ofa1['obs_ana']
    ts_ofa1['innov (seas-scaled)'] = ts_ofa2['obs_obs'] - ts_ofa2['obs_ana']
    sns.lineplot(data=ts_ofa1[['innov (clim-scaled)','innov (seas-scaled)']], dashes=False, ax=ax, linewidth=1.5)
    plt.axhline(color='black', linewidth=1, linestyle='--')
    plt.xlabel('')
    ax.get_xaxis().set_ticks([])
    # plt.ylabel('O-F')
    plt.ylim(-25,25)
    plt.xlim(date(2015, 3, 1), date(2020, 5, 1))

    ax = plt.subplot(2, 1, 2)
    sns.lineplot(data=ts_sm0, dashes=False, ax=ax, linewidth=1)
    sns.lineplot(data=ts_sm1, dashes=False, ax=ax, linewidth=1, palette=['darkorange'])
    sns.lineplot(data=ts_sm2, dashes=False, ax=ax, linewidth=1, palette=['green'])
    plt.xlabel('')
    # plt.ylabel('SM')
    plt.ylim(0.0,0.55)
    plt.xlim(date(2015, 3, 1), date(2020, 5, 1))

    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    plot_ts()