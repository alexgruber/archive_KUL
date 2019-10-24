

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from pyldas.interface import LDAS_io
from pyldas.templates import template_ObsFcstAnaEns, template_ObsFcstAna, template_error_Tb40
from pyldas.visualize.plots import plot_ease_img

def plot_image():

    # exp = 'US_M36_SMOS40_TB_ens_test_OL'
    # param = 'ObsFcstAna'

    io = LDAS_io()

    # fname = '/data_sets/LDAS_runs/US_M36_SMOS40_TB_ens_test_DA/output/SMAP_EASEv2_M36_US/ana/ens0008/Y2013/M10/US_M36_SMOS40_TB_ens_test_DA.ens0008.ldas_ObsFcstAnaEns.20131014_1200z.bin'
    # fname = '/Users/u0116961/Desktop/temp/US_M36_SMOS40_TB_ens_test_DA2.ens0000.ldas_ObsFcstAnaEns.20100114_1200z.bin'

    fname = '/work/MadKF/CLSM/iter_2/error_files/SMOS_fit_Tb_A.bin'

    dtype, hdr, length = template_error_Tb40()
    img = io.read_fortran_binary(fname, dtype, hdr=hdr, length=length)

    # img = img[img.obs_species==1]
    # img.index=img.obs_tilenum
    #
    # img[img==-9999] = np.nan

    img.index += 1

    plot_ease_img(img,'err_Tbh', cbrange=(0,10),
                  llcrnrlat=14,
                  urcrnrlat=61,
                  llcrnrlon=-138,
                  urcrnrlon=-54,)

def plot_ens_ts():
    lat = 36.02747
    lon = -100.112

    spc = 1

    exp = 'US_M36_SMOS40_TB_ens_test_DA'

    param = 'ObsFcstAna'
    io_avg = LDAS_io(param, exp)

    param = 'ObsFcstAnaEns'
    io_ens = LDAS_io(param, exp)

    fid = plt.figure(figsize=(23,8))

    width_ens = 0.2
    width_avg = 2.5

    ts = io_avg.read_ts('obs_fcst', lon, lat, species=spc).dropna()
    ts.plot(linewidth=width_avg, color='red', ax=plt.gca(), marker='o', markersize=6)

    ts = io_avg.read_ts('obs_obs', lon, lat, species=spc).dropna()
    # ts.plot(linewidth=width_avg, color='blue', ax=plt.gca())
    ts.plot(linewidth=width_avg, color='blue', ax=plt.gca(), marker='o', markersize=6)

    ts = io_avg.read_ts('obs_ana', lon, lat, species=spc).dropna()
    ts.plot(linewidth=width_avg, color='green', ax=plt.gca(), marker='o', markersize=6)

    plt.legend(['Forecast','Observation','Analysis'], fontsize=30)

    ts = io_ens.read_ts('obs_fcst', lon, lat, species=spc).dropna()
    ts.plot(linestyle='--', linewidth=width_ens, color='red', ax=plt.gca(),legend=None)
    ts = io_ens.read_ts('obs_obs', lon, lat, species=spc).dropna()
    ts.plot(linestyle='--', linewidth=width_ens, color='blue', ax=plt.gca(), legend=None)
    # ts.plot(linestyle='--', linewidth=width_ens, color='blue', ax=plt.gca(),legend=None, marker='o', markersize=6)
    ts = io_ens.read_ts('obs_ana', lon, lat, species=spc).dropna()
    ts.plot(linestyle='--', linewidth=width_ens, color='green', ax=plt.gca(),legend=None)

    plt.xlim(['2013-08-25 00:00:00','2013-10-21 00:00:00'])
    plt.ylim([210,280])


    # plt.xlim(['2013-03-01','2013-12-31'])
    # plt.ylim([230,280])

    # plt.xlim(['2012-01-01','2014-12-31'])
    # plt.ylim([230,280])

    plt.tight_layout()
    plt.show()

if __name__=='__main__':

    plot_ens_ts()
    # plot_image()

    #
    # ts = io.read_ts('obs_obs', lon, lat, species=spc).dropna()
    # ts.plot(linewidth=width,color='blue', ax=plt.gca())
    #
    # ts = io.read_ts('obs_ana', lon, lat, species=spc).dropna()
    # ts.plot(linewidth=width,color='green', ax=plt.gca())
    #
    #
    #
    # plt.show()

    #
    # latmin = 35.
    # latmax = 45.
    # lonmin = -180.
    # lonmax = -170.
    # io.bin2netcdf(overwrite=True)


