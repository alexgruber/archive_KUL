

import numpy as np
import pandas as pd

from pathlib import Path

from netCDF4 import Dataset, num2date, date2num

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def calc_averages():

    root = Path('/data_sets/ESA_CCI_SM/ACTIVE/v04.4')

    years = ['%i' % yr for yr in range(2007,2018)]

    yr_avg = np.full((len(years), 720, 1440), 0.)
    yr_cnt = np.full((len(years), 720, 1440), 0)

    mo_avg = np.full((len(years)*12, 720, 1440), 0.)
    mo_cnt = np.full((len(years)*12, 720, 1440), 0)

    with Dataset('/data_sets/ESA_CCI_SM/ESA-CCI-SOILMOISTURE-LAND_AND_RAINFOREST_MASK-fv04.2.nc') as f:
        land_mask = f['land'][:,:]

    files = []
    for yr in years:
        files += list((root / yr).glob('*.nc'))

    files = np.sort(files)

    for i_yr, yr, in enumerate(years):
        for i_mo in range(12):

            print(yr, '%i' % (i_mo+1))

            i = i_yr*12 + i_mo

            for f in root.glob('**/*' + yr + '%02i*.nc' % (i_mo+1)):

                with Dataset(f) as ds:

                    mask = np.where((ds['flag'][0,:,:] == 0) & (land_mask == 1))

                    mo_avg[i,mask[0],mask[1]] += ds['sm'][0,:,:][mask]
                    mo_cnt[i,mask[0],mask[1]] += 1

                    yr_avg[i_yr,mask[0],mask[1]] += ds['sm'][0,:,:][mask]
                    yr_cnt[i_yr,mask[0],mask[1]] += 1

    mo_avg[mo_cnt >= 7] /= mo_cnt[mo_cnt >= 7]
    yr_avg[yr_cnt >= 13] /= yr_cnt[yr_cnt >= 13]
    mo_avg[mo_cnt < 7] = np.nan
    yr_avg[yr_cnt < 13] = np.nan

    np.save('/data_sets/ESA_CCI_SM/_aggregates/monthly_avg_active_v04.4_2007_2017', mo_avg)
    np.save('/data_sets/ESA_CCI_SM/_aggregates/yearly_avg_active_v04.4_2007_2017', yr_avg)


def plot_figure(img, title=''):

    m = Basemap(projection='mill',
                    llcrnrlat=-60,
                    urcrnrlat=90,
                    llcrnrlon=-180,
                    urcrnrlon=180,
                    resolution='c')

    fontsize = 14

    lons = (np.arange(360 * 4) * 0.25) - 179.875
    lats = (np.arange(180 * 4) * 0.25) - 89.875
    lons, lats = np.meshgrid(lons, lats)

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, img, cmap='RdYlBu', latlon=True)
    im.set_clim(vmin=0, vmax=100)

    cb = m.colorbar(im, "bottom", size="8%", pad="4%")
    for t in cb.ax.get_xticklabels():
         t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
         t.set_fontsize(fontsize)

    plt.title(title, fontsize=fontsize)



def plot_averages():


    yr_avg = np.load('/data_sets/ESA_CCI_SM/_aggregates/yearly_avg_active_v04.4_2007_2017.npy')
    for i_yr, yr in enumerate(range(2007,2018)):

        fout = '/data_sets/ESA_CCI_SM/_aggregates/plots/yearly/%i.png' % yr

        f = plt.figure(figsize=(20,8))
        img = np.ma.masked_invalid(np.flipud(yr_avg[i_yr,:,:]))
        plot_figure(img, title='%i' % yr)
        f.savefig(fout, dpi=300, bbox_inches='tight')
        plt.close()

    mo_avg = np.load('/data_sets/ESA_CCI_SM/_aggregates/monthly_avg_active_v04.4_2007_2017.npy')
    for i_yr, yr in enumerate(range(2007,2018)):
        for i_mo in range(12):
            i = i_yr*12 + i_mo

            fout = '/data_sets/ESA_CCI_SM/_aggregates/plots/monthly/%i%02i.png' % (yr, i_mo+1)

            f = plt.figure(figsize=(20,8))
            img = np.ma.masked_invalid(np.flipud(mo_avg[i,:,:]))
            plot_figure(img, title='%i-%02i' % (yr,i_mo+1))
            f.savefig(fout, dpi=300, bbox_inches='tight')
            plt.close()

if __name__=='__main__':

    calc_averages()
    plot_averages()














