
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from mpl_toolkits.basemap import Basemap
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk', font_scale=0.8)

from myprojects.publications.deforestation_paper.interface import io

def plot_img(lons, lats, data,
              llcrnrlat=-56.,
              urcrnrlat=13.,
              llcrnrlon=-82,
              urcrnrlon=-34,
              cbrange=None,
              cmap='jet',
              title='',
              fontsize=16,
              plot_cmap=True):

    img_masked = np.ma.masked_invalid(data)

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

    if cbrange:
        im.set_clim(vmin=cbrange[0], vmax=cbrange[1])

    if plot_cmap:
        cb = m.colorbar(im, "bottom", size="4%", pad="2%")
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(fontsize)

    if title != '':
        plt.title(title,fontsize=fontsize)

    # x, y = m(-79, 27.5)
    # plt.text(x, y, 'mean', fontsize=fontsize - 5)
    # x, y = m(-74, 27.5)
    # plt.text(x, y, '   =%.2f' % np.ma.median(img_masked), fontsize=fontsize - 5)
    # x, y = m(-79, 25)
    # plt.text(x, y, 'std.', fontsize=fontsize - 5)
    # x, y = m(-74, 25)
    # plt.text(x, y, '   =%.2f' % np.ma.std(img_masked), fontsize=fontsize - 5)

    return im



def AGB_VOD_LAI():

    ds_agb = io('AGB')
    ds_vod = io('SMOS_IC')
    ds_lai = io('LAI')

    agb = ds_agb.read_img('AGB')
    agb_err = ds_agb.read_img('AGB_err')
    # agb[agb_err > 150] = np.nan

    lai = np.nanmean(ds_lai.read_img('LAI', date_from='01-01-2010', date_to='31-12-2010'), axis=0)

    vod = ds_vod.read_img('VOD', date_from='01-01-2010', date_to='31-12-2010')
    invalid = (ds_vod.read_img('Flags', date_from='01-01-2010', date_to='31-12-2010') > 0) | \
              (ds_vod.read_img('RMSE', date_from='01-01-2010', date_to='31-12-2010') > 8) | \
              (ds_vod.read_img('VOD_StdErr', date_from='01-01-2010', date_to='31-12-2010') > 1.2)
    vod[invalid] = np.nan

    vod = np.nanmean(vod, axis=0)

    # Collocation
    invalid = np.where(np.isnan(agb) | np.isnan(vod) | np.isnan(lai))
    agb[invalid] = np.nan
    lai[invalid] = np.nan
    vod[invalid] = np.nan
    valid = np.where(~np.isnan(agb))

    fontsize = 10

    f = plt.figure(figsize=(13,11))

    gs = gridspec.GridSpec(3, 3, height_ratios=[2,1,0.025], wspace=0.35, hspace=0.20)

    plt.subplot(gs[0,0])
    plot_img(ds_agb.lon, ds_agb.lat, agb, cbrange=(0,100), cmap='viridis_r', title='AGB [Mg/ha]', fontsize=fontsize)

    plt.subplot(gs[0,1])
    plot_img(ds_agb.lon, ds_agb.lat, lai, cbrange=(0,5), cmap='viridis_r', title='LAI [-]', fontsize=fontsize)

    plt.subplot(gs[0,2])
    plot_img(ds_agb.lon, ds_agb.lat, vod, cbrange=(0,0.8), cmap='viridis_r', title='VOD [-]', fontsize=fontsize)

    ax = plt.subplot(gs[1,0])
    x, y = agb[valid], lai[valid]
    ax.hexbin(x, y,
              gridsize=35, bins='log',
              cmap='viridis', mincnt=1)
    xs = np.linspace(x.min(),x.max(), 100)
    # p = np.poly1d(np.polyfit(x, y, 1))
    # plt.plot(xs, p(xs), linestyle='--', linewidth=1, color='k')
    p = np.poly1d(np.polyfit(x, y, 3))
    plt.plot(xs, p(xs), linestyle='-.', linewidth=1, color='k')
    corr = np.corrcoef(x, y)[0,1]
    plt.title(f'R = {corr:.3f}', fontsize=fontsize)
    plt.xlabel('AGB [Mg/ha]', fontsize=fontsize)
    plt.ylabel('LAI [-]', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    ax = plt.subplot(gs[1,1])
    x, y = agb[valid], vod[valid]
    ax.hexbin(x, y,
              gridsize=35, bins='log',
              cmap='viridis', mincnt=1)
    xs = np.linspace(x.min(),x.max(), 100)
    # p = np.poly1d(np.polyfit(x, y, 1))
    # plt.plot(xs, p(xs), linestyle='--', linewidth=1, color='k')
    p = np.poly1d(np.polyfit(x, y, 3))
    plt.plot(xs, p(xs), linestyle='-.', linewidth=1, color='k')
    corr = np.corrcoef(x, y)[0,1]
    plt.title(f'R = {corr:.3f}', fontsize=fontsize)
    plt.xlabel('AGB [Mg/ha]', fontsize=fontsize)
    plt.ylabel('VOD [-]', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    ax = plt.subplot(gs[1,2])
    x, y = lai[valid], vod[valid]
    ax.hexbin(x, y,
              gridsize=35, bins='log',
              cmap='viridis', mincnt=1)
    xs = np.linspace(x.min(),x.max(), 100)
    # p = np.poly1d(np.polyfit(x, y, 1))
    # plt.plot(xs, p(xs), linestyle='--', linewidth=1, color='k')
    p = np.poly1d(np.polyfit(x, y, 3))
    plt.plot(xs, p(xs), linestyle='-.', linewidth=1, color='k')
    corr = np.corrcoef(x, y)[0,1]
    plt.title(f'R = {corr:.3f}', fontsize=fontsize)
    plt.xlabel('LAI [-]', fontsize=fontsize)
    plt.ylabel('VOD [-]', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # plt.tight_layout()
    # plt.show()

    fname = '/Users/u0116961/Documents/work/deforestation_paper/AGB_LAI_VOD.png'
    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def mean_veg_met():

    ds_lai = io('LAI')
    ds_vod = io('SMOS_IC')
    ds_met = io('MERRA2')

    date_from = '2011-01-01'
    date_to = '2011-12-31'

    lai = np.nanmean(ds_lai.read_img('LAI', date_from=date_from, date_to=date_to), axis=0)

    vod = ds_vod.read_img('VOD', date_from=date_from, date_to=date_to)
    invalid = (ds_vod.read_img('Flags', date_from=date_from, date_to=date_to) > 0) | \
              (ds_vod.read_img('RMSE', date_from=date_from, date_to=date_to) > 8) | \
              (ds_vod.read_img('VOD_StdErr', date_from=date_from, date_to=date_to) > 1.2)
    vod[invalid] = np.nan

    vod = np.nanmean(vod, axis=0)

    temp = np.nanmean(ds_met.read_img('T2M', date_from=date_from, date_to=date_to), axis=0)
    prec = np.nanmean(ds_met.read_img('PRECTOTLAND', date_from=date_from, date_to=date_to), axis=0)
    rad = np.nanmean(ds_met.read_img('LWLAND', date_from=date_from, date_to=date_to)+
                     ds_met.read_img('SWLAND', date_from=date_from, date_to=date_to), axis=0)

    temp_ease = np.full(lai.shape, np.nan)
    prec_ease = np.full(lai.shape, np.nan)
    rad_ease = np.full(lai.shape, np.nan)
    temp_ease[ds_lai.lut.row_ease, ds_lai.lut.col_ease] = temp[ds_lai.lut.row_merra, ds_lai.lut.col_merra]
    prec_ease[ds_lai.lut.row_ease, ds_lai.lut.col_ease] = prec[ds_lai.lut.row_merra, ds_lai.lut.col_merra]
    rad_ease[ds_lai.lut.row_ease, ds_lai.lut.col_ease] = rad[ds_lai.lut.row_merra, ds_lai.lut.col_merra]

    # Collocation
    invalid = np.where(np.isnan(vod) | np.isnan(lai) | np.isnan(temp_ease) | np.isnan(prec_ease) | np.isnan(rad_ease) )
    lai[invalid] = np.nan
    vod[invalid] = np.nan
    temp_ease[invalid] = np.nan
    prec_ease[invalid] = np.nan
    rad_ease[invalid] = np.nan

    fontsize = 12

    f = plt.figure(figsize=(18,10))

    # gs = gridspec.GridSpec(3, 3, height_ratios=[2,1,0.025], wspace=0.35, hspace=0.20)

    plt.subplot(1, 5, 1)
    plot_img(ds_lai.lon, ds_lai.lat, lai, cbrange=(0,5), cmap='viridis_r', title='LAI [-]', fontsize=fontsize)

    plt.subplot(1, 5, 2)
    plot_img(ds_lai.lon, ds_lai.lat, vod, cbrange=(0,0.8), cmap='viridis_r', title='VOD [-]', fontsize=fontsize)

    plt.subplot(1,5,3)
    plot_img(ds_lai.lon, ds_lai.lat, temp_ease, cbrange=(5, 25), cmap='viridis_r', title='T [$^\circ$C]', fontsize=fontsize)

    plt.subplot(1, 5, 4)
    plot_img(ds_lai.lon, ds_lai.lat, prec_ease, cbrange=(0,15), cmap='viridis_r', title='P [kg / m2 / s]', fontsize=fontsize)

    plt.subplot(1, 5, 5)
    plot_img(ds_lai.lon, ds_lai.lat, rad_ease, cbrange=(50,150), cmap='viridis_r', title='Rad. [W / m2]', fontsize=fontsize)

    plt.tight_layout()
    plt.show()

    # fname = '/Users/u0116961/Documents/work/deforestation_paper/mean_veg_met.png'
    # f.savefig(fname, dpi=300, bbox_inches='tight')
    # plt.close()


if __name__=='__main__':
    AGB_VOD_LAI()
    mean_veg_met()