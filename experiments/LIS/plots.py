
import os
import pathlib
import rasterio
import shapefile

import numpy as np
import pandas as pd

from netCDF4 import Dataset, num2date

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from mpl_toolkits.basemap import Basemap
from matplotlib.lines import Line2D

from datetime import date

def plot_img(img,lats,lons,
             cmap='jet_r',
             cbrange=(0.0,0.5),
             title='',
             cblabel='',
             fontsize=16,
             landcover=False):

    m = Basemap(projection='mill',
                llcrnrlat=49.395,
                urcrnrlat=51.605,
                llcrnrlon=2.395,
                urcrnrlon=6.505,
                resolution='h')

    m.drawcoastlines(linewidth=2.5)
    m.drawcountries(linewidth=2)
    m.readshapefile('/data_sets/LIS/shapefiles/catchments/be_demer_wgs84/be_demer_wgs84', 'Demer', linewidth=3, color='magenta')
    m.readshapefile('/data_sets/LIS/shapefiles/catchments/be_ourthe_wgs84/be_ourthe_wgs84', 'Ourthe', linewidth=3, color='darkred')

    custom_lines = [Line2D([0], [0], color='magenta', lw=4, label='Demer'),
                    Line2D([0], [0], color='darkred', lw=4, label='Ourthe')]
    plt.gca().legend(handles=custom_lines, loc='lower left', fontsize=fontsize)

    if landcover:
        cmap = plt.cm.get_cmap(cmap, 13)

    im = m.pcolormesh(lons, lats, img, cmap=cmap, latlon=True)

    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="5%", pad="2.5%")
    cb.set_label(cblabel, fontsize=fontsize, labelpad=-44)

    cb.ax.get_xaxis().set_ticks(np.arange(13))

    if landcover:
        labels = ['ENF', 'EBF', 'DNF', 'DBF', 'MF', 'WL', 'WGL', 'CSL', 'OSL', 'GL', 'CL', 'BS', 'U']
        cb.ax.set_xticklabels(labels)

    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)

    plt.title(title,fontsize=fontsize)

    with shapefile.Reader('/data_sets/LIS/shapefiles/BEL_adm/BEL_adm0') as sf:
        shape_rec = sf.shapeRecords()[0]
    vertices = []
    codes = []
    pts = shape_rec.shape.points
    prt = list(shape_rec.shape.parts) + [len(pts)]
    for i in range(len(prt) - 1):
        for j in range(prt[i], prt[i + 1]):
            vertices.append(m(pts[j][0], pts[j][1]))
        codes += [Path.MOVETO]
        codes += [Path.LINETO] * (prt[i + 1] - prt[i] - 2)
        codes += [Path.CLOSEPOLY]
    clip = PathPatch(Path(vertices, codes), transform=plt.gca().transData)
    im.set_clip_path(clip)


def plot_image(date=None, mode='ASCAT'):

    if not date:
        date = '2017-05-10 06:00:00'

    if mode == 'ASCAT':
        cbrange = [0, 50]
        cblabel = 'ASCAT H113/114 [%sat]'
    else:
        cbrange = [0.18, 0.35]
        cblabel = 'Noah-MP [m3/m3]'

    ds = Dataset('/data_sets/LIS/NoahMP_belgium/images.nc')
    lats = ds['lat'][:,:]
    lons = ds['lon'][:,:]
    lats = np.linspace(lats.min(), lats.max(), lats.shape[0])
    lons = np.linspace(lons.min(), lons.max(), lons.shape[1])
    lons, lats = np.meshgrid(lons, lats)

    if mode == 'ASCAT':
        ds.close()
        ds = Dataset('/data_sets/LIS/ASCAT/images.nc')

    dates = pd.to_datetime(num2date(ds['time'][:], units=ds['time'].units))
    ind = np.where(dates == date)[0][0]
    img = ds['SoilMoisture'][ind,:,:]
    ds.close()

    with rasterio.open('/data_sets/LIS/NoahMP_belgium/mask.tif') as ds:
        mask = np.flipud(ds.read()[0,:,:])
    img.mask[mask == 0] = True

    plt.figure(figsize=(12,10))

    plot_img(img, lats, lons, title=date, cbrange=cbrange, cblabel=cblabel)

    plt.tight_layout()
    plt.show()


def plot_catchment_averages():

    date = '2017-05-10 06:00:00'

    res = pd.read_csv('/data_sets/LIS/catchment_averages.csv', index_col=0, parse_dates=True)

    res.loc[:,'ascat_demer'] = (res.loc[:,'ascat_demer'] - res.loc[:,'ascat_demer'].mean()) / res.loc[:,'ascat_demer'].std() * res.loc[:,'noah_demer'].std() + res.loc[:,'noah_demer'].mean()
    res.loc[:,'ascat_ourthe'] = (res.loc[:,'ascat_ourthe'] - res.loc[:,'ascat_ourthe'].mean()) / res.loc[:,'ascat_ourthe'].std() * res.loc[:,'noah_ourthe'].std() + res.loc[:,'noah_ourthe'].mean()

    xlim = ['2017-01-01', '2018-01-01']
    ylim = [0.18, 0.42]
    fontsize = 14

    plt.figure(figsize=(18,8))

    ax = plt.subplot(2,1,1)

    res['noah_demer'].plot(ax=ax)
    res['ascat_demer'].dropna().plot(ax=ax)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(fontsize=fontsize, visible=False)
    plt.yticks(fontsize=fontsize)
    plt.title('Demer', fontsize=fontsize+2)
    plt.legend(['Noah-MP', 'ASCAT H113/114 (scaled)'], fontsize=fontsize, loc='upper left')
    plt.axvline(date, linestyle='--', color='k', linewidth=1.5)

    ax = plt.subplot(2,1,2)

    res['noah_ourthe'].plot(ax=ax)
    res['ascat_ourthe'].dropna().plot(ax=ax)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title('Ourthe', fontsize=fontsize+2)
    plt.axvline(date, linestyle='--', color='k', linewidth=1.5)

    plt.tight_layout()
    plt.show()

def plot_landcover():

    with Dataset('/data_sets/LIS/NoahMP_belgium/lis_input.noahMp36_OL_Belgium.nc') as ds:
        tmp_lc = ds.variables['LANDCOVER'][:,:,:]
        lats = ds.variables['lat'][:,:]
        lons = ds.variables['lon'][:,:]

    lc = np.full(lats.shape, -1)

    for i in range(tmp_lc.shape[0]):
        lc[tmp_lc[i,:,:]==1] = i

    plt.figure(figsize=(12, 10))
    plot_img(lc, lats, lons,
             cmap='nipy_spectral_r',
             cbrange=(-0.5, 12.5),
             title='Land Cover',
             cblabel='',
             fontsize=16,
             landcover=True)

    plt.tight_layout()
    plt.show()

if __name__=='__main__':

    # plot_image(mode='Noah')
    # plot_catchment_averages()
    plot_landcover()



