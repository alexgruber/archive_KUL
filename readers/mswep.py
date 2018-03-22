

import os
import platform

import numpy as np
import pandas as pd
import xarray as xr

from collections import OrderedDict

from netCDF4 import Dataset, num2date, date2num

from myprojects.timeseries import calc_anomaly
from myprojects.readers.ascat import HSAF_io
from myprojects.functions import find_files

from scipy.optimize import fminbound
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


class MSWEP_io(object):

    def __init__(self, root=None):

        if root is None:
            if platform.system() == 'Windows':
                root = os.path.join('D:','data_sets', 'MSWEP_v21')
            else:
                root = os.path.join('/', 'data', 'leuven', '320', 'vsc32046', 'data_sets', 'MSWEP')

        self.ds = Dataset(np.atleast_1d(find_files(root,'.nc4'))[0])
        self.grid = pd.read_csv(find_files(root,'grid.csv'), index_col=0)


    def read(self, *args):

        if len(args) == 1:
            row = self.grid.loc[args[0],'row']
            col = self.grid.loc[args[0],'col']
        else:
            col = args[0]
            row = args[1]

        ts = self.ds['precipitation'][:, row, col]
        dates = num2date(self.ds['time'][:], units=self.ds['time'].units)

        return pd.Series(ts, index=dates)


    def iter_gp(self):
        for cell in np.unique(self.grid.dgg_cell):
            gpis = self.grid[self.grid.dgg_cell==cell]
            for gpi, info in gpis.iterrows():
                data = self.read(gpi)
                yield data, info


    def iter_cell(self, cell):
        gpis = self.grid[self.grid.dgg_cell==cell]
        for gpi, info in gpis.iterrows():
            data = self.read(gpi)
            yield data, info


    def close(self):
        self.ds.close()


def combine_tile_files():

    root = r'D:\data_sets\MSWEP_V21\data'
    fname = os.path.join(root,'CONUS_2007_2016_daily_01deg.nc4')
    gridfile = os.path.join(root,'grid.csv')

    dgg_grid = Dataset(r"D:\data_sets\ASCAT\warp5_grid\TUW_WARP5_grid_info_2_2.nc")
    conus_gpis = pd.read_csv(r"D:\data_sets\ASCAT\warp5_grid\pointlist_warp_conus.csv")['point'].values
    dgg_lats = dgg_grid['lat'][:]
    dgg_lons = dgg_grid['lon'][:]
    dgg_cells = dgg_grid['cell'][:]
    dgg_gpis = dgg_grid['gpi'][:]
    dgg_land = dgg_grid['land_flag'][:]

    tiles = [150, 151, 152, 153, 154, 155, 156, 186, 187, 188, 189, 190, 191, 192, 222, 223, 224, 225, 226, 227, 228]

    timeunit = 'hours since 2000-01-01 00:00'
    punit = 'mm/d'

    ds = xr.open_mfdataset(os.path.join(root,'tiles','*.nc4'))
    lats = np.unique(ds.lat).astype('float32')
    lons = np.unique(ds.lon).astype('float32')
    dates = date2num(pd.to_datetime(ds['time'].values).to_pydatetime(), timeunit).astype('int32')

    ds = Dataset(fname,mode='w')
    dimensions = OrderedDict([('time', dates), ('lat', lats), ('lon', lons)])

    chunksizes = []
    for key, values in dimensions.iteritems():

        if key in ['lon', 'lat']:
            chunksize = 1
        else:
            chunksize = len(values)
        chunksizes.append(chunksize)

        dtype = values.dtype
        ds.createDimension(key, len(values))
        ds.createVariable(key, dtype,
                          dimensions=(key,),
                          chunksizes=(chunksize,),
                          zlib=True)
        ds[key][:] = values

    ds.createVariable('precipitation', 'float32',
                      dimensions=dimensions.keys(),
                      chunksizes=chunksizes,
                      fill_value=-9999.,
                      zlib=True)

    ds.variables['time'].setncattr('units', timeunit)
    ds.variables['precipitation'].setncattr('units', punit)

    n_gp = len(lats) * len(lons)

    grid = pd.DataFrame(columns=['lat', 'lon', 'row', 'col', 'dgg_cell', 'dgg_gpi'], index=np.arange(n_gp))

    i = -1
    for tile in tiles:

        tmp_tile = xr.open_mfdataset(os.path.join(root,'tiles','*%i.nc4' % tile))
        tmp_lats = tmp_tile['lat'].values.astype('float32')
        tmp_lons = tmp_tile['lon'].values.astype('float32')

        for r, lat in enumerate(tmp_lats):
            for c, lon in enumerate(tmp_lons):

                i += 1
                print '%i / %i' % (i, n_gp)

                if (lat<25.) | (lat>49.) | (lon<-124.5) | (lon>-67):
                    continue

                latdiff = abs(dgg_lats - lat)
                londiff = abs(dgg_lons - lon)
                dist = np.linalg.norm(np.vstack((latdiff, londiff)), axis=0)
                ind_dgg = np.where(dist-dist.min()<0.0001)[0][0]

                if (dgg_land[ind_dgg]==0) | (dgg_gpis[ind_dgg] not in conus_gpis):
                    continue

                grid.loc[i,'lat'] = lat
                grid.loc[i,'lon'] = lon
                grid.loc[i,'row'] = r
                grid.loc[i,'col'] = c
                grid.loc[i,'dgg_cell'] = dgg_cells[ind_dgg]
                grid.loc[i,'dgg_gpi'] = dgg_gpis[ind_dgg]

                ind_lat = np.where(lats == lat)[0][0]
                ind_lon = np.where(lons == lon)[0][0]

                ds['precipitation'][:,ind_lat,ind_lon] = tmp_tile['precipitation'][:,r,c]

    grid.dropna().to_csv(gridfile)

    ds.close()


def correct_grid_col_row():

    io = MSWEP_io()
    lats = io.ds['lat'][:]
    lons = io.ds['lon'][:]

    for gpi, data in io.grid.iterrows():
        io.grid.loc[gpi, 'row'] = np.where(lats == data['lat'])[0][0]
        io.grid.loc[gpi, 'col'] = np.where(lons == data['lon'])[0][0]

    io.grid.to_csv(r"D:\data_sets\MSWEP_V21\data\grid_corrected.csv")

def calc_api(gamma, precip):
    n = len(precip)
    API = np.zeros(len(precip))
    for t in np.arange(1, n):
        API[t] = gamma * API[t - 1] + precip[t]
    return API

def corr_gamma(gamma, precip, sm):
    api = calc_api(gamma, precip)
    tmp = pd.DataFrame({1: api, 2: sm}).dropna()
    return (1 - pearsonr(tmp[1], tmp[2])[0])

def estimate_gamma(precip, sm):
    gamma = fminbound(corr_gamma, 0.0, 1.0, args=(precip, sm), xtol=1e-03)
    return gamma

def calc_gamma_map():

    fname = r"D:\data_sets\MSWEP_V21\data\grid_new.csv"

    ascat = HSAF_io()
    mswep = MSWEP_io()

    mswep.grid['gamma'] = np.nan

    for i, (precip, info) in enumerate(mswep.iter_gp()):
        print i

        if len(precip.dropna()) == 0:
            continue
        try:
            precip = calc_anomaly(precip, method='harmonic', longterm=False)
            sm = calc_anomaly(ascat.read(info.dgg_gpi)['2007-01-01':'2016-12-31'], method='harmonic', longterm=False)
            ts = pd.concat((precip, sm), axis=1).values
            mswep.grid.loc[info.name,'gamma'] = estimate_gamma(ts[:,0], ts[:,1])
        except:
            continue

    mswep.grid.dropna().to_csv(fname)


def plot_gamma():

    io = MSWEP_io()

    lats = io.ds['lat'][:]
    lons = io.ds['lon'][:]
    lons, lats = np.meshgrid(lons, lats)

    rows = io.grid['row'].values
    cols = io.grid['col'].values
    gamma = io.grid['gamma'].values

    img = np.full(lats.shape, np.nan)
    ind = (rows,cols)
    img[ind] = gamma
    img_masked = np.ma.masked_invalid(img)

    llcrnrlat = 24
    urcrnrlat = 51
    llcrnrlon = -128
    urcrnrlon = -64

    figsize = (20, 10)
    cbrange = [0.4,0.9]
    cmap = 'YlGn'

    plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')

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
    m.colorbar(im, "bottom", size="7%", pad="8%")

    plt.show()

if __name__=='__main__':
    plot_gamma()

