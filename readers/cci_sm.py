
import os
import platform

import numpy as np
import pandas as pd

from collections import OrderedDict

from myprojects.functions import find_files
from netCDF4 import Dataset, num2date, date2num

import matplotlib.pyplot as plt

class CCISM_io(object):

    def __init__(self):

        if platform.system() == 'Windows':
            self.root = os.path.join('D:','data_sets', 'ESA_CCI_SM')
        else:
            self.root = os.path.join('/','data_sets', 'ESA_CCI_SM')


        self.versions = ['v02.2','v03.3', 'v04.4']
        self.modes = ['ACTIVE', 'PASSIVE', 'COMBINED']

        self.date_range = ['2007-10-01', '2014-12-31']

        self.lats = np.arange(-89.875,90,0.25)[::-1]
        self.lons = np.arange(-179.875,180,0.25)

        names = [m + '_' + v for m in self.modes for v in self.versions]
        files = [os.path.join(self.root, '_rechunked', name + '.nc') for name in names]
        datasets = [Dataset(f) if os.path.exists(f) else None for f in files]
        self.data = dict(zip(names, datasets))

    def reshuffle(self):

        timeunit = 'hours since 2000-01-01 00:00'

        for version in self.versions:
            for mode in self.modes:
                files = find_files(os.path.join(self.root, mode, version), '.nc')
                dates = pd.DatetimeIndex([f[-24:-16] for f in files])
                meta = pd.Series(files, index=dates)
                meta = meta[self.date_range[0]:self.date_range[1]]

                fname = os.path.join(self.root, '_reshuffled', mode + '_' + version + '.nc')
                ds = Dataset(fname, mode='w')

                dates = date2num(meta.index.to_pydatetime(), timeunit).astype('int32')
                dimensions = OrderedDict([('time', dates), ('lat', self.lats), ('lon', self.lons)])

                chunksizes = []
                for key, values in dimensions.iteritems():
                    if key == 'time':
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
                ds.variables['time'].setncattr('units', timeunit)

                ds.createVariable('sm', 'float32',
                                  dimensions=dimensions.keys(),
                                  chunksizes=chunksizes,
                                  fill_value=-9999.,
                                  zlib=True)
                for i, f in enumerate(meta.values):
                    tmp_ds = Dataset(f)
                    ds['sm'][i, :, :] = tmp_ds.variables['sm'][0, :, :].data
                    tmp_ds.close()

                ds.close()


    def lonlat2colrow(self, lon, lat):

        londiff = np.abs(self.lons - lon)
        latdiff = np.abs(self.lats - lat)

        col = np.where((londiff - londiff.min()) < 0.0001)[0][0]
        row = np.where((latdiff - latdiff.min()) < 0.0001)[0][0]

        return col, row

    def read(self, lon, lat, mode='all', version='all'):

        col, row = self.lonlat2colrow(lon, lat)

        if mode == 'all':
            mode = self.modes
        if version == 'all':
            version = self.versions

        res = pd.DataFrame()
        for m in np.atleast_1d(mode):
            for v in np.atleast_1d(version):
                dates = num2date(self.data[m + '_' + v].variables['time'][:], units=self.data[m + '_' + v]['time'].units)
                ts = pd.Series(self.data[m + '_' + v].variables['sm'][:, row, col].data, index=dates)
                ts.replace(-9999, np.nan, inplace=True)
                if v == 'v02.2':
                    ts /= 1e2
                else:
                    if (m == 'PASSIVE') | (m == 'COMBINED'):
                        ts *= 1e2
                res[m + '_' + v] = ts
        return res

if __name__=='__main__':

    lat = 40.189451
    lon = -103.451243

    io = CCISM_io()

    sdate = '2010-01-01'
    edate = '2011-12-31'

    ts = io.read(lon, lat, mode='COMBINED', version='v04.4')[sdate:edate]
    ts.plot()

    plt.show()


    # io.reshuffle()
