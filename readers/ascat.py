

import os
import platform

import numpy as np
import pandas as pd

from pathlib import Path

from netCDF4 import Dataset, num2date

from pyldas.interface import LDAS_io

class HSAF_io(object):

    def __init__(self, root=None, version='h115', ext='h116'):

        if root is None:
            if platform.system() == 'Linux':
                self.root = Path('/staging/leuven/stg_00024/OUTPUT/alexg/data_sets/ASCAT')
            else:
                self.root = Path('~/data_sets/ASCAT').expanduser()
        else:
            self.root = Path(root)

        self.data_path = os.path.join(self.root, version)
        self.version = version.upper()

        self.grid = Dataset(self.root / 'warp5_grid' / 'TUW_WARP5_grid_info_2_2.nc')
        self.gpis = self.grid['gpi'][:][self.grid['land_flag'][:]==1]
        self.cells = self.grid['cell'][:][self.grid['land_flag'][:]==1]

        self.loaded_cell = None
        self.fid = None

        if ext is not None:
            self.ext = HSAF_io(root=self.root, version=ext, ext=None)
        else:
            self.ext = None

        # self.frozen_snow_prob = xr.open_dataset(os.path.join(self.root, 'static_layer', 'frozen_snow_probability.nc'))
        # quite slow to read!

    def latlon2gpi(self, lat, lon):
        return np.argmin((self.grid['lat'][:] - lat)**2 + (self.grid['lon'][:] - lon)**2)

    def load(self, cell):

        fname = os.path.join(self.data_path, self.version + '_%04i.nc' % cell)
        if not os.path.exists(fname):
            print('File not found: ' + fname)
            return False

        try:
            if self.fid is not None:
                self.fid.close()
            self.fid = Dataset(fname)
        except:
            print('Corrupted cell: %i' % cell)
            return False

        self.loaded_cell = cell

        return True

    def read(self, *args, resample_time=True, var='sm'):

        if len(args) == 1:
            gpi = int(args[0])
        else:
            gpi = self.latlon2gpi(*args)

        if not gpi in self.gpis:
            print('GPI not found')
            return None

        cell = self.cells[self.gpis==gpi][0]

        if self.loaded_cell != cell:
            loaded = self.load(cell)
            if loaded is False:
                return None

        loc = np.where(self.fid['location_id'][:] == gpi)[0][0]
        start = self.fid['row_size'][0:loc].sum()
        end = start + self.fid['row_size'][loc]

        corr_flag = self.fid['corr_flag'][start:end]
        conf_flag = self.fid['conf_flag'][start:end]
        proc_flag = self.fid['proc_flag'][start:end]
        ssf = self.fid['ssf'][start:end]
        ind_valid = ((corr_flag==0)|(corr_flag==4)) & (conf_flag == 0) & (proc_flag == 0) & (ssf == 1)

        if len(np.where(ind_valid)[0]) == 0:
            print('No valid ASCAT data for gpi %i' % gpi)
            return None

        sm = self.fid[var][start:end][ind_valid]

        if resample_time is True:
            time = num2date(self.fid['time'][start:end][ind_valid].round(), units=self.fid['time'].units,
                            only_use_python_datetimes=True, only_use_cftime_datetimes=False)
        else:
            time = num2date(self.fid['time'][start:end][ind_valid], units=self.fid['time'].units,
                            only_use_python_datetimes=True, only_use_cftime_datetimes=False)

        ts = pd.Series(sm, index=time)
        ts = ts.groupby(ts.index).mean()

        if self.ext is not None:
            ts_ext = self.ext.read(gpi, resample_time=resample_time)
            ts = pd.concat((ts,ts_ext))

        ts.name = 'ascat'
        return ts

    def close(self):
        if self.fid is not None:
            self.fid.close()
        if self.ext is not None:
            self.ext.close()
        if self.grid is not None:
            self.grid.close()


def append_ease_gpis():

    gpi_list = pd.read_csv(r"D:\data_sets\ASCAT\warp5_grid\pointlist_warp_conus.csv",index_col=0)

    gpi_list['ease_col'] = 0
    gpi_list['ease_row'] = 0

    LDAS = LDAS_io(exp='US_M36_SMOS40_noDA_cal_scaled')

    i = 0
    for idx, info in gpi_list.iterrows():
        i += 1
        print('%i / %i' % (i, len(gpi_list)))

        col, row = LDAS.grid.lonlat2colrow(gpi_list.loc[idx, 'lon'], gpi_list.loc[idx, 'lat'], domain=True)

        gpi_list.loc[idx,'ease_col'] = col
        gpi_list.loc[idx,'ease_row'] = row

    gpi_list.to_csv(r"D:\data_sets\ASCAT\warp5_grid\pointlist_warp_conus_w_ease_colrow.csv")


# if __name__=='__main__':
#     append_ease_gpis()























