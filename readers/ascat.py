

import os
import platform

import numpy as np
import pandas as pd

from netCDF4 import Dataset, num2date

class HSAF_io(object):

    def __init__(self, root=None, version='H113'):

        if root is None:
            if platform.system() == 'Windows':
                root = os.path.join('D:','data_sets', 'ASCAT')
            else:
                root = os.path.join('/', 'data', 'leuven', '320', 'vsc32046', 'data_sets', 'ASCAT')

        self.data_path = os.path.join(root, version)

        grid = Dataset(os.path.join(root, 'warp5_grid', 'TUW_WARP5_grid_info_2_2.nc'))
        self.gpis = grid['gpi'][:][grid['land_flag'][:]==1]
        self.cells = grid['cell'][:][grid['land_flag'][:]==1]
        grid.close()

        self.loaded_cell = None
        self.fid = None

        # self.frozen_snow_prob = xr.open_dataset(os.path.join(self.root, 'static_layer', 'frozen_snow_probability.nc'))
        # quite slow to read!

    def load(self, cell):

        fname = os.path.join(self.data_path, 'H113_%04i.nc' % cell)
        if not os.path.exists(fname):
            print 'File not found: ' + fname
            return False

        try:
            if self.fid is not None:
                self.fid.close()
            self.fid = Dataset(fname)
        except:
            print 'Corrupted cell: %i' % cell
            return False

        self.loaded_cell = cell

        return True

    def read(self, gpi):

        if not gpi in self.gpis:
            print 'GPI not found'
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
            print 'No valid data for gpi %i' % gpi
            return None

        sm = self.fid['sm'][start:end][ind_valid]
        time = num2date(self.fid['time'][start:end][ind_valid].round(),
                        units=self.fid['time'].units)

        ts = pd.Series(sm, index=time)

        return ts.groupby(ts.index).mean()

    def close(self):
        if self.fid is not None:
            self.fid.close()
