
import pickle
import platform

import numpy as np
import pandas as pd

from pathlib import Path


from ismn.interface import ISMN_Interface
from pyldas.interface import GEOSldas_io

class ISMN_io(object):

    def __init__(self, path=None, col_offs=0, row_offs=0):

        self.col_offs = col_offs
        self.row_offs = row_offs

        if path is None:
            self.root = Path('~/data_sets/ISMN/CONUS_20150401_20200401_downloaded_20210920').expanduser()
        else:
            self.root = Path(path)

        self.io = ISMN_Interface(self.root)

        self.list_file = self.root / 'station_list.csv'
        if not self.list_file.exists():
            print('Station list does not exist.')
            self.generate_station_list()
        else:
            self.list = pd.read_csv(self.list_file, index_col=0)

    def generate_station_list(self):

        tmplist = self.io.metadata[[('network','val'),('station','val'),('latitude','val'),('longitude','val')]]
        tmplist.columns = tmplist.columns.droplevel('key')
        tmplist.columns.name = None

        tmplist = tmplist.iloc[np.unique(tmplist.network+tmplist.station, return_index=True)[1]]
        tmplist.index = np.arange(len(tmplist))

        grid = GEOSldas_io().grid
        vfindcolrow = np.vectorize(grid.lonlat2colrow)
        col, row = vfindcolrow(tmplist.longitude.values, tmplist.latitude.values)
        tmplist['ease_col'] = col
        tmplist['ease_row'] = row

        tmplist['ease_col'] -= self.col_offs
        tmplist['ease_row'] -= self.row_offs

        tmplist.to_csv(self.list_file)
        self.list = tmplist


    def read(self, network, station, surface_depth=0.1, surface_only=False):

        names = ['sm_surface']
        depths = [[0, surface_depth]]

        if not surface_only:
            names += ['sm_rootzone', 'sm_profile']
            depths += [[surface_depth,1],[0,1]]

        tss = []
        for name, depth in zip(names,depths):
            tmp_tss = []
            for data in self.io[network][station].iter_sensors(variable='soil_moisture', depth=depth):
                ts = data.read_data()
                tmp_tss += [ts[ts['soil_moisture_flag'] == 'G']['soil_moisture']]
            if len(tmp_tss) > 0:
                ts = pd.concat(tmp_tss, axis=1)
                ts = ts.mean(axis=1)
                ts.name = name
            else:
                ts = pd.Series(name=name)
            tss += [ts]

        res = pd.concat(tss, axis=1)

        if len(res) > 0:
            res.index = pd.DatetimeIndex(res.index)
            res = res.resample('6h').mean()

        return res



    def iter_stations(self, surface_depth=0.1, surface_only=True):

        for idx, station in self.list.iterrows():
            yield station, self.read(station.network, station.station, surface_depth=surface_depth)


if __name__=='__main__':

    io = ISMN_io()
    io.generate_station_list()


