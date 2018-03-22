
import os
import pickle

import numpy as np
import pandas as pd

from ismn.readers import read_data
from ismn.metadata_collector import collect_from_folder


from pyldas.grids import EASE2

class ISMN_io(object):

    def __init__(self, path=None, col_offs=0, row_offs=0):

        self.col_offs = col_offs
        self.row_offs = row_offs

        if path is None:
            self.root = r"D:\data_sets\ISMN_global_20100101_20171231"
        else:
            self.root = path

        self.meta_file = os.path.join(self.root, 'meta.bin')
        if not os.path.exists(self.meta_file):
            print 'Meta file does not exist.'
            self.meta = None
        else:
            f = open(self.meta_file, 'r')
            self.meta = pickle.load(f)
            f.close()

        self.list_file = os.path.join(self.root, 'station_list.csv')
        if not os.path.exists(self.list_file):
            print 'Station list does not exist.'
        else:
            self.list = pd.read_csv(self.list_file)
            self.list['ease_col'] -= self.col_offs
            self.list['ease_row'] -= self.row_offs

    def generate_meta_file(self):

        self.meta = collect_from_folder(self.root)

        f = open(self.meta_file, 'wb')
        pickle.dump(self.meta, f)
        f.close()

    def generate_station_list(self):

        if self.meta is None:
            print 'Generating meta file...'
            generate_meta_file(self.meta_file)

        unique, ind = np.unique(self.meta['network'] + self.meta['station'], return_index=True)

        self.list = pd.DataFrame({'network': self.meta[ind]['network'], 'station': self.meta[ind]['station'], \
                                  'lat': self.meta[ind]['latitude'], 'lon': self.meta[ind]['longitude']})

        grid = EASE2()
        vfindcolrow = np.vectorize(grid.lonlat2colrow)
        col, row = vfindcolrow(self.list.lon.values, self.list.lat.values)
        self.list['ease_col'] = col
        self.list['ease_row'] = row

        self.list.to_csv(self.list_file)
        self.list['ease_col'] -= self.col_offs
        self.list['ease_row'] -= self.row_offs


    def read(self, network, station, var='soil moisture'):

        meta = self.meta[(self.meta['network'] == network) & \
                         (self.meta['station'] == station) & \
                         (self.meta['variable'] == var)]

        surf_files = meta[meta['depth_to'] <= 0.05]['filename']
        root_files = meta[meta['depth_to'] <= 1.00]['filename']
        prof_files = meta['filename']

        surf = dict()
        for i,f in enumerate(surf_files):
            try:
                ts = read_data(f).data
                surf[i] = ts[ts[var + '_flag'] == 'G'][var].resample('3h',label='right',closed='right').mean().dropna()
            except:
                continue

        root = dict()
        for i,f in enumerate(root_files):
            try:
                ts = read_data(f).data
                root[i] = ts[ts[var + '_flag'] == 'G'][var].resample('3h',label='right',closed='right').mean().dropna()
            except:
                continue

        prof = dict()
        for i,f in enumerate(prof_files):
            try:
                ts = read_data(f).data
                prof[i] = ts[ts[var + '_flag'] == 'G'][var].resample('3h',label='right',closed='right').mean().dropna()
            except:
                continue

        return pd.DataFrame({'sm_surface':pd.DataFrame(surf).mean(axis=1),
                             'sm_rootzone':pd.DataFrame(root).mean(axis=1),
                             'sm_profile':pd.DataFrame(prof).mean(axis=1)})

    def iter_stations(self):

        for idx,station in self.list.iterrows():
            yield station, self.read(station.network, station.station)


# if __name__=='__main__':
#
#     io = ISMN_io()


