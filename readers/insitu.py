
import os
import pickle
import platform

import numpy as np
import pandas as pd

from ismn.readers import read_data
from ismn.metadata_collector import collect_from_folder

from pyldas.interface import LDAS_io

class ISMN_io(object):

    def __init__(self, path=None, col_offs=0, row_offs=0):

        self.col_offs = col_offs
        self.row_offs = row_offs

        if path is None:

            if platform.system() == 'Windows':
                self.root = os.path.join('D:', 'data_sets', 'ISMN_global_20071003_20141231')
            else:
                self.root = os.path.join('/', 'data_sets', 'ISMN_global_20071003_20141231')
        else:
            self.root = path

        self.meta_file = os.path.join(self.root, 'meta2.bin')
        if not os.path.exists(self.meta_file):
            print('Meta file does not exist.')
            self.meta = None
        else:
            f = open(self.meta_file, 'rb')
            self.meta = pickle.load(f)
            f.close()

        self.list_file = os.path.join(self.root, 'station_list.csv')
        if not os.path.exists(self.list_file):
            print('Station list does not exist.')
        else:
            self.list = pd.read_csv(self.list_file, index_col=0)
            self.list['ease_col'] -= self.col_offs
            self.list['ease_row'] -= self.row_offs

    def generate_meta_file(self):

        self.meta = collect_from_folder(self.root)

        f = open(self.meta_file, 'wb')
        pickle.dump(self.meta, f)
        f.close()

    def generate_station_list(self):

        if self.meta is None:
            print('Generating meta file...')
            self.generate_meta_file()

        unique, ind = np.unique(self.meta['network'] + self.meta['station'], return_index=True)

        self.list = pd.DataFrame({'network': self.meta[ind]['network'], 'station': self.meta[ind]['station'],
                                  'lat': self.meta[ind]['latitude'], 'lon': self.meta[ind]['longitude']})

        grid = LDAS_io().grid
        vfindcolrow = np.vectorize(grid.lonlat2colrow)
        col, row = vfindcolrow(self.list.lon.values, self.list.lat.values)
        self.list['ease_col'] = col
        self.list['ease_row'] = row

        self.list['ease_col'] -= self.col_offs
        self.list['ease_row'] -= self.row_offs
        self.list.to_csv(self.list_file)

    def read_first_surface_layer(self, network, station, var='soil moisture', surf_depth=0.1):

        tmp_meta = self.meta[(self.meta['network'] == network) & (self.meta['station'] == station)]
        tmp_meta = tmp_meta[tmp_meta['variable'] == var]
        if len(tmp_meta) == 0:
            return None
        tmp_meta = tmp_meta[tmp_meta['depth_from'] == min(tmp_meta['depth_from'])]
        if len(tmp_meta) > 1:
            tmp_meta = tmp_meta[tmp_meta['depth_to'] == min(tmp_meta['depth_to'])]
        if len(tmp_meta) > 1:
            tmp_meta = tmp_meta[0]
        if tmp_meta['depth_from'] > surf_depth:
            return None

        try:
            if len(tmp_meta['filename']) == 1:
                tmp_data = read_data(tmp_meta['filename'][0]).data
            else:
                tmp_data = read_data(tmp_meta['filename']).data
        except:
            print('Data could not be read from ' + tmp_meta['filename'])
            return None

        tmp_data = pd.Series(tmp_data[tmp_data[var + '_flag'] == 'G'][var])
        if len(tmp_data) < 10:
            return None
        tmp_data.name = 'insitu'

        return tmp_data

    def read(self, network, station, var='soil moisture', surf_depth=0.05):

        meta = self.meta[(self.meta['network'] == network) & \
                         (self.meta['station'] == station) & \
                         (self.meta['variable'] == var)]

        surf_files = meta[meta['depth_to'] <= surf_depth]['filename']
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

    def iter_stations(self, surf_depth=0.1, surface_only=True):

        for idx,station in self.list.iterrows():
            if surface_only is True:
                yield station, self.read_first_surface_layer(station.network, station.station, surf_depth=surf_depth)
            else:
                yield station, self.read(station.network, station.station, surf_depth=surf_depth)


# if __name__=='__main__':
#
#     io = ISMN_io()
#     io.generate_station_list()
#

