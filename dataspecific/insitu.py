
import os
import pickle

import numpy as np
import pandas as pd

import ismn.readers as ISMN

from ismn.metadata_collector import collect_from_folder

from pyldas.grids import EASE2

# from pytesmo.time_series.anomaly import calc_anomaly

def read_first_layer_soil_moisture(meta,network,station,short=False):

    var = 'sm' if short==True else 'soil moisture'

    tmp_meta = meta[(meta['network']==network)&(meta['station']==station)]
    tmp_meta = tmp_meta[tmp_meta['variable']==var]
    if len(tmp_meta) ==0:
        return None
    tmp_meta = tmp_meta[tmp_meta['depth_from']==min(tmp_meta['depth_from'])]
    if len(tmp_meta) > 1:
        tmp_meta = tmp_meta[tmp_meta['depth_to']==min(tmp_meta['depth_to'])]
    if len(tmp_meta) > 1:
        tmp_meta = tmp_meta[0]
    if tmp_meta['depth_from'] > 0.1:
        return None

    if len(tmp_meta['filename'])==1:
        tmp_data = ISMN.read_data(tmp_meta['filename'][0]).data
    else:
        tmp_data = ISMN.read_data(tmp_meta['filename']).data

    tmp_data = pd.Series(tmp_data[tmp_data['soil moisture_flag']=='G']['soil moisture'])
    if len(tmp_data) < 10:
        return None

    tmp_data.name = 'insitu'

    # if anomaly==True:
    #
    #     tmp_data = calc_anomaly(tmp_data[(tmp_data.index.hour==0)|\
    #                                     (tmp_data.index.hour==6)|\
    #                                     (tmp_data.index.hour==12)|\
    #                                     (tmp_data.index.hour==18)],return_clim=True)
    #     tmp_data['insitu_bulk'] = tmp_data['insitu_clim'] + tmp_data['insitu_anom']

    return tmp_data


def generate_meta_file(path):

    fname = os.path.join(path,'meta.bin')
    meta = collect_from_folder(path)

    f = open(fname,'wb')
    pickle.dump(meta,f)
    f.close()

    return meta


def generate_station_list_ismn(path):

    grid_era = EASE2()

    meta_file = os.path.join(path,'meta.bin')
    if not os.path.exists(meta_file):
        meta = generate_meta_file(path)
    else:
        f = open(meta_file,'r')
        meta = pickle.load(f)
        f.close()

    unique, ind = np.unique(meta['network']+meta['station'],return_index=True)

    station_list = pd.DataFrame({'network': meta[ind]['network'], 'station': meta[ind]['station'], \
                                 'lat': meta[ind]['latitude'], 'lon' :meta[ind]['longitude']})

    vfindcolrow = np.vectorize(grid_era.lonlat2colrow)

    col,row = vfindcolrow(station_list.lon.values,station_list.lat.values)
    station_list['ease_col'] = col
    station_list['ease_row'] = row

    station_list.to_csv(path+'\station_list.csv')



if __name__=='__main__':

    path = r"C:\Users\u0116961\Documents\ISMN_global_20100101_20171231"
    # generate_meta_file(path)
    generate_station_list_ismn(path)

#     # generiert meta daten bank und station list mit grid point mapping
#     path = r'D:\data_sets\ISMN\2017-04-19_global_2007-2011' + '\\'
#     generate_station_list_ismn(path)
#
#
#     # meta daten lesen (fuer leseroutine)
#     meta_file = r'D:\data_sets\ISMN\2015-04-21_global_2007-2011\meta.bin'
#     f = open(meta_file,'r')
#     meta = pickle.load(f)
#     f.close()
#
#     # stationsliste zum iterieren ueber netzwerke, stationen und grid punkte
#     station_list = pd.DataFrame.from_csv(r"D:\data_sets\ISMN\2017-04-19_global_2007-2011\station_list.csv")
#
#     # daten lesen
#     network = station_list.network.values[0]
#     station= station_list.station.values[0]
#     data = read_first_layer_soil_moisture(meta, network, station)


