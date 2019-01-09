
import os
import h5py

import numpy as np
import pandas as pd

'''
activate dev
python
import sys
sys.path.append(r'I:\python')
from validation_good_practice.data_readers.smap import reshuffle_smap
reshuffle_smap()
'''

def reshuffle_smap():

    # generate idx. array to map ease col/row to gpi
    n_row = 406; n_col = 964
    idx_arr = np.arange(n_row*n_col,dtype='int64').reshape((n_row,n_col))

    # get a list of all CONUS gpis
    ease_gpis = pd.read_csv(r"D:\data_sets\EASE2_grid\grid_lut.csv", index_col=0).index.values

    # Collect orbit file list and extract date info from file name
    fdir = r'D:\data_sets\SMAP_L2\raw'
    files = os.listdir(fdir)
    dates = pd.to_datetime([f[21:36] for f in files]).round('min')

    # Array with ALL possible dates and ALL CONUS gpis
    res_arr = np.full((len(dates),len(ease_gpis)), np.nan)

    # Fill in result array from orbit files
    for i, f in enumerate(files):
        print "%i / %i" % (i, len(files))

        tmp = h5py.File(os.path.join(fdir,f))
        row = tmp['Soil_Moisture_Retrieval_Data']['EASE_row_index'][:]
        col = tmp['Soil_Moisture_Retrieval_Data']['EASE_column_index'][:]
        idx = idx_arr[row,col]

        # Check for valid data within orbit files
        for res_ind, gpi in enumerate(ease_gpis):
            sm_ind = np.where(idx == gpi)[0]
            if len(sm_ind) > 0:
                qf = tmp['Soil_Moisture_Retrieval_Data']['retrieval_qual_flag'][sm_ind[0]]
                if (qf == 0)|(qf == 8):
                    res_arr[i, res_ind] = tmp['Soil_Moisture_Retrieval_Data']['soil_moisture'][sm_ind[0]]

        tmp.close()

    # Write out valid time series of all CONIS GPIS into separate .csv files
    dir_out = r'D:\data_sets\SMAP_L2\timeseries'
    for i, gpi in enumerate(ease_gpis):
        Ser = pd.Series(res_arr[:,i],index=dates).dropna()
        if len(Ser) > 0:
            Ser = Ser.groupby(Ser.index).last()
            fname = os.path.join(dir_out, '%i.csv' % gpi)
            Ser.to_csv(fname,float_format='%.4f')

if __name__=='__main__':
    reshuffle_smap()
