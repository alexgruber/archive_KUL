
import numpy as np
import pandas as pd
from netCDF4 import Dataset

def read_grid():

    lats = np.arange(-89.875, 90, 0.25)[::-1]
    lons = np.arange(-179.875, 180, 0.25)
    lons, lats = np.meshgrid(lons, lats)
    gpis = np.arange(lons.size).reshape(lons.shape)

    return lats.flatten(), lons.flatten(), gpis.flatten()


def get_valid_gpis(latmin=-90., latmax=90., lonmin=-180., lonmax=180., wfrac=0.05):

    # mask permanent water and open water fraction
    ds1 = Dataset('/data_sets/GLEAM/v33b_forcing/2017/fracW_MOD44b_v52_025deg_2017.nc')
    ds2 = Dataset('/data_sets/GLEAM/gleam_v33b_static_ens.nc')
    wfrac_mask = ds1.variables['fracW'][0, :, :].transpose().flatten()
    static_mask = ds2.variables['mask_WAT'][:, :].transpose().flatten()
    gpis_valid = np.where((wfrac_mask < wfrac) & (static_mask == 0))[0]

    # clip region of interest
    lats, lons, _ = read_grid()
    lats = lats[gpis_valid]
    lons = lons[gpis_valid]

    ind_valid = np.where((lats>=latmin)&(lats<=latmax)&(lons>=lonmin)&(lons<=lonmax))

    return gpis_valid[ind_valid]
