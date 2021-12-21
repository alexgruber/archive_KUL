
import numpy as np
import pandas as pd

from pathlib import Path
from netCDF4 import Dataset, date2num
from collections import OrderedDict


def get_roi():
    latmin= -56.
    lonmin= -82
    latmax= 13.
    lonmax= -34

    return latmin, lonmin, latmax, lonmax


def ncfile_init(fname, dimensions, variables):
    """"
    Method to initialize dimensions/variables of a image-chunked netCDF file

    Parameters
    ----------
    fname : str
        Filename of the netCDF file to be created
    dimensions : dict
        Dictionary containing the dimension names and values
    variables : list
        list of variables to be created with the specified dimensions

    Returns
    -------
    ds : fileid
        File ID of the created netCDF file

    """

    ds = Dataset(fname, mode='w')
    timeunit = 'hours since 1900-01-01 00:00'

    # Initialize dimensions
    chunksizes = []
    for dim in dimensions:

        # convert pandas Datetime Index to netCDF-understandable numeric format
        if dim == 'time':
            dimensions[dim] = date2num(dimensions[dim].to_pydatetime(), timeunit).astype('int32')

        # Files are per default image chunked
        if dim in ['lon','lat']:
            chunksize = len(dimensions[dim])
        else:
            chunksize = 1
        chunksizes.append(chunksize)

        dtype = dimensions[dim].dtype
        ds.createDimension(dim, len(dimensions[dim]))
        ds.createVariable(dim,dtype,
                          dimensions=(dim,),
                          chunksizes=(chunksize,),
                          zlib=True)
        ds.variables[dim][:] = dimensions[dim]

    # Coordinate attributes following CF-conventions

    if 'time' in dimensions:
        ds.variables['time'].setncatts({'long_name': 'time',
                                        'units': timeunit})
    ds.variables['lon'].setncatts({'long_name': 'longitude',
                                   'units':'degrees_east'})
    ds.variables['lat'].setncatts({'long_name': 'latitude',
                                    'units':'degrees_north'})

    # Initialize variables
    for var in variables:
        ds.createVariable(var, 'float32',
                          dimensions=list(dimensions.keys()),
                          chunksizes=chunksizes,
                          fill_value=-9999.,
                          zlib=True)

    return ds

def reformat_MERRA2():

    # root = Path('/Users/u0116961/data_sets/MERRA2')             )
    root = Path('/staging/leuven/stg_00024/input/met_forcing/MERRA2_land_forcing/MERRA2_400/diag')

    # dir_out = Path('/Users/u0116961/data_sets/MERRA2')
    dir_out = Path('/staging/leuven/stg_00024/OUTPUT/alexg/data_sets/MERRA2')
    fout = dir_out / 'MERRA2_images.nc'

    names = ['tavg1_2d_slv_Nx', 'tavg1_2d_lnd_Nx']
    vars_slv = ['T2M']
    vars_lnd = ['PRECTOTLAND','LWLAND', 'SWLAND', 'SFMC', 'RZMC', 'TSOIL1', 'SNOMAS']

    files = [np.array(sorted(root.glob(f'**/*{name}*'))) for name in names]
    dates = pd.DatetimeIndex([f.name.split('.')[-2] for f in files[0]])

    lats = Dataset(files[0][0])['lat'][:].data
    lons = Dataset(files[0][0])['lon'][:].data
    latmin, lonmin, latmax, lonmax = get_roi()
    i_lat = np.where((lats>=latmin)&(lats<=latmax))[0]
    i_lon = np.where((lons>=lonmin)&(lons<=lonmax))[0]
    lats = lats[i_lat]
    lons = lons[i_lon]

    dimensions = OrderedDict([('time', dates), ('lat', lats), ('lon', lons)])
    variables = vars_slv + vars_lnd

    ds = ncfile_init(fout, dimensions, variables)

    for i, (f_slv, f_lnd) in enumerate(zip(*files)):

        print(f'{i} / {len(dates)}')

        with Dataset(f_slv) as slv, Dataset(f_lnd) as lnd:

            for var in vars_slv:
                ds.variables[var][i, :, :] = np.mean(slv[var][:,i_lat, i_lon], axis=0)
            for var in vars_lnd:
                ds.variables[var][i, :, :] = np.mean(lnd[var][:,i_lat, i_lon], axis=0)

    ds.close()

if __name__=='__main__':

    reformat_MERRA2()

'''
import sys
sys.path.append('/data/leuven/320/vsc32046/python/')
from myprojects.publications.deforestation_paper.preprocessing import reformat_MERRA2

'''
