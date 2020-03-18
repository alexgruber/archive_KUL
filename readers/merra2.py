

import numpy as np
import pandas as pd

from pathlib import Path
from collections import OrderedDict

from netCDF4 import Dataset, date2num, num2date


def ncfile_init(fname, dimensions, variables):

    ds = Dataset(fname, mode='w')
    timeunit = 'days since 1900-01-01 00:00'

    # Initialize dimensions
    chunksizes = []
    for dim in dimensions:

        # convert pandas Datetime Index to netCDF-understandable numeric format
        if dim == 'time':
            dimensions[dim] = date2num(dimensions[dim].to_pydatetime(), timeunit).astype('int32')

        # Files are per default image chunked
        if dim in ['lon', 'lat']:
            chunksize = len(dimensions[dim])
        else:
            chunksize = 1
        chunksizes.append(chunksize)

        dtype = dimensions[dim].dtype
        ds.createDimension(dim, len(dimensions[dim]))
        ds.createVariable(dim, dtype,
                          dimensions=(dim,),
                          chunksizes=(chunksize,),
                          zlib=True)
        ds.variables[dim][:] = dimensions[dim]

    # Coordinate attributes following CF-conventions
    if 'time' in dimensions:
        ds.variables['time'].setncatts({'long_name': 'time',
                                        'units': timeunit})
    ds.variables['lon'].setncatts({'long_name': 'longitude',
                                   'units': 'degrees_east'})
    ds.variables['lat'].setncatts({'long_name': 'latitude',
                                   'units': 'degrees_north'})

    # Initialize variables
    for var in variables:
        ds.createVariable(var, 'float32',
                          dimensions=list(dimensions.keys()),
                          chunksizes=chunksizes,
                          fill_value=-9999.,
                          zlib=True)

    return ds


def reformat():

    fname_out = Path('/staging/leuven/stg_00024/OUTPUT/alexg/MERRA2.nc4')
    root = Path('/staging/leuven/stg_00024/input/met_forcing/MERRA2_land_forcing')

    # fname_out = Path('/Users/u0116961/work/test.nc4')
    # root = Path('/Users/u0116961/work')

    variables = ['SFMC', 'RZMC', 'PRMC', 'TSOIL1', 'TSURF', 'SWLAND', 'LWLAND']
    latmin, latmax, lonmin, lonmax = 24., 51., -128., -64.
    dates = pd.date_range('2010-01-01', '2018-12-31')

    with Dataset(list(root.glob('**/*_lnd_*' + dates[0].strftime('%Y%m%d') + '.nc4'))[0]) as ds:
        lats = ds['lat'][:]
        lons = ds['lon'][:]

    ind_lats = np.where((lats >= latmin) & (lats <= latmax))[0]
    ind_lons = np.where((lons >= lonmin) & (lons <= lonmax))[0]
    lats = lats[ind_lats].data
    lons = lons[ind_lons].data

    data = np.full((7, len(dates), len(lats), len(lons)), -9999.)

    for i, date in enumerate(dates):
        try:
            with Dataset(list(root.glob('**/*_lnd_*' + date.strftime('%Y%m%d') + '.nc4'))[0]) as ds:
                for j, var in enumerate(variables):
                    data[j, i, :, :] = ds[var][:, ind_lats, ind_lons].mean(axis=0)

        except:
            print('No data on ', date)

    dimensions = OrderedDict([('time', dates), ('lat', lats), ('lon', lons)])

    with ncfile_init(fname_out, dimensions, variables) as ds:
        for i, var in enumerate(variables):
            ds[var][:,:,:] = data[i,:,:,:]


if __name__=='__main__':
    reformat()

