
import os

import numpy as np
import pandas as pd

from pathlib import Path

from netCDF4 import Dataset, num2date, date2num

from collections import OrderedDict


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

def reformat(path_in, path_out):

    path_in = Path(path_in)

    fname_img = Path(path_out) / 'DMP_COPERNICUS_images.nc'
    fname_ts = Path(path_out) / 'DMP_COPERNICUS_timeseries.nc'

    if fname_img.exists():
        fname_img.unlink()
    if fname_ts.exists():
        fname_ts.unlink()

    variables = ['DMP', 'QFLAG']

    latmin = 35
    latmax = 45
    lonmin = -100
    lonmax = -90

    files = np.sort(list(root.glob('**/*.nc')))
    dates = pd.DatetimeIndex([f.name.split('_')[-4] for f in files])

    with Dataset(files[0]) as ds:
        lats = ds['lat'][:].data
        lons = ds['lon'][:].data
    ind_lat = np.where((lats >= latmin) & (lats <= latmax))[0]
    ind_lon = np.where((lons >= lonmin) & (lons <= lonmax))[0]
    lats = lats[ind_lat]
    lons = lons[ind_lon]

    dimensions = OrderedDict([('time', dates), ('lat', lats), ('lon', lons)])

    with ncfile_init(fname_img, dimensions, variables) as fout:
        for i,f in enumerate(files):
            print('%i / %i' % (i+1, len(files)))
            with Dataset(f) as ds:
                for var in variables:
                    fout[var][i,:,:] = ds[var][:,ind_lat,ind_lon]


    cmdBase = 'ncks -4 -L 4 --cnk_dmn time,%i --cnk_dmn lat,1 --cnk_dmn lon,1 ' % len(dates)
    cmd = ' '.join([cmdBase, str(fname_img), str(fname_ts)])
    os.system(cmd)

if __name__=='__main__':

    path_in = '/data_sets/COPERNICUS_DMP'
    path_out = path_in

    reformat('/data_sets/COPERNICUS_DMP')