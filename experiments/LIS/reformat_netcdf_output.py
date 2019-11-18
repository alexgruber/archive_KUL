
import os

import pandas as pd

from pathlib import Path
from netCDF4 import Dataset, date2num

def main():

    path = Path('/data_sets/LIS/NoahMP_belgium/exp1_OL/OUT_OL_Belgium/SURFACEMODEL')
    outfile_img = path / 'images.nc'
    outfile_ts = path / 'timeseries.nc'

    files = sorted(path.glob('**/*.d01.nc'))

    timeunit = 'hours since 2000-01-01 00:00'
    dates = date2num(pd.to_datetime([f.name[-19:-7] for f in files], format='%Y%m%d%H%M').to_pydatetime(), timeunit).astype('int32')

    with Dataset(outfile, mode='w') as res:

        for i,file in enumerate(sorted(files)):
            print('%i / %i' % (i+1, len(files)))

            with Dataset(file) as ds:

                if i == 0:
                    res.createDimension('lat', ds['lat'].shape[0])
                    res.createDimension('lon', ds['lon'].shape[1])
                    res.createDimension('time', len(dates))

                    res.createVariable('lat', ds['lat'].dtype, dimensions=('lat','lon'), chunksizes=ds['lat'].shape, zlib=True)
                    res.createVariable('lon', ds['lon'].dtype, dimensions=('lat','lon'), chunksizes=ds['lon'].shape, zlib=True)
                    res.createVariable('time', dates.dtype, dimensions=('time',), chunksizes=(1,), zlib=True)
                    res.variables['lat'][:,:] = ds['lat'][:,:]
                    res.variables['lon'][:,:] = ds['lon'][:,:]
                    res.variables['time'][:] = dates

                    # Coordinate attributes following CF-conventions
                    res.variables['time'].setncatts({'long_name': 'time', 'units': timeunit})
                    res.variables['lon'].setncatts({'long_name': 'longitude', 'units':'degrees_east'})
                    res.variables['lat'].setncatts({'long_name': 'latitude', 'units':'degrees_north'})

                    res.createVariable('SoilMoisture', 'float32',
                                       dimensions=('time', 'lat', 'lon'), chunksizes=(1,) + ds['lat'].shape, zlib=True)

                res.variables['SoilMoisture'][i, :, :] = ds['SoilMoist_inst'][0,:,:]

    cmdBase = 'ncks -4 -L 4 --cnk_dmn time,%i --cnk_dmn lat,1 --cnk_dmn lon,1 ' % len(dates)
    cmd = ' '.join([cmdBase, str(outfile_img), str(outfile_ts)])
    os.system(cmd)

if __name__=='__main__':
    main()

