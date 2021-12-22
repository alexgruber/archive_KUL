
import numpy as np
import pandas as pd

from pathlib import Path
from netCDF4 import Dataset, date2num
from collections import OrderedDict

from pyldas.grids import EASE2

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

    if not dir_out.exists():
        Path.mkdir(dir_out, parents=True)

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
    with ncfile_init(fout, dimensions, variables) as ds:

        for i, (f_slv, f_lnd) in enumerate(zip(*files)):

            print(f'{i} / {len(dates)}')

            with Dataset(f_slv) as slv, Dataset(f_lnd) as lnd:

                for var in vars_slv:
                    ds.variables[var][i, :, :] = np.mean(slv[var][:,i_lat, i_lon], axis=0)
                for var in vars_lnd:
                    ds.variables[var][i, :, :] = np.mean(lnd[var][:,i_lat, i_lon], axis=0)


def reformat_SMOS_IC():

    # root = Path('/Users/u0116961/data_sets/SMOS_IC')
    root = Path('/staging/leuven/stg_00024/l_data/obs_satellite/SMOS_IC/v2/ASC')

    # dir_out = Path('/Users/u0116961/data_sets/SMOS_IC')
    dir_out = Path('/staging/leuven/stg_00024/OUTPUT/alexg/data_sets/SMOS_IC')
    fout = dir_out / 'SMOS_IC_images.nc'

    if not dir_out.exists():
        Path.mkdir(dir_out, parents=True)

    variables = ['Optical_Thickness_Nad', 'Optical_Thickness_Nad_StdError', 'Soil_Moisture','Soil_Moisture_StdError', 'RMSE', 'Scene_Flags']
    var_names = ['VOD', 'VOD_StdErr', 'SM', 'SM_StdErr', 'RMSE', 'Flags']

    files = np.array(sorted(root.glob(f'**/SM_RE06_*.nc')))
    dates = pd.DatetimeIndex([f.name.split('_')[4][:8] for f in files])

    lats = Dataset(files[0])['lat'][:].data
    lons = Dataset(files[0])['lon'][:].data
    latmin, lonmin, latmax, lonmax = get_roi()
    i_lat = np.where((lats>=latmin)&(lats<=latmax))[0]
    i_lon = np.where((lons>=lonmin)&(lons<=lonmax))[0]
    lats = lats[i_lat]
    lons = lons[i_lon]

    dimensions = OrderedDict([('time', dates), ('lat', lats), ('lon', lons)])
    with ncfile_init(fout, dimensions, var_names) as ds:

        for i, file in enumerate(files):

            print(f'{i} / {len(dates)}')

            with Dataset(file) as smos:

                for var, name in zip(variables, var_names):
                    ds.variables[name][i, :, :] = smos[var][i_lat, i_lon]

def reformat_COPERNICUS_LAI():

    # root = Path('/Users/u0116961/data_sets/COPERNICUS_LAI')
    root = Path('/staging/leuven/stg_00024/l_data/obs_satellite/COPERNICUS_LAI')

    # dir_out = Path('/Users/u0116961/data_sets/COPERNICUS_LAI')
    dir_out = Path('/staging/leuven/stg_00024/OUTPUT/alexg/data_sets/COPERNICUS_LAI')
    fout = dir_out / 'COPERNICUS_LAI_images.nc'

    if not dir_out.exists():
        Path.mkdir(dir_out, parents=True)

    variables = ['LAI']

    files1 = np.array(sorted(root.glob(f'**/c_gls_LAI_*VGT*.nc')))
    files2 = np.array(sorted(root.glob(f'**/c_gls_LAI-RT6_*PROBAV*.nc')))

    dates1 = [f.name.split('_')[3][:8] for f in files1]
    dates2 = [f.name.split('_')[3][:8] for f in files2]

    files = np.hstack((files1,files2))
    dates = pd.DatetimeIndex(np.hstack((dates1,dates2)))

    # COPERNICUS 1km grid
    lats = Dataset(files[0])['lat'][:].data
    lons = Dataset(files[0])['lon'][:].data
    latmin, lonmin, latmax, lonmax = get_roi()
    i_lat_1km = np.where((lats>=latmin)&(lats<=latmax))[0]
    i_lon_1km = np.where((lons>=lonmin)&(lons<=lonmax))[0]
    lats_1km = lats[i_lat_1km]
    lons_1km = lons[i_lon_1km]

    # EASE25 grid
    grid = EASE2(gtype='M25')
    lats, lons = grid.ease_lats, grid.ease_lons
    i_lat_25km = np.where((lats >= latmin) & (lats <= latmax))[0]
    i_lon_25km = np.where((lons >= lonmin) & (lons <= lonmax))[0]
    lats_25km = lats[i_lat_25km]
    lons_25km = lons[i_lon_25km]

    i_lat, i_lon = [], []
    for lat in lats_1km:
        i_lat += [np.argmin(np.abs(lats_25km-lat))]
    i_lat = np.array(i_lat)
    for lon in lons_1km:
        i_lon += [np.argmin(np.abs(lons_25km-lon))]
    i_lon = np.array(i_lon)
    # i_lons, i_lats = np.meshgrid(i_lon, i_lat)

    dimensions = OrderedDict([('time', dates), ('lat', lats_25km), ('lon', lons_25km)])
    with ncfile_init(fout, dimensions, variables) as ds:

        for i, file in enumerate(files):

            print(f'{i} / {len(dates)}')

            with Dataset(file) as lai:
                for var in variables:

                    for i_lon_25 in range(len(lons_25km)):
                        for i_lat_25 in range(len(lats_25km)):
                            i_lons_cell = np.where(i_lon==i_lon_25)
                            i_lats_cell = np.where(i_lat==i_lat_25)
                            tmp_qflag = lai['QFLAG'][0, i_lat_1km[i_lats_cell], i_lon_1km[i_lons_cell]]
                            tmp_lai = lai[var][0, i_lat_1km[i_lats_cell], i_lon_1km[i_lons_cell]]
                            tmp_lai[tmp_qflag > 0] = 255.
                            tmp_lai = np.ma.masked_where(tmp_qflag > 0, tmp_lai)
                            tmp_lai.set_fill_value(255.)
                            ds.variables[var][i, i_lat_25, i_lon_25] = np.ma.mean(tmp_lai)


if __name__=='__main__':

    # reformat_MERRA2()
    # reformat_SMOS_IC()
    # reformat_COPERNICUS_LAI()
    pass

