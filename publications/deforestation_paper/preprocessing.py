
import numpy as np
import pandas as pd
import rasterio as rio

from pathlib import Path
from netCDF4 import Dataset, date2num
from collections import OrderedDict

from pyldas.grids import EASE2

from myprojects.publications.deforestation_paper.interface import io

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
                    if var == 'PRECTOTLAND':
                        ds.variables[var][i, :, :] = np.sum(lnd[var][:,i_lat, i_lon], axis=0)
                    else:
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


def reformat_AGB():

    root = Path('/Users/u0116961/data_sets/AGB')
    fout = root / 'resampled' / 'AGB_25km.nc'

    maskfile = root.parent / 'EASE2_M25km.LOCImask_land50_coast0km.1388x584.bin'
    mask = np.fromfile(maskfile, dtype='byte').reshape(584,1388)

    # tile name indicators
    rows = ['N40','N00','S40']
    cols = ['W100','W060']

    # 45 000 pixels in each 40 x 40 deg box
    n_xy = 45000
    spc = 40 / n_xy

    # AGB grid
    agb = np.full((len(rows)*n_xy,len(cols)*n_xy), 2**16-1, dtype='uint16')
    err = agb.copy()
    lats_100m = (np.arange(agb.shape[0]) * spc - 80 + spc/2)[::-1]
    lons_100m = (np.arange(agb.shape[1]) * spc - 100 + spc/2)
    latmin, lonmin, latmax, lonmax = get_roi()
    i_lat_100m = np.where((lats_100m>=latmin)&(lats_100m<=latmax))[0]
    i_lon_100m = np.where((lons_100m>=lonmin)&(lons_100m<=lonmax))[0]
    lats_100m = lats_100m[i_lat_100m]
    lons_100m = lons_100m[i_lon_100m]


    # EASE25 grid
    grid = EASE2(gtype='M25')
    lats, lons = grid.ease_lats, grid.ease_lons
    i_lat_25km = np.where((lats >= latmin) & (lats <= latmax))[0]
    i_lon_25km = np.where((lons >= lonmin) & (lons <= lonmax))[0]
    lats_25km = lats[i_lat_25km]
    lons_25km = lons[i_lon_25km]
    agb_25km = np.full((len(lats_25km), len(lons_25km)), np.nan)
    err_25km = agb_25km.copy()
    dimensions = OrderedDict([('lat', lats_25km), ('lon', lons_25km)])
    variables = ['AGB', 'AGB_err']

    # Matchup
    i_lat, i_lon = [], []
    for lat in lats_100m:
        i_lat += [np.argmin(np.abs(lats_25km-lat))]
    i_lat = np.array(i_lat)
    for lon in lons_100m:
        i_lon += [np.argmin(np.abs(lons_25km-lon))]
    i_lon = np.array(i_lon)

    # Read and concatenate .tif images
    for i_row, row in enumerate(rows):
        for i_col, col in enumerate(cols):
            fname_agb = root / 'raw'/ f'{row}{col}_agb/{row}{col}_agb.tif'
            fname_err = root / 'raw'/ f'{row}{col}_agb/{row}{col}_agb_err.tif'
            with rio.open(fname_agb) as ds_agb, rio.open(fname_err) as ds_err:
                agb[n_xy*i_row : n_xy*i_row+n_xy, n_xy*i_col : n_xy*i_col+n_xy] = ds_agb.read(1)
                err[n_xy*i_row : n_xy*i_row+n_xy, n_xy*i_col : n_xy*i_col+n_xy] = ds_err.read(1)

    # clip domain to South America
    agb = agb[i_lat_100m[0]:i_lat_100m[-1]+1, i_lon_100m[0]:i_lon_100m[-1]+1]
    err = err[i_lat_100m[0]:i_lat_100m[-1]+1, i_lon_100m[0]:i_lon_100m[-1]+1]

    # resample to 25km and store to .nc
    with ncfile_init(fout, dimensions, variables) as ds:
        for i_lon_25 in range(len(lons_25km)):
            for i_lat_25 in range(len(lats_25km)):

                i_lons_cell = np.where(i_lon == i_lon_25)[0]
                i_lats_cell = np.where(i_lat == i_lat_25)[0]

                if mask[i_lat_25km[i_lat_25], i_lon_25km[i_lon_25]] == 0:
                    ds.variables['AGB'][i_lat_25, i_lon_25] = np.nanmean(
                        agb[i_lats_cell[0]:i_lats_cell[-1] + 1, i_lons_cell[0]:i_lons_cell[-1] + 1])
                    ds.variables['AGB_err'][i_lat_25, i_lon_25] = np.nanmean(
                        err[i_lats_cell[0]:i_lats_cell[-1] + 1, i_lons_cell[0]:i_lons_cell[-1] + 1])

def reformat_TCL():

    root = Path('/Users/u0116961/data_sets/tree_cover_loss')
    fout = root / 'resampled' / 'TCL_25km.nc'

    maskfile = root.parent / 'EASE2_M25km.LOCImask_land50_coast0km.1388x584.bin'
    mask = np.fromfile(maskfile, dtype='byte').reshape(584,1388)

    # tile name indicators
    rows = ['20N', '10N', '00N', '10S', '20S', '30S', '40S', '50S']
    cols = ['090W', '080W', '070W', '060W', '050W', '040W']

    # 40 000 pixels in each 10 x 10 deg box
    n_xy = 40000
    spc = 10 / n_xy

    # TCL grid
    tcl = np.full((len(rows)*n_xy,len(cols)*n_xy), 2**8-1, dtype='uint8')
    lats_30m = (np.arange(tcl.shape[0]) * spc - 60 + spc/2)[::-1]
    lons_30m = (np.arange(tcl.shape[1]) * spc - 90 + spc/2)
    latmin, lonmin, latmax, lonmax = get_roi()
    i_lat_30m = np.where((lats_30m>=latmin)&(lats_30m<=latmax))[0]
    i_lon_30m = np.where((lons_30m>=lonmin)&(lons_30m<=lonmax))[0]
    lats_30m = lats_30m[i_lat_30m]
    lons_30m = lons_30m[i_lon_30m]


    # EASE25 grid
    grid = EASE2(gtype='M25')
    lats, lons = grid.ease_lats, grid.ease_lons
    i_lat_25km = np.where((lats >= latmin) & (lats <= latmax))[0]
    i_lon_25km = np.where((lons >= lonmin) & (lons <= lonmax))[0]
    lats_25km = lats[i_lat_25km]
    lons_25km = lons[i_lon_25km]
    tcl_25km = np.full((len(lats_25km), len(lons_25km)), np.nan)
    dimensions = OrderedDict([('lat', lats_25km), ('lon', lons_25km)])
    variables = ['TCL',]

    # Matchup
    i_lat, i_lon = [], []
    for lat in lats_30m:
        i_lat += [np.argmin(np.abs(lats_25km-lat))]
    i_lat = np.array(i_lat)
    for lon in lons_30m:
        i_lon += [np.argmin(np.abs(lons_25km-lon))]
    i_lon = np.array(i_lon)

    # Read and concatenate .tif images
    for i_row, row in enumerate(rows):
        for i_col, col in enumerate(cols):
            fname_tcl = root / 'raw'/ f'Hansen_GFC-2020-v1.8_lossyear_{row}_{col}.tif'
            with rio.open(fname_tcl) as ds_tcl:
                tcl[n_xy*i_row : n_xy*i_row+n_xy, n_xy*i_col : n_xy*i_col+n_xy] = ds_tcl.read(1)

    # clip domain to South America
    tcl = tcl[i_lat_30m[0]:i_lat_30m[-1]+1, i_lon_30m[0]:i_lon_30m[-1]+1]

    # resample to 25km and store to .nc
    with ncfile_init(fout, dimensions, variables) as ds:
        for i_lon_25 in range(len(lons_25km)):
            for i_lat_25 in range(len(lats_25km)):

                i_lons_cell = np.where(i_lon == i_lon_25)[0]
                i_lats_cell = np.where(i_lat == i_lat_25)[0]

                if mask[i_lat_25km[i_lat_25], i_lon_25km[i_lon_25]] == 0:
                    tmp = tcl[i_lats_cell[0]:i_lats_cell[-1] + 1, i_lons_cell[0]:i_lons_cell[-1] + 1]
                    ds.variables['TCL'][i_lat_25, i_lon_25] = len(np.where(tmp != 0)[0]) / tmp.size

def generate_MERRA2_EASE_LUT():

    fout = '/Users/u0116961/data_sets/LUT_EASE25_MERRA2_South_America.csv'

    # MERRA2 coordinates
    with Dataset('/Users/u0116961/data_sets/MERRA2/south_america_2010_2020/MERRA2_images.nc') as ds:
        lats_merra = ds['lat'][:].data
        lons_merra = ds['lon'][:].data

    # EASE2 coordinates
    grid = EASE2(gtype='M25')
    lats, lons = grid.ease_lats, grid.ease_lons
    latmin, lonmin, latmax, lonmax = get_roi()
    i_lat_25km = np.where((lats >= latmin) & (lats <= latmax))[0]
    i_lon_25km = np.where((lons >= lonmin) & (lons <= lonmax))[0]
    i_lons, i_lats = np.meshgrid(i_lon_25km, i_lat_25km)
    i_lons = i_lons.flatten()
    i_lats = i_lats.flatten()

    # Extract land points
    maskfile = '/Users/u0116961/data_sets/EASE2_M25km.LOCImask_land50_coast0km.1388x584.bin'
    mask = np.fromfile(maskfile, dtype='byte').reshape(584,1388)
    mask = mask[i_lats, i_lons]
    ind_valid = np.where(mask == 0)[0]

    # Land coordinates on the EASE grid
    lats = lats[i_lats[ind_valid]]
    lons = lons[i_lons[ind_valid]]

    # EASE columns/rows for south american land points only
    i_lons, i_lats = np.meshgrid(np.arange(len(i_lon_25km)), np.arange(len(i_lat_25km)))
    i_lons = i_lons.flatten()
    i_lats = i_lats.flatten()

    res = pd.DataFrame({'lat': lats,
                        'lon': lons,
                        'row_ease': i_lats[ind_valid],
                        'col_ease': i_lons[ind_valid],
                        'row_merra': -9999,
                        'col_merra': -9999})

    for i, val in res.iterrows():
        res.loc[i,'row_merra'] = np.argmin(abs(lats_merra-val.lat))
        res.loc[i,'col_merra'] = np.argmin(abs(lons_merra-val.lon))

    res.to_csv(fout, float_format='%0.4f')


if __name__=='__main__':

    # reformat_MERRA2()
    # reformat_SMOS_IC()
    # reformat_COPERNICUS_LAI()
    # reformat_AGB()
    # reformat_TCL()
    # generate_MERRA2_EASE_LUT()
    pass

