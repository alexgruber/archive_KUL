
import os
import pathlib

import numpy as np
import pandas as pd

from netCDF4 import Dataset, date2num, num2date

from osgeo import osr
from osgeo import gdal

import rasterio
import shapefile
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from mpl_toolkits.basemap import Basemap

from myprojects.readers.ascat import HSAF_io

def create_mask():

    root = pathlib.Path('/data_sets/LIS/NoahMP_belgium/')
    tmp_fname = root / 'tmp_mask.tif'

    shp_bel = '/data_sets/LIS/shapefiles/BEL_adm/BEL_adm0.shp'
    fname_bel = root / 'mask.tif'
    shp_dem = '/data_sets/LIS/shapefiles/catchments/be_demer_wgs84/be_demer_wgs84.shp'
    fname_dem = root / 'mask_demer.tif'
    shp_our = '/data_sets/LIS/shapefiles/catchments/be_ourthe_wgs84/be_ourthe_wgs84.shp'
    fname_our = root / 'mask_ourthe.tif'

    with Dataset(root / 'images.nc') as ds:
        lats = ds['lat'][:,:]; lons = ds['lon'][:,:]

    latmin, latmax, n_lats = lats.min(), lats.max(), lats.shape[0]
    lonmin, lonmax, n_lons = lons.min(), lons.max(), lons.shape[1]
    image_size = lats.shape

    nx = n_lons
    ny = n_lats
    xmin, ymin, xmax, ymax = [lonmin, latmin, lonmax, latmax]
    xres = (xmax - xmin) / float(nx)
    yres = (ymax - ymin) / float(ny)
    geotransform = (xmin, xres, 0, ymax, 0, -yres)

    img = np.ones(image_size, dtype=np.byte)

    dst_ds = gdal.GetDriverByName('GTiff').Create(str(tmp_fname), nx, ny, 1, gdal.GDT_Byte)
    dst_ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds.GetRasterBand(1).WriteArray(img)
    dst_ds.FlushCache()
    dst_ds = None

    gdal.Warp(str(fname_bel), str(tmp_fname), dstNodata=0, cutlineDSName=str(shp_bel))
    gdal.Warp(str(fname_dem), str(tmp_fname), dstNodata=0, cutlineDSName=str(shp_dem))
    gdal.Warp(str(fname_our), str(tmp_fname), dstNodata=0, cutlineDSName=str(shp_our))

    tmp_fname.unlink()



def reformat_ascat():

    outfile_ts = '/data_sets/LIS/ASCAT/timeseries.nc'
    outfile_img = '/data_sets/LIS/ASCAT/images.nc'

    with rasterio.open('/data_sets/LIS/NoahMP_belgium/mask.tif') as ds:
        mask = np.flipud(ds.read()[0, :, :])

    with Dataset('/data_sets/LIS/NoahMP_belgium/images.nc') as ds:
        lats = ds.variables['lat'][:,:]
        lons = ds.variables['lon'][:,:]
        timeunit = ds['time'].units
        dates = ds['time'][:]
        pydates = pd.to_datetime(num2date(dates, units=timeunit))

    io = HSAF_io()
    gpis = pd.read_csv('/data_sets/LIS/NoahMP_belgium/pointlist_Belgium_warp.csv', index_col=0)

    lats.mask[mask == 0] = True
    lons.mask[mask == 0] = True
    inds = np.where(~lats.mask)

    tmp_list = pd.DataFrame({'row': inds[0],
                             'col': inds[1],
                             'gpi': np.full(len(inds[0]), 0, dtype='int64'),
                             'cell': np.full(len(inds[0]), 0, dtype='int64')})
    for idx, data in tmp_list.iterrows():
        print('%i / %i' % (idx+1, len(tmp_list)))
        r, c = data['row'], data['col']
        gpi = ((gpis.lat - lats[r,c])**2 + (gpis.lon - lons[r,c])**2).idxmin()
        tmp_list.loc[idx,'gpi'] = gpi
        tmp_list.loc[idx,'cell'] = gpis.loc[gpi, 'cell']
    tmp_list.to_csv('/data_sets/LIS/NoahMP_belgium/tmp_list.csv')
    # tmp_list = pd.read_csv('/data_sets/LIS/NoahMP_belgium/tmp_list.csv', index_col=0)


    with Dataset(outfile_ts, mode='w') as res:

        res.createDimension('lat', lats.shape[0])
        res.createDimension('lon', lons.shape[1])
        res.createDimension('time', len(dates))

        res.createVariable('lat', ds['lat'].dtype, dimensions=('lat','lon'), chunksizes=(1,1), zlib=True)
        res.createVariable('lon', ds['lon'].dtype, dimensions=('lat','lon'), chunksizes=(1,1), zlib=True)
        res.createVariable('time', dates.dtype, dimensions=('time',), chunksizes=(len(dates),), zlib=True)
        res.variables['lat'][:,:] = lats
        res.variables['lon'][:,:] = lons
        res.variables['time'][:] = dates

        # Coordinate attributes following CF-conventions
        res.variables['time'].setncatts({'long_name': 'time', 'units': timeunit})
        res.variables['lon'].setncatts({'long_name': 'longitude', 'units':'degrees_east'})
        res.variables['lat'].setncatts({'long_name': 'latitude', 'units':'degrees_north'})

        res.createVariable('SoilMoisture', 'float32',
                           dimensions=('time', 'lat', 'lon'), chunksizes=(len(dates),1,1), zlib=True)
        res.variables['SoilMoisture'].setncatts({'missing_value': -9999})

        i = 0
        for cell in tmp_list['cell'].unique():
            for gpi in tmp_list.loc[tmp_list['cell']==cell, 'gpi'].unique():
                print('%i / %i' % (i, len(tmp_list)))

                cell_gpi_list = tmp_list.loc[tmp_list['gpi']==gpi]

                try:
                    ts = io.read(gpi, resample_time=False).resample('6h').mean().dropna()[pydates].values
                    np.place(ts, np.isnan(ts), -9999)
                    for idx, data in cell_gpi_list.iterrows():
                        i += 1
                        res.variables['SoilMoisture'][:, data['row'], data['col']] = ts

                except:
                    print('gpi %i failed' % gpi)
                    continue

    cmdBase = 'ncks -4 -L 4 --cnk_dmn time,1 --cnk_dmn lat,%i --cnk_dmn lon,%i ' % (lats.shape)
    cmd = ' '.join([cmdBase, outfile_ts, outfile_img])
    os.system(cmd)



def reformat_lis_files():

    path = pathlib.Path('/data_sets/LIS/NoahMP_belgium/exp1_OL/OUT_OL_Belgium/SURFACEMODEL')
    outfile_img = path / 'images.nc'
    outfile_ts = path / 'timeseries.nc'

    files = sorted(path.glob('**/*.d01.nc'))

    timeunit = 'hours since 2000-01-01 00:00'
    dates = date2num(pd.to_datetime([f.name[-19:-7] for f in files], format='%Y%m%d%H%M').to_pydatetime(), timeunit).astype('int32')

    with Dataset(outfile_img, mode='w') as res:

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
    # reformat_ascat()
    create_mask()

