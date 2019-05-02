

import os
import tarfile
import platform

import numpy as np
import pandas as pd

from collections import OrderedDict
from netCDF4 import Dataset, date2num, num2date

from myprojects.functions import find_files

from pyldas.interface import LDAS_io


class SMOS_io(object):

    def __init__(self, root=None):

        if root is None:
            if platform.system() == 'Windows':
                root = os.path.join('D:','data_sets', 'SMOS_L3')
            elif platform.system() == 'Linux':
                root = os.path.join('/', 'data', 'leuven', '320', 'vsc32046', 'data_sets', 'SMOS')
            else:
                root = os.path.join('/','data_sets', 'SMOS_L3')

        self.loaded_cell=None
        self.ds = None

        self.grid = pd.read_csv(find_files(root,'grid.csv'), index_col=0)

        self.root = os.path.join(root, 'cellfiles')

    def load(self, cell):

        fname = os.path.join(self.root, '%04i.nc' % cell)
        if not os.path.exists(fname):
            print('File not found: ' + fname)
            return False

        try:
            if self.ds is not None:
                self.ds.close()
            self.ds = Dataset(fname)
        except:
            print('Corrupted cell: %i' % cell)
            return False

        self.loaded_cell = cell

        return True

    def read(self, *args):
        # TODO: PASSING OF COL/ROW NOT SUPPORTED FOR CELLFILE STRUCTURE!!
        # 1 args = gpi
        # 2 args = col , row

        if len(args) == 1:

            if hasattr(self,'loaded_cell'):
                if self.loaded_cell != self.grid.loc[args[0],'dgg_cell']:
                    loaded = self.load(self.grid.loc[args[0],'dgg_cell'])
                    if loaded is False:
                        return None
            row = self.grid.loc[args[0],'row']
            col = self.grid.loc[args[0],'col']

        else:
            col = args[0]
            row = args[1]

        ts = self.ds['soil_moisture'][:, row, col]
        dates = num2date(self.ds['time'][:], units=self.ds['time'].units)

        return pd.Series(ts, index=dates)

    def iter_gp(self):
        for cell in np.unique(self.grid.dgg_cell):
            gpis = self.grid[self.grid.dgg_cell==cell]
            for gpi, info in gpis.iterrows():
                data = self.read(gpi)
                yield data, info

    def iter_cell(self, cell):
        gpis = self.grid[self.grid.dgg_cell==cell]
        for gpi, info in gpis.iterrows():
            data = self.read(gpi)
            yield data, info

    def close(self):
        if hasattr(self,'ds'):
            if self.ds is not None:
                self.ds.close()


def generate_grid_file():

    files = find_files(r'D:\data_sets\SMOS_L3\cellfiles', '.nc')

    dgg = pd.read_csv(r"D:\data_sets\ASCAT\warp5_grid\pointlist_warp_conus.csv", index_col=0)
    ease_grid = LDAS_io(exp='US_M36_SMOS_DA_cal_scaled_yearly').grid

    grid = pd.DataFrame()

    for cnt, f in enumerate(files):
        print('%i / %i' % (cnt, len(files)))

        tmp = Dataset(f)
        lats = tmp.variables['lat'][:]
        lons = tmp.variables['lon'][:]
        tmp.close()

        offset = grid.index.values[-1] + 1 if len(grid) > 0 else 0
        idx = np.arange(offset, len(lats)*len(lons) + offset)
        tmp_grid = pd.DataFrame(columns=['lat', 'lon', 'row', 'col', 'ease_row', 'ease_col', 'dgg_cell', 'dgg_gpi'], index=idx)

        for row, lat in enumerate(lats):
            for col, lon in enumerate(lons):
                tmp_grid.loc[offset, 'lat'] = lat
                tmp_grid.loc[offset, 'lon'] = lon

                tmp_grid.loc[offset, 'row'] = row
                tmp_grid.loc[offset, 'col'] = col

                ease_col, ease_row = ease_grid.lonlat2colrow(lon, lat, domain=True)
                tmp_grid.loc[offset, 'ease_row'] = ease_row
                tmp_grid.loc[offset, 'ease_col'] = ease_col

                tmp_grid.loc[offset, 'dgg_cell'] = int(os.path.basename(f)[0:4])
                r = np.sqrt((dgg.lon - lon)**2 + (dgg.lat - lat)**2)
                tmp_grid.loc[offset, 'dgg_gpi'] = dgg.iloc[np.where(abs(r - r.min()) < 0.0001)[0][0], 0]

                offset += 1

        grid = pd.concat((grid,tmp_grid))

    grid.to_csv(r'D:\data_sets\SMOS_L3\grid.csv')

def generate_cell_files():

    path_in = r'D:\data_sets\SMOS_L3\unzipped' + '\\'
    path_out = r'D:\data_sets\SMOS_L3\cellfiles' + '\\'
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # SMOS image coordinates
    tmp = Dataset(r"D:\data_sets\SMOS_L3\unzipped\2010\015\SM_RE04_MIR_CLF31A_20100115T000000_20100115T235959_300_001_7.DBL.nc")
    lats = tmp.variables['lat'][:]
    lons = tmp.variables['lon'][:]
    tmp.close()

    # WARP cell's and coordinates
    dgg_grid = Dataset(r"D:\data_sets\ASCAT\warp5_grid\TUW_WARP5_grid_info_2_2.nc")
    dgg_lats = dgg_grid['lat'][:]
    dgg_lons = dgg_grid['lon'][:]
    dgg_cells = dgg_grid['cell'][:]
    dgg_grid.close()

    # Cell list
    conus_gpis = pd.read_csv(r"D:\data_sets\ASCAT\warp5_grid\pointlist_warp_conus.csv",index_col=0)
    cells = np.unique(conus_gpis['cell'])

    # NC parameters
    timeunit = 'hours since 2000-01-01 00:00'
    smunit = 'm3/m3'

    # Date range
    dates = pd.date_range('2010-01-15','2015-05-06').to_pydatetime()
    num_dates = date2num(dates, timeunit).astype('int32')

    for cell in cells:
        print(cell)

        latmin = dgg_lats[dgg_cells==cell].min(); latmax = dgg_lats[dgg_cells==cell].max()
        lonmin = dgg_lons[dgg_cells==cell].min(); lonmax = dgg_lons[dgg_cells==cell].max()

        ind_lats = np.where((lats>=latmin)&(lats<=latmax))[0]; ind_lons = np.where((lons>=lonmin)&(lons<=lonmax))[0]
        tmp_lats = lats[ind_lats]; tmp_lons = lons[ind_lons]

        res_arr = np.full((len(dates), len(tmp_lats), len(tmp_lons)), np.nan)

        # Read in SMOS native files
        for idx, date in enumerate(dates):
            print('%i / %i' % (idx, len(dates)))

            files = find_files(os.path.join(path_in, date.strftime('%Y')), date.strftime('%Y%m%d'))
            if files is None:
                continue

            tmp_res = np.full((len(tmp_lats),len(tmp_lons),2),np.nan)
            for i,f in enumerate(files):
                ds = Dataset(f)
                data = ds.variables['Soil_Moisture'][ind_lats,ind_lons]
                tmp_res[:, :, i] = data
                if hasattr(data,'fill_value'):
                    tmp_res[tmp_res == data.fill_value] = np.nan
                ds.close()
            res_arr[idx, :, :] = np.nanmean(tmp_res,axis=2)


        # store to NetCDF cell file
        fname = os.path.join(path_out,'%04i.nc' % cell)
        ds = Dataset(fname, mode='w')
        dimensions = OrderedDict([('time', num_dates), ('lat', tmp_lats), ('lon', tmp_lons)])

        # Create/Write dimensions
        chunksizes = []
        for key, values in dimensions.iteritems():

            if key in ['lon', 'lat']:
                chunksize = 1
            else:
                chunksize = len(values)
            chunksizes.append(chunksize)

            dtype = values.dtype
            ds.createDimension(key, len(values))
            ds.createVariable(key, dtype,
                              dimensions=(key,),
                              chunksizes=(chunksize,),
                              zlib=True)
            ds[key][:] = values

        # Create/Write data
        ds.createVariable('soil_moisture', 'float32',
                          dimensions=dimensions.keys(),
                          chunksizes=chunksizes,
                          fill_value=-9999.,
                          zlib=True)
        ds['soil_moisture'][:, :, :] = res_arr

        ds.variables['time'].setncattr('units', timeunit)
        ds.variables['soil_moisture'].setncattr('units', smunit)

        ds.close()


def extract_L3_tar_files():

    root = r"D:\data_sets\SMOS_L3\raw"

    files = find_files(root, '.tgz')

    for cnt,f in enumerate(files):
        print('%i / %i' % (cnt, len(files)))

        out_path = os.path.dirname(f).replace('raw', 'unzipped').replace('asc', 'ascdsc').replace('dsc', 'ascdsc')
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        tmp = tarfile.open(f)
        tmp.extract(tmp.getmember([x for x in tmp.getnames() if x.find('.nc') != -1][0]), out_path)
        tmp.close()


if __name__ == '__main__':

    generate_grid_file()
    #
    # io = SMOS_io()
    # data = io.read(11852)