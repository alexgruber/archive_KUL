import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
from netCDF4 import Dataset, num2date

from smecv.process.smecv_utils import get_smecv_grid
from ascat.read_native.cdr import load_grid
from pyldas.grids import EASE2

from smecv.input.smecv_netcdf import default_ds_class
from smecv.input.gldas import GLDAS025v21
from myprojects.readers.ascat import HSAF_io
from myprojects.readers.insitu import ISMN_io


class io(object):

    def __init__(self):

        self.lut = pd.read_csv(r"D:\data_sets\auxiliary\lut_ease_025_dgg.csv", index_col=0)

        grid = get_smecv_grid()()

        path_ascat_hsaf = r"D:\data_sets\HSAF"
        path_gldas = r"R:\Datapool\GLDAS\02_processed\GLDAS_NOAH025_3H.2.1\datasets\netcdf"
        path_ascat_025 = r"D:\data_sets\SMECV\ESA_CCI_SM_v07.1\020c_active_joinedTS\ascat"
        path_amsr2d = r"D:\data_sets\SMECV\ESA_CCI_SM_v07.1\011_resampledTemporal\amsr2_d"
        path_smap = r"D:\data_sets\SMAP\SPL3SMP.008\reformated\timeseries.nc"

        self.ds_hsaf = HSAF_io(root=path_ascat_hsaf)
        self.ds_gldas = GLDAS025v21(path_gldas, mask_frozen=True, grid=grid)
        self.ds_ascat025 = default_ds_class(path_ascat_025, grid=grid)
        self.ds_amsr2d = default_ds_class(path_amsr2d, grid=grid)
        self.ds_smap = Dataset(path_smap)

    def lonlat2idx(self, lon, lat):
        return np.argmin((self.lut.ease_lon - lon) ** 2 + (self.lut.ease_lat - lat) ** 2)

    def lonlat2colrow(self, lon, lat):

        londiff = np.abs(self.lut.ease_lon - lon)
        latdiff = np.abs(self.lut.ease_lat - lat)

        col = self.lut.loc[np.argmin(londiff), 'ease_col']
        row = self.lut.loc[np.argmin(latdiff), 'ease_row']

        return col, row

    def close(self):
        self.ds_hsaf.close()
        self.ds_gldas.close()
        self.ds_ascat025.close()
        self.ds_amsr2d.close()
        self.ds_smap.close()

    def read_smap_aux(self, idx):
        variables = ['vegetation_opacity', 'vegetation_water_content',
                      'landcover_class', 'landcover_class_fraction']
        keys = ['vegetation_opacity_med', 'vegetation_water_content_med',
                'vegetation_opacity_std', 'vegetation_water_content_std',
                'vegetation_opacity_iqr', 'vegetation_water_content_iqr',
                'landcover_class', 'landcover_class_fraction']
        res = dict(zip(keys, np.full(len(keys),np.nan)))
        try:
            for var in variables:
                data = self.ds_smap[var][:, self.lut.loc[idx, 'ease_row'], self.lut.loc[idx, 'ease_col']]
                if 'landcover' in var:
                    data = data[data != 254]
                if len(data) == 0:
                    res[var] = np.nan
                else:
                    if 'vegetation' in var:
                        perc = np.nanpercentile(data,[25,50,75])
                        res[f'{var}_med'] = perc[1]
                        res[f'{var}_std'] = np.nanstd(data)
                        res[f'{var}_iqr'] = perc[2]-perc[0]
                    else:
                        res[var] = np.nanmedian(data)
        except:
            pass

        return pd.Series(res)


    def read_ts(self, idx, name):
        """ Read time series for the index in the LUT """

        ts = pd.Series()
        try:
            if name.upper() == 'HSAF':
                ts = self.ds_hsaf.read(self.lut.loc[idx,'gpi_dgg'], sampling_freq=24.)['2015-07-01':'2021-07-01']
            elif name.upper() == 'ASCAT':
                ts = self.ds_ascat025.read(self.lut.loc[idx, 'gpi_025'], mask_sm_nan=True)['sm']['2015-07-01':'2021-07-01']
            elif name.upper() == 'GLDAS':
                ts = self.ds_gldas.read(self.lut.loc[idx, 'gpi_025'])['sm']['2015-07-01':'2021-07-01'].resample('1D').nearest()
            elif name.upper() == 'AMSR2':
                ts = self.ds_amsr2d.read(self.lut.loc[idx, 'gpi_025'], mask_sm_nan=True)['sm']['2015-07-01':'2021-07-01']
            elif name.upper() == 'SMAP':
                time = pd.DatetimeIndex(num2date(self.ds_smap['time'][:], units=self.ds_smap['time'].units,
                                         only_use_python_datetimes=True, only_use_cftime_datetimes=False))
                data = self.ds_smap['soil_moisture'][:, self.lut.loc[idx, 'ease_row'], self.lut.loc[idx, 'ease_col']]
                ts = pd.Series(data, index=time).dropna()
            else:
                print(f"Unknown dataset: {name}")
        except:
            pass

        ts.name = name.upper()
        return ts

    def read_df(self, idx, names=None):

        if names is None:
            names = ['GLDAS', 'ASCAT', 'AMSR2', 'SMAP']

        dss = [self.read_ts(idx, name) for name in names]

        # ts_gldas = self.read_ts(idx, 'GLDAS') / 100.
        # ts_hsaf = self.read_ts(idx, 'HSAF') / 100.
        # ts_ascat = self.read_ts(idx, 'ASCAT') / 100.
        # ts_amsr2 = self.read_ts(idx, 'AMSR2') / 100.
        # ts_smap = self.read_ts(idx, 'SMAP')

        return pd.concat(dss, axis=1).dropna()

def generate_lut():

    fout = r"D:\data_sets\auxiliary\lut_ease_025_dgg.csv"

    grid_025 = get_smecv_grid()()
    grid_ease = EASE2(gtype='M36')
    grid_dgg = load_grid(r"D:\data_sets\auxiliary\warp5_grid\TUW_WARP5_grid_info_2_3.nc")

    cols = ['ease_lat', 'ease_lon', 'ease_row', 'ease_col', 'gpi_025', 'cell_025', 'gpi_dgg', 'cell_dgg']

    res = pd.DataFrame(columns=cols, index=np.arange(len(grid_ease.ease_lats)*len(grid_ease.ease_lons)))

    i = -1
    for row, lat in enumerate(grid_ease.ease_lats):
        for col, lon in enumerate(grid_ease.ease_lons):
            i += 1
            print(f'{i} / {len(res)}')

            gpi_025 = grid_025.find_nearest_gpi(lon, lat, max_dist=18000)[0]
            if gpi_025 not in grid_025.activegpis:
                continue

            gpi_dgg = grid_dgg.find_nearest_gpi(lon, lat, max_dist=18000)[0]
            if gpi_dgg not in grid_dgg.activegpis:
                continue

            res.loc[i, 'ease_lat'] = lat
            res.loc[i, 'ease_lon'] = lon
            res.loc[i, 'ease_row'] = row
            res.loc[i, 'ease_col'] = col
            res.loc[i, 'gpi_025'] = gpi_025
            res.loc[i, 'cell_025'] = grid_025.activearrcell[grid_025.activegpis == gpi_025][0]
            res.loc[i, 'gpi_dgg'] = gpi_dgg
            res.loc[i, 'cell_dgg'] = grid_dgg.activearrcell[grid_dgg.activegpis == gpi_dgg][0]

    res = res.dropna()
    res.sort_values("cell_025", inplace=True)
    res.index = np.arange(len(res))
    res.to_csv(fout, float_format='%0.4f')

