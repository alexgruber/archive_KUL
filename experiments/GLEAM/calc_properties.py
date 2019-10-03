
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
from multiprocessing import Pool

from netCDF4 import Dataset, num2date

from pygleam_ag.GLEAM_model import GLEAM
from pygleam_ag.GLEAM_IO import output_ts
from pygleam_ag.grid import get_valid_gpis

import pygeogrids.netcdf as ncgrid
from smecv.common_format import CCIDs

from scipy.stats import pearsonr
from validation_good_practice.ancillary.metrics import TCA_calc

from myprojects.functions import merge_files

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from scipy.ndimage import gaussian_filter


def proc(function, procs = 30):
    ''' Wrapper routine for parallelized processing '''

    gpis_valid = get_valid_gpis(latmin=24., latmax=51., lonmin=-128., lonmax=-64.)

    if procs == 1:
        function(gpis_valid)

    else:
        subs = (np.arange(procs + 1) * len(gpis_valid) / procs).astype('int')
        subs[-1] = len(gpis_valid)
        gpi_subs = [gpis_valid[subs[i]:subs[i+1]] for i in np.arange(procs)]

        p = Pool(procs)
        p.map(function, gpi_subs)


def write_output(fname,out_dict,gpi):

    output = pd.DataFrame(out_dict, index=(gpi,))

    if not fname.exists():
        output.to_csv(fname, float_format='%0.4f')
    else:
        output.to_csv(fname, float_format='%0.4f', mode='a', header=False)



def fill_error_gaps():

    res = pd.read_csv('/work/GLEAM/errors/result.csv', index_col=0)

    gpis_valid = get_valid_gpis(latmin=24., latmax=51., lonmin=-128., lonmax=-64.)
    ind_valid = np.unravel_index(gpis_valid, (720,1440))

    res_gapfilled = pd.DataFrame(index=gpis_valid)

    r_min = 0.0

    ind = (res['R_GLEAM_ASCAT'] <= r_min) | \
          (res['R_GLEAM_AMSR2'] <= r_min) | \
          (res['R_ASCAT_AMSR2'] <= r_min)

    res.loc[ind, ['TC1_R2_GLEAM', 'TC1_R2_ASCAT', 'TC1_R2_AMSR2']] = np.nan
    res.loc[ind, ['TC1_RMSE_GLEAM', 'TC1_RMSE_ASCAT', 'TC1_RMSE_AMSR2']] = np.nan

    # ind = (res['p_GLEAM_ASCAT'] >= 0.05) | \
    #       (res['p_GLEAM_AMSR2'] >= 0.05) | \
    #       (res['p_ASCAT_AMSR2'] >= 0.05)
    #
    # res.loc[ind, ['TC1_R2_GLEAM', 'TC1_R2_ASCAT', 'TC1_R2_AMSR2']] = np.nan
    # res.loc[ind, ['TC1_RMSE_GLEAM', 'TC1_RMSE_ASCAT', 'TC1_RMSE_AMSR2']] = np.nan

    ind = (res['R_GLEAM_ASCAT'] <= r_min) | \
          (res['R_GLEAM_SMAP'] <= r_min) | \
          (res['R_ASCAT_SMAP'] <= r_min)

    res.loc[ind, ['TC2_R2_GLEAM', 'TC2_R2_ASCAT', 'TC2_R2_SMAP']] = np.nan
    res.loc[ind, ['TC2_RMSE_GLEAM', 'TC2_RMSE_ASCAT', 'TC2_RMSE_SMAP']] = np.nan

    # ind = (res['p_GLEAM_ASCAT'] >= 0.05) | \
    #       (res['p_GLEAM_SMAP'] >= 0.05) | \
    #       (res['p_ASCAT_SMAP'] >= 0.05)
    #
    # res.loc[ind, ['TC2_R2_GLEAM', 'TC2_R2_ASCAT', 'TC2_R2_SMAP']] = np.nan
    # res.loc[ind, ['TC2_RMSE_GLEAM', 'TC2_RMSE_ASCAT', 'TC2_RMSE_SMAP']] = np.nan

    # ---------------------

    # tags = ['TC1_R2_GLEAM',]
    tags = ['TC1_R2_GLEAM', 'TC1_R2_ASCAT', 'TC1_R2_AMSR2',
            'TC2_R2_GLEAM', 'TC2_R2_ASCAT', 'TC2_R2_SMAP',
            'TC1_RMSE_GLEAM', 'TC1_RMSE_ASCAT', 'TC1_RMSE_AMSR2',
            'TC2_RMSE_GLEAM', 'TC2_RMSE_ASCAT', 'TC2_RMSE_SMAP']

    imp = IterativeImputer(max_iter=10, random_state=0)
    ind = np.unravel_index(res.index.values, (720,1440))
    for tag in tags:
        img = np.full((720,1440), np.nan)
        img[ind] = res[tag]

        # find all non-zero values
        idx = np.where(~np.isnan(img))
        vmin, vmax = np.percentile(img[idx], [2.5, 97.5])
        img[img<vmin] = vmin
        img[img>vmax] = vmax

        # calculate fitting parameters
        imp.set_params(min_value=vmin, max_value=vmax)
        imp.fit(img)

        # Define an anchor pixel to infer fitted image dimensions
        tmp_img = img.copy()
        tmp_img[idx[0][100],idx[1][100]] = 1000000

        # transform image with and without anchor pixel
        tmp_img_fitted = imp.transform(tmp_img)
        img_fitted = imp.transform(img)


        # # Get indices of fitted image
        idx_anchor = np.where(tmp_img_fitted == 1000000)[1][0]
        start = idx[1][100] - idx_anchor
        end = start + img_fitted.shape[1]

        # write output
        img[:,start:end] = img_fitted
        img = gaussian_filter(img, sigma=0.7, truncate=1)

        res_gapfilled.loc[:, tag] = img[ind_valid]

        # np.save('/work/GLEAM/test', img)

        print(tag, 'finished.')

    res_gapfilled.to_csv('/work/GLEAM/errors/result_gapfilled_sig07.csv')


def calc_errors(gpis):

    outpath = Path('/work/GLEAM/errors')

    if not outpath.exists():
        outpath.mkdir(parents=True)

    fname = outpath / ('part_%i.csv' % gpis[0])

    cci_gpis = np.flipud(np.arange(720 * 1440).reshape((720, 1440))).flatten()
    cci_grid = ncgrid.load_grid('/data_sets/ESA_CCI_L2/ESA-CCI-SOILMOISTURE-LAND_AND_RAINFOREST_MASK-fv04.2.nc',
                            subset_flag='land', subset_value=1.)

    asc_io = CCIDs('/data_sets/ESA_CCI_L2/data/ascata', grid=cci_grid)
    ams_io = CCIDs('/data_sets/ESA_CCI_L2/data/amsr2', grid=cci_grid)
    sma_io = CCIDs('/data_sets/ESA_CCI_L2/data/smap', grid=cci_grid)

    for cnt, gpi in enumerate(np.atleast_1d(gpis)):

        try:
            gleam_io = Dataset('/data_sets/GLEAM/_output/timeseries/%i.nc' % gpi)
            gle_ts = pd.Series(gleam_io.variables['w1'][:, 0],
                               index=num2date(gleam_io['time'][:], units=gleam_io['time'].units), name='GLEAM') * 100

            asc_ts = asc_io.read(cci_gpis[gpi], only_valid=True)['sm'];asc_ts.name = 'ASCAT'
            ams_ts = ams_io.read(cci_gpis[gpi], only_valid=True)['sm'];ams_ts.name = 'AMSR2'
            sma_ts = sma_io.read(cci_gpis[gpi], only_valid=True)['sm'];sma_ts.name = 'SMAP'

            df = pd.concat((gle_ts, asc_ts, ams_ts, sma_ts), axis='columns').dropna()

            result = {'n': len(df)}

            for i,ds1 in enumerate(['GLEAM', 'ASCAT', 'AMSR2']):
                for ds2 in ['ASCAT', 'AMSR2', 'SMAP'][i::]:
                    R, p = pearsonr(df[ds1].values, df[ds2].values)
                    result['R_'+ds1+'_'+ds2] = R
                    result['p_'+ds1+'_'+ds2] = p

            tc1 = TCA_calc(df[['GLEAM','ASCAT','AMSR2']], ref_ind=0)
            tc2 = TCA_calc(df[['GLEAM','ASCAT','SMAP']], ref_ind=0)

            for i,ds in enumerate(['GLEAM','ASCAT','AMSR2']):
                result['TC1_R2_'+ds] = tc1[0][i]
                result['TC1_RMSE_'+ds] = tc1[1][i]

            for i,ds in enumerate(['GLEAM','ASCAT','SMAP']):
                result['TC2_R2_'+ds] = tc2[0][i]
                result['TC2_RMSE_'+ds] = tc2[1][i]

            write_output(fname, result, gpi)

            print('gpi %i finished (%i / %i).' % (gpi, cnt+1, len(np.atleast_1d(gpis))))

        except:
            continue

def calc_autocorr(gpis):

    outpath = Path('/work/GLEAM/autocorrelation')

    if not outpath.exists():
        outpath.mkdir(parents=True)

    fname = outpath / ('part_%i.csv' % gpis[0])

    params = {'nens': 1}
    gleam = GLEAM(params)

    for cnt, gpi in enumerate(np.atleast_1d(gpis)):

        data = gleam.proc_ol(gpi)

        result ={'gamma': np.corrcoef(data['w1'][0:-1, 0], data['w1'][1::, 0])[0, 1]}

        write_output(fname, result, gpi)

        print('gpi %i finished (%i / %i).' % (gpi, cnt+1, len(np.atleast_1d(gpis))))


if __name__=='__main__':

    # proc(calc_errors, procs=30)
    # merge_files('/work/GLEAM/errors')
    fill_error_gaps()