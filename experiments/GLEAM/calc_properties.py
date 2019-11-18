
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from copy import deepcopy

from pathlib import Path
from multiprocessing import Pool

from netCDF4 import Dataset, num2date

from pygleam_ag.GLEAM_model import GLEAM
from pygleam_ag.GLEAM_IO import output_ts
from pygleam_ag.grid import get_valid_gpis, read_grid, find_nearest_gpi

import pygeogrids.netcdf as ncgrid
from smecv.common_format import CCIDs

from scipy.stats import pearsonr
from validation_good_practice.ancillary.metrics import TCA_calc

from myprojects.functions import merge_files

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from scipy.ndimage import gaussian_filter

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from datetime import datetime


# 258128
# 338732

def proc(function, procs = 30):
    ''' Wrapper routine for parallelized processing '''

    gpis_valid = get_valid_gpis(latmin=24., latmax=51., lonmin=-128., lonmax=-64.)

    gpis_proc = set(pd.read_csv('/work/GLEAM/perturbation_correction_test_v5/result_incomplete.csv', index_col=0).index.values)
    gpis_valid = np.array(list(set(gpis_valid).difference(gpis_proc)))

    if procs == 1:
        function(gpis_valid)

    else:
        subs = (np.arange(procs + 1) * len(gpis_valid) / procs).astype('int')
        subs[-1] = len(gpis_valid)
        gpi_subs = [gpis_valid[subs[i]:subs[i+1]] for i in np.arange(procs)]
        gpi_subs = [subs for subs in gpi_subs if len(subs) > 0]

        if len(gpi_subs) > 0:
            p = Pool(procs)
            p.map(function, gpi_subs)


def write_output(fname,out_dict,gpi,precision=4):

    output = pd.DataFrame(out_dict, index=(gpi,))

    f = '%0.' + str(precision) + 'f'

    if not fname.exists():
        output.to_csv(fname, float_format=f)
    else:
        output.to_csv(fname, float_format=f, mode='a', header=False)



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

    # fname = outpath / ('part_%i.csv' % gpis[0])

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
                               index=num2date(gleam_io['time'][:], units=gleam_io['time'].units), name='GLEAM')

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

            # write_output(fname, result, gpi)

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

def run_gleam(arg):

    gleam, gpi = arg
    np.random.seed()
    return gleam.proc_ol(gpi)['w1'].var(axis=1).mean()

def calc_pert_corr_v2(gpis):

    outpath = Path('/work/GLEAM/perturbation_correction_v2')

    if not outpath.exists():
        outpath.mkdir(parents=True)

    fname = outpath / ('part_%i.csv' % np.atleast_1d(gpis)[0])

    n_steps = 10
    n_ens = 25

    gleam_det = GLEAM({'nens': 1})
    gleam = GLEAM({'nens': n_ens})

    # perturbation - ensemble variance regression function
    fct = lambda xx, a, b, c: a * xx ** b + c

    for cnt, gpi in enumerate(np.atleast_1d(gpis)):
        print('processing gpi %i (%i / %i).' % (gpi, cnt + 1, len(np.atleast_1d(gpis))))

        # reset random seed for parallelization
        np.random.seed()

        max_var = np.nanvar(gleam_det.proc_ol(gpi)['w1'])
        perts = np.linspace(0, max_var, n_steps+1)[1::]

        # ens_vars = np.full(n_steps, np.nan)
        # for i, pert in enumerate(perts):
        #     t1 = datetime.now()
        #     gleam.mod_pert = {'w1': ['normal', 'additive', pert]}
        #     ens_vars[i]  = gleam.proc_ol(gpi)['w1'].var(axis=1).mean()
        #
        #     print('%i / %i' % (i, n_steps))
        #     print('%2fs' % (datetime.now() - t1).total_seconds())


        # # parallelized version for testing of individual GPIs
        p = Pool(n_steps)
        mods = [GLEAM(params={'nens': n_ens, 'mod_pert': {'w1': ['normal', 'additive', pert]}}) for pert in perts]
        gpis = [gp for gp in np.repeat(gpi,n_steps)]
        args = zip(mods,gpis)
        ens_vars = np.array(p.map(run_gleam,args))


        pl,pu = np.percentile(ens_vars, [5,95])
        ind = np.where((ens_vars>pl)&(ens_vars<pu))

        try:
            (a1, b1, c1), cov1 = curve_fit(fct, perts, ens_vars, [1,1,0])
        except:
            a1, b1, c1 = (np.nan,np.nan,np.nan); cov1 = np.full((3,3),np.nan)
        try:
            (a2, b2, c2), cov2 = curve_fit(fct, perts[ind], ens_vars[ind], [1,1,0])
        except:
            a2, b2, c2 = (np.nan,np.nan,np.nan); cov2 = np.full((3, 3), np.nan)

        result = {'a1': a1, 'b1' : b1, 'c1' : c1, 'c_a1': cov1[0,0], 'c_b1': cov1[1,1], 'c_c1': cov1[2,2],
                  'a2': a2, 'b2' : b2, 'c2' : c2, 'c_a2': cov2[0,0], 'c_b2': cov2[1,1], 'c_c2': cov2[2,2]}

        # write_output(fname, result, gpi, precision=8)

        plt.figure(figsize=(10, 8))
        perts_corr = a1 * perts ** b1 + c1
        perts_corr2 = a2 * perts ** b2 + c2
        plt.plot([0, max_var], [0, max_var], '--k', linewidth=3)
        plt.xlim(0, max_var + max_var / 20.)
        # plt.ylim(0, np.nanmax(ens_vars)+ np.nanmax(ens_vars) / 20.)
        plt.ylim(0, 0.020)
        plt.plot(perts, ens_vars, 'or', linewidth=2)
        # plt.plot(perts, perts_corr, '--b', linewidth=2)
        # plt.plot(perts, perts_corr2, '--g', linewidth=2)
        # plt.title('%i ens %i steps' % (n_ens, n_steps))
        plt.xlabel('model perturbation')
        plt.ylabel('ensemble variance')
        plt.tight_layout()
        plt.show()


def calc_pert_corr(gpis):

    outpath = Path('/work/GLEAM/perturbation_correction')

    if not outpath.exists():
        outpath.mkdir(parents=True)

    fname = outpath / ('part_%i.csv' % np.atleast_1d(gpis)[0])

    params = {'nens': 1}
    gleam_det = GLEAM(params)

    nens = 100
    params = {'nens': nens}
    gleam = GLEAM(params)

    fct = lambda xx, a, b: a * xx ** b

    for cnt, gpi in enumerate(np.atleast_1d(gpis)):

        det = gleam_det.proc_ol(gpi)['w1']
        max_var = np.nanvar(det) * 1

        pert = np.linspace(0, max_var, nens)
        gleam.mod_pert = {'w1': ['normal', 'additive', pert]}
        res = gleam.proc_ol(gpi)['w1']

        ens_vars = np.array([(res[:, 0] - res[:, i]).var() for i in np.arange(res.shape[1])])

        pl,pu = np.percentile(ens_vars, [5,95])
        ind = np.where((ens_vars>pl)&(ens_vars<pu))

        (a1, b1), cov1 = curve_fit(fct, pert, ens_vars)
        (a2, b2), cov2 = curve_fit(fct, pert[ind], ens_vars[ind])

        result = {'a1': a1, 'b1' : b1, 'c_a1': cov1[0,0], 'c_b1': cov1[1,1], 'c_ab1': cov1[0,1],
                  'a2': a2, 'b2' : b2, 'c_a2': cov2[0,0], 'c_b2': cov2[1,1], 'c_ab2': cov2[0,1]}

        # write_output(fname, result, gpi, precision=8)

        plt.figure(figsize=(10, 8))
        perts_corr = a * pert ** b
        perts_corr2 = a2 * pert ** b2
        plt.plot([0, max_var], [0, max_var], '--k', linewidth=3)
        plt.xlim(0, max_var + max_var / 20.)
        plt.ylim(0, np.nanmax(ens_vars)+ np.nanmax(ens_vars) / 20.)
        plt.plot(pert, ens_vars, 'or', linewidth=2)
        # plt.plot(pert, perts_corr, '--b', linewidth=2)
        # plt.plot(pert, perts_corr2, '--g', linewidth=2)
        plt.tight_layout()
        plt.show()

        print('gpi %i finished (%i / %i).' % (gpi, cnt + 1, len(np.atleast_1d(gpis))))


def smooth_pert_corr():

    res = pd.read_csv('/work/GLEAM/perturbation_correction_v2/result.csv', index_col=0)

    gpis_valid = get_valid_gpis(latmin=24., latmax=51., lonmin=-128., lonmax=-64.)
    ind_valid = np.unravel_index(gpis_valid, (720, 1440))

    imp = IterativeImputer(max_iter=10, random_state=0)
    ind = np.unravel_index(res.index.values, (720, 1440))
    for tag in ['a1', 'b1', 'c1','a2', 'b2', 'c2']:

        img = np.full((720, 1440), np.nan)
        img[ind] = res[tag]

        # find all non-zero values
        idx = np.where(~np.isnan(img))
        vmin, vmax = np.percentile(img[idx], [2.5, 97.5])
        img[img < vmin] = vmin
        img[img > vmax] = vmax

        # calculate fitting parameters
        imp.set_params(min_value=vmin, max_value=vmax)
        imp.fit(img)

        # Define an anchor pixel to infer fitted image dimensions
        tmp_img = img.copy()
        tmp_img[idx[0][100], idx[1][100]] = 1000000

        # transform image with and without anchor pixel
        tmp_img_fitted = imp.transform(tmp_img)
        img_fitted = imp.transform(img)

        # # Get indices of fitted image
        idx_anchor = np.where(tmp_img_fitted == 1000000)[1][0]
        start = idx[1][100] - idx_anchor
        end = start + img_fitted.shape[1]

        # write output
        img[:, start:end] = img_fitted
        img = gaussian_filter(img, sigma=0.6, truncate=1)

        res.loc[:, tag + '_s'] = img[ind_valid]

    res.to_csv('/work/GLEAM/perturbation_correction_v2/result_smoothed.csv', float_format='%.8f')



def test_pert_corr(gpis):

    outpath = Path('/work/GLEAM/perturbation_correction_test_v5')

    if not outpath.exists():
        outpath.mkdir(parents=True)

    fname = outpath / ('part_%i.csv' % np.atleast_1d(gpis)[0])

    pert = pd.read_csv('/work/GLEAM/errors/result_gapfilled_sig07.csv', index_col=0)['TC2_RMSE_GLEAM'] ** 2 / 100 ** 2
    corr = pd.read_csv('/work/GLEAM/perturbation_correction_v2/result.csv', index_col=0)

    pert_corr = ((pert - corr['c2']) / corr['a2']) ** (1 / corr['b2'])
    pert_corr.loc[np.isnan(pert_corr)] = pert_corr.median()

    params = {'nens': 25}
    gleam = GLEAM(params)

    for cnt, gpi in enumerate(np.atleast_1d(gpis)):

        gleam.mod_pert = {'w1': ['normal', 'additive', pert_corr.loc[gpi]]}
        res = gleam.proc_ol(gpi)['w1']

        result = {'pert': pert.loc[gpi], 'pert_corr': pert_corr.loc[gpi], 'avg_ens_var': res.var(axis=1).mean()}

        write_output(fname, result, gpi, precision=8)

        print('gpi %i finished (%i / %i).' % (gpi, cnt + 1, len(np.atleast_1d(gpis))))


if __name__=='__main__':

    proc(test_pert_corr, procs=30)

    # merge_files('/work/GLEAM/errors')
    # fill_error_gaps()

    # proc(calc_pert_corr_v2, procs=30)

    # smooth_pert_corr()

    # lat = 35.062078
    # lon = -117.258583

    # lat = 41.299531
    # lon = -117.400013

    # gpi = find_nearest_gpi(lat,lon)

    # test_pert_corr(259498)
