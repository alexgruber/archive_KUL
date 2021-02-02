
import os
import logging

import numpy as np
import pandas as pd

from pathlib import Path

from pyldas.interface import LDAS_io
from pyldas.templates import template_error_Tb40

from myprojects.timeseries import calc_anom
from myprojects.experiments.MadKF.CLSM.ensemble_covariance import fill_gaps

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def run():

    anom = False
    longterm = False
    fcst_err_corrected = False

    exp = 'US_M36_SMAP_TB_MadKF_OL_it11'

    io = LDAS_io('ObsFcstAna', exp)

    froot = Path('/Users/u0116961/Documents/work/MadKF/CLSM/SMAP/rmsd_pert/error_files')
    fbase = 'SMOS_fit_Tb_'

    dir_out = froot / ((('anom_' + ('lt' if longterm else 'st')) if anom else 'abs') + ('_fcst_corr' if fcst_err_corrected else '_uncorr'))
    if not dir_out.exists():
        Path.mkdir(dir_out, parents=True)

    dtype = template_error_Tb40()[0]

    angles = np.array([40.,])
    orbits = ['A', 'D']

    tiles = io.grid.tilecoord['tile_id'].values.astype('int32')
    ind_lat = io.grid.tilecoord.loc[:, 'j_indg'].values - io.grid.tilegrids.loc['domain', 'j_offg']
    ind_lon = io.grid.tilecoord.loc[:, 'i_indg'].values - io.grid.tilegrids.loc['domain', 'i_offg']

    template = pd.DataFrame(columns=dtype.names, index=tiles).astype('float32')
    template['lon'] = io.grid.tilecoord['com_lon'].values.astype('float32')
    template['lat'] = io.grid.tilecoord['com_lat'].values.astype('float32')

    modes = np.array([0, 0])
    sdate = np.array([2015, 4, 1, 0, 0])
    edate = np.array([2020, 4, 31, 0, 0])
    lengths = np.array([len(tiles), len(angles)])  # tiles, incidence angles, whatever

    dims = io.timeseries['obs_obs'].shape

    obs_errstd = np.full(dims[1::], np.nan)

    # ----- Calculate anomalies -----
    cnt = 0
    for spc in np.arange(dims[1]):
        for lat in np.arange(dims[2]):
            for lon in np.arange(dims[3]):
                cnt += 1
                logging.info('%i / %i' % (cnt, np.prod(dims[1::])))

                try:
                    if anom:
                        obs = calc_anom(io.timeseries['obs_obs'][:, spc, lat, lon].to_dataframe()['obs_obs'], longterm=longterm)
                        fcst = calc_anom(io.timeseries['obs_fcst'][:, spc, lat, lon].to_dataframe()['obs_fcst'], longterm=longterm)
                    else:
                        obs = io.timeseries['obs_obs'][:, spc, lat, lon].to_dataframe()['obs_obs']
                        fcst = io.timeseries['obs_fcst'][:, spc, lat, lon].to_dataframe()['obs_fcst']

                    fcst_errvar = np.nanmean(io.timeseries['obs_fcstvar'][:, spc, lat, lon].values) if fcst_err_corrected else 0

                    tmp_obs_errstd = (((obs - fcst) ** 2).mean() - fcst_errvar) ** 0.5
                    if not np.isnan(tmp_obs_errstd):
                        obs_errstd[spc, lat, lon] = tmp_obs_errstd

                except:
                    pass

    np.place(obs_errstd, obs_errstd < 0, 0)
    np.place(obs_errstd, obs_errstd > 20, 20)

    # ----- write output files -----
    for orb in orbits:
        # !!! inconsistent with the definition in the obs_paramfile (species) !!!
        modes[0] = 1 if orb == 'A' else 0

        res = template.copy()
        res.index = np.arange(len(res))+1
        res['row'] = ind_lat
        res['col'] = ind_lon

        spc = 0 if orb == 'A' else 1
        res['err_Tbh'] = obs_errstd[spc,ind_lat,ind_lon]

        spc = 2 if orb == 'A' else 3
        res['err_Tbv'] = obs_errstd[spc,ind_lat,ind_lon]

        res = fill_gaps(res, ['err_Tbh', 'err_Tbv'], smooth=False, grid=io.grid)

        fname = os.path.join(dir_out, fbase + orb + '.bin')

        fid = open(fname, 'wb')
        io.write_fortran_block(fid, modes)
        io.write_fortran_block(fid, sdate)
        io.write_fortran_block(fid, edate)
        io.write_fortran_block(fid, lengths)
        io.write_fortran_block(fid, angles)

        for f in res.drop(['row', 'col'], axis='columns').columns.values:
            io.write_fortran_block(fid, res[f].values)
        fid.close()

if __name__ == '__main__':
    run()