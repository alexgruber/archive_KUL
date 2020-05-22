import warnings
warnings.filterwarnings("ignore")

import logging

import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

import matplotlib.pyplot as plt

from pyldas.grids import EASE2
from pyldas.interface import LDAS_io
from pyldas.templates import template_scaling
from pyldas.visualize.plots import plot_ease_img

from myprojects.timeseries import calc_clim_harmonic, calc_pentadal_mean_std

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def calc_clim_p(ts, mode='pentadal', n=3):

    if mode == 'pentadal':
        return calc_pentadal_mean_std(ts)
    else:
        clim = calc_clim_harmonic(ts, n=n)
        pentads = np.floor((clim.index.values - 1) / 5.)
        clim = clim.groupby(pentads,axis=0).mean()
        clim.index = np.arange(73)+1
        return clim


def run(sensor, date_from=None, date_to=None, mode='longterm'):
    '''
    :param sensor: 'SMOS' or 'SMAP' or 'SMOSSMAP'
    :param date_from: 'yyyy-mm-dd'
    :param date_to: 'yyyy-mm-dd'
    :param mode: 'longterm' or "shortterm'
    '''

    exp_smos = 'US_M36_SMOS40_TB_OL_noScl'
    exp_smap = 'US_M36_SMAP_TB_OL_noScl'

    froot = Path(f'/Users/u0116961/data_sets/LDASsa_runs/scaling_files/{sensor}')

    if not froot.exists():
        Path.mkdir(froot)

    ios = []
    if 'SMOS' in sensor:
        ios += [LDAS_io('ObsFcstAna', exp=exp_smos)]
    if 'SMAP' in sensor:
        ios += [LDAS_io('ObsFcstAna', exp=exp_smap)]

    if not date_from:
        date_from = pd.to_datetime(np.min([io.timeseries['time'].values[0] for io in ios]))
    else:
        date_from = pd.to_datetime(date_from)
    if not date_to:
        date_to = pd.to_datetime(np.max([io.timeseries['time'].values[-1] for io in ios]))
    else:
        date_to = pd.to_datetime(date_to)

    pent_from = int(np.floor((date_from.dayofyear - 1) / 5.) + 1)
    pent_to = int(np.floor((date_to.dayofyear - 1) / 5.) + 1)
    fbase = f'Thvf_TbSM_001_{sensor}_zscore_stats_{date_from.year}_p{pent_from:02}_{date_to.year}_p{pent_to:02}_hscale_0.00_W_9p_Nmin_20'

    dtype, _, _ = template_scaling(sensor='SMOS40')

    tiles = ios[0].grid.tilecoord['tile_id'].values.astype('int32')
    angles = np.array([40,], 'int')
    pols = ['H','V']
    orbits = [['A', 'D'],['D', 'A']] # To match SMOS and SMAP species!

    template = pd.DataFrame(columns=dtype.names, index=tiles).astype('float32')
    template['lon'] = ios[0].grid.tilecoord['com_lon'].values.astype('float32')
    template['lat'] = ios[0].grid.tilecoord['com_lat'].values.astype('float32')
    template['tile_id'] = tiles.astype('int32')

    pentads = np.arange(73)+1

    if mode == 'longterm':
        dummy = np.full([len(tiles),len(pentads),len(angles),len(pols),len(orbits[0])],-9999)
        coords = {'tile_id': tiles,
                  'pentad': pentads,
                  'angle': angles,
                  'pol': pols,
                  'orbit': orbits[0]}
        darr = xr.DataArray(dummy, coords=coords, dims=['tile_id','pentad','angle','pol','orbit'])
    else:
        years = np.arange(2010, 2017)
        dummy = np.full([len(tiles), len(pentads), len(years), len(angles), len(pols), len(orbits[0])], -9999)
        coords = {'tile_id': tiles,
                  'pentad': pentads,
                  'year': years,
                  'angle': angles,
                  'pol': pols,
                  'orbit': orbits[0]}
        darr = xr.DataArray(dummy, coords=coords, dims=['tile_id', 'pentad', 'year', 'angle', 'pol', 'orbit'])

    data = xr.Dataset({'m_obs':darr.astype('float32'),
                       's_obs': darr.astype('float32'),
                       'm_mod':darr.astype('float32'),
                       's_mod': darr.astype('float32'),
                       'N_data':darr.astype('int32')})

    # ----- calculate mean and reshuffle -----
    for i,til in enumerate(tiles):
        logging.info(f'{i} / {len(tiles)}')
        for pol in pols:
            for ang in angles:
                for orb1, orb2 in zip(orbits[0], orbits[1]):

                    col, row = ios[0].grid.tileid2colrow(til)
                    if sensor.upper() == 'SMOSSMAP':
                        spcs = [io.get_species(pol=pol, ang=ang, orbit=orb) for io, orb in zip(ios,[orb1, orb2])]
                        orb = orb2 # because they are used for SMAP rescaling!
                    else:
                        spcs = [ios[0].get_species(pol=pol, ang=ang, orbit=orb1)]
                        orb = orb1 # If sensors are not combined, orb1 is always the correct one!

                        # obs = io.timeseries['obs_obs'][:, spc-1, row, col].to_series()
                        # mod = io.timeseries['obs_fcst'][:, spc-1, row, col].to_series()
                    obs = pd.concat([io.timeseries['obs_obs'][:, spc-1, row, col].to_series() for io, spc in zip(ios,spcs)]).sort_index()
                    mod = pd.concat([io.timeseries['obs_fcst'][:, spc-1, row, col].to_series() for io, spc in zip(ios,spcs)]).sort_index()

                    if mode == 'longterm':
                        data['m_obs'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb)[:],\
                        data['s_obs'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb)[:] = calc_clim_p(obs[date_from:date_to])
                        data['m_mod'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb)[:],\
                        data['s_mod'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb)[:] = calc_clim_p(mod[date_from:date_to])
                        data['N_data'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb)[:] = len(obs[date_from:date_to].dropna())
                    else:
                        for yr in years:
                            data['m_obs'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb, year=yr)[:],\
                            data['s_obs'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb, year=yr)[:] = calc_clim_p(obs[date_from:date_to][obs.index.year==yr]).values
                            data['m_mod'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb, year=yr)[:],\
                            data['s_mod'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb, year=yr)[:] = calc_clim_p(mod[date_from:date_to][obs.index.year==yr]).values
                            data['N_data'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb, year=yr)[:] = len(obs[obs[date_from:date_to].index.year==yr].dropna())

    modes = np.array([0, 0])
    sdate = np.array([date_from.year, date_from.month, date_from.day, 0, 0])
    edate = np.array([date_to.year, date_to.month, date_to.day, 0, 0])
    lengths = np.array([len(tiles), len(angles), 1])  # tiles, incidence angles, whatever

    # ----- write output files -----
    for pent in pentads:
        for orb in orbits[0]:
            # !!! inconsistent with the definition in the obs_paramfile (species) !!!
            modes[0] = 1 if orb == 'A' else 0

            if mode == 'longterm':
                res = template.copy()
                for ang in angles:
                    for pol in pols:
                        res.loc[:, f'm_obs_{pol}_{ang}'] = data['m_obs'].sel(pol=pol, angle=ang, orbit=orb, pentad=pent).to_series()
                        res.loc[:, f's_obs_{pol}_{ang}'] = data['s_obs'].sel(pol=pol, angle=ang, orbit=orb, pentad=pent).to_series()
                        res.loc[:, f'm_mod_{pol}_{ang}'] = data['m_mod'].sel(pol=pol, angle=ang, orbit=orb, pentad=pent).to_series()
                        res.loc[:, f's_mod_{pol}_{ang}'] = data['s_mod'].sel(pol=pol, angle=ang, orbit=orb, pentad=pent).to_series()
                        res.loc[:, f'N_data_{pol}_{ang}'] = data['N_data'].sel(pol=pol, angle=ang, orbit=orb, pentad=pent).to_series()
                res.replace(np.nan, -9999, inplace=True)
                fname = froot / f'{fbase}_{orb}_p{pent:02}.bin'
                fid = open(fname, 'wb')
                ios[0].write_fortran_block(fid, modes)
                ios[0].write_fortran_block(fid, sdate)
                ios[0].write_fortran_block(fid, edate)
                ios[0].write_fortran_block(fid, lengths)
                ios[0].write_fortran_block(fid, angles)
                for f in res.columns.values:
                    ios[0].write_fortran_block(fid, res[f].values)
                fid.close()
            else:
                for yr in years:
                    res = template.copy()
                    for ang in angles:
                        for pol in pols:
                            res.loc[:, f'm_obs_{pol}_{ang}'] = data['m_obs'].sel(pol=pol, angle=ang, orbit=orb, pentad=pent, year=yr).to_series()
                            res.loc[:, f's_obs_{pol}_{ang}'] = data['s_obs'].sel(pol=pol, angle=ang, orbit=orb, pentad=pent, year=yr).to_series()
                            res.loc[:, f'm_mod_{pol}_{ang}'] = data['m_mod'].sel(pol=pol, angle=ang, orbit=orb, pentad=pent, year=yr).to_series()
                            res.loc[:, f's_mod_{pol}_{ang}'] = data['s_mod'].sel(pol=pol, angle=ang, orbit=orb, pentad=pent, year=yr).to_series()
                            res.loc[:, f'N_data_{pol}_{ang}'] = data['N_data'].sel(pol=pol, angle=ang, orbit=orb, pentad=pent, year=yr).to_series()
                    res.replace(np.nan, -9999, inplace=True)
                    fname = froot / f'{fbase}_{orb}_p{pent:02}_y{yr:04}.bin'
                    fid = open(fname, 'wb')
                    ios[0].write_fortran_block(fid, modes)
                    ios[0].write_fortran_block(fid, sdate)
                    ios[0].write_fortran_block(fid, edate)
                    ios[0].write_fortran_block(fid, lengths)
                    ios[0].write_fortran_block(fid, angles)
                    for f in res.columns.values:
                        ios[0].write_fortran_block(fid, res[f].values)
                    fid.close()


def replace_orbit_field():
    '''
    This was used to correct for a bug in the run() routine,
    which should be fixed by now, so I THINK (but not sure) this is obsolete.
    '''
    root = r'C:\Users\u0116961\Documents\VSC\vsc_data_copies\scratch_TEST_RUNS\US_M36_SMOS_noDA_unscaled\obs_scaling'
    for f in find_files(root,'_D_p'):
        data = np.fromfile(f,'>i4')
        data[1] = 0
        data.tofile(f)

if __name__ == '__main__':

    # 'SMOS' / 'SMAP' / 'SMOSSMAP'
    sensor = 'SMOSSMAP'

    run(sensor, date_from='2015-04-01', date_to='2019-12-31')
