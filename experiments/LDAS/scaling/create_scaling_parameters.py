import os
os.environ["PROJ_LIB"] = '/Users/u0116961/opt/miniconda3/pkgs/proj4-5.2.0-h6de7cb9_1006/share/proj'

import warnings
warnings.filterwarnings("ignore")

import logging

import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

import matplotlib.pyplot as plt

from pytesmo.temporal_matching import df_match

from pyldas.grids import EASE2
from pyldas.interface import GEOSldas_io
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

def PCA(Ser1, Ser2, window=1):

    df1 = pd.DataFrame(Ser1).dropna(); df1.columns = ['ds1']
    df2 = pd.DataFrame(Ser2).dropna(); df2.columns = ['ds2']

    if (len(df1) < 10) | (len(df2) <= 10):
        return pd.DataFrame(columns=['PC-1', 'PC-2'])

    if len(df1) < len(df2):
        matched = df_match(df1, df2, window=window)
        df = df1.join(matched['ds2']).dropna()
    else:
        matched = df_match(df2, df1, window=window)
        df = df2.join(matched['ds1']).dropna()

    if len(df) < 10:
        return pd.DataFrame(columns=['PC-1', 'PC-2'])

    X = df.values.copy()
    X_mean = X.mean(axis=0)
    X -= X_mean

    C = (X.T @ X) / (len(X)-1)
    eigen_vals, eigen_vecs = np.linalg.eig(C)

    # Rotate Eigenvectors 180 degrees if major PC is pointing in the "wrong" direction.
    if (np.sign(eigen_vecs[:,np.argmax(eigen_vals)]).sum() == -2) & (np.sign(np.corrcoef(X.T)[0,1]) == 1):
        eigen_vecs *= -1

    X_pca = X @ eigen_vecs
    if eigen_vals[0] < eigen_vals[1]:
        X_pca = np.roll(X_pca, 1, axis=1)
    X_pca[:,0] += X_mean.mean()

    df_pca = pd.DataFrame(X_pca, columns=['PC-1', 'PC-2'], index=df.index)
    return pd.concat((df, df_pca), axis='columns'), eigen_vals, eigen_vecs

def run(args, scale_target='SMAP', mode='longterm', use_pc=False):
    '''
    :param args: summarizes the following three for multiprocessing purposes:
        sensor: 'SMOS' or 'SMAP' or 'SMOSSMAP'
        date_from: 'yyyy-mm-dd'
        date_to: 'yyyy-mm-dd'
    :param scale_target: 'SMOS' or 'SMAP'
    :param mode: 'longterm' or "shortterm'
    :param use_pc: If true, the first principal component of SMOS/SMAP Tb will be used
    '''

    sensor, date_from, date_to = args

    pc = 'Pcorr'

    exp_smap = f'NLv4_M36_US_OL_{pc}'
    exp_smos = f'NLv4_M36_US_OL_{pc}_SMOS'

    ext = '_yearly' if mode == 'shortterm' else ''
    froot = Path(f'~/data_sets/GEOSldas_runs/_scaling_files_{pc}{ext}').expanduser()
    if not froot.exists():
        Path.mkdir(froot, parents=True)

    ios = []
    if 'SMAP' in sensor:
        ios += [GEOSldas_io('ObsFcstAna', exp=exp_smap)]
    if 'SMOS' in sensor:
        ios += [GEOSldas_io('ObsFcstAna', exp=exp_smos)]

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
    sub = '_PCA' if (use_pc and (sensor == 'SMOSSMAP')) else ''
    fbase = f'Thvf_TbSM_001_src_{sensor}{sub}_trg_{scale_target}_{date_from.year}_p{pent_from:02}_{date_to.year}_p{pent_to:02}_W_9p_Nmin_20'

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
        years = np.arange(date_from.year, date_to.year+1)
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
                        # orb = orb2 if scale_target == 'SMAP' else orb1 # POSSIBLY WRONG!!!!
                        orb = orb1 if scale_target == 'SMAP' else orb2
                    else:
                        spcs = [ios[0].get_species(pol=pol, ang=ang, orbit=orb1)]
                        if sensor.upper() == 'SMAP':
                            orb = orb1 if scale_target == 'SMAP' else orb2
                        else:
                            orb = orb2 if scale_target == 'SMAP' else orb1

                    if use_pc and (sensor == 'SMOSSMAP'):
                        dss = [io.timeseries['obs_obs'][:, spc-1, row, col].to_series() for io, spc in zip(ios,spcs)]
                        obs = PCA(*dss, window=1.5)['PC-1']
                        dss = [io.timeseries['obs_fcst'][:, spc-1, row, col].to_series() for io, spc in zip(ios,spcs)]
                        mod = PCA(*dss, window=1.5)['PC-1']
                    else:
                        obs = pd.concat([io.timeseries['obs_obs'][:, spc-1, row, col].to_series() for io, spc in zip(ios,spcs)]).sort_index()
                        mod = pd.concat([io.timeseries['obs_fcst'][:, spc-1, row, col].to_series() for io, spc in zip(ios,spcs)]).sort_index()

                    if (len(obs) == 0) | (len(mod) == 0):
                        continue

                    if mode == 'longterm':
                        data['m_obs'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb)[:],\
                        data['s_obs'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb)[:] = calc_clim_p(obs[date_from:date_to])
                        data['m_mod'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb)[:],\
                        data['s_mod'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb)[:] = calc_clim_p(mod[date_from:date_to])
                        data['N_data'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb)[:] = len(obs[date_from:date_to].dropna())
                    else:
                        for yr in years:
                            data['m_obs'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb, year=yr)[:],\
                            data['s_obs'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb, year=yr)[:] = calc_clim_p(obs[obs.index.year==yr][date_from:date_to])
                            data['m_mod'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb, year=yr)[:],\
                            data['s_mod'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb, year=yr)[:] = calc_clim_p(mod[mod.index.year==yr][date_from:date_to])
                            data['N_data'].sel(tile_id=til, pol=pol, angle=ang, orbit=orb, year=yr)[:] = len(obs[obs.index.year==yr][date_from:date_to].dropna())

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
                ios[0].write_fortran_block(fid, angles.astype('float')) # required by LDASsa!!
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
                    ios[0].write_fortran_block(fid, angles.astype('float'))  # required by LDASsa!!
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

def replace_angle_field():
    '''
    Angle needs to be stored as float.... This is to correct wrongly stored SMAP angle information
    '''

    root = Path('/Users/u0116961/data_sets/LDASsa_runs/scaling_files')

    for fname in root.glob('*.bin'):
        data = np.fromfile(fname, '>f4')
        data[24] = 40.0
        data.tofile(fname)

if __name__ == '__main__':

    # 'SMOS' / 'SMAP' / 'SMOSSMAP'
    args = ('SMOSSMAP', '2015-04-01', '2020-04-01')
    run(args, scale_target='SMAP', mode='longterm')

    # replace_angle_field()