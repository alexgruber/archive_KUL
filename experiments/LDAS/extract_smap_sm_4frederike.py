import warnings
warnings.filterwarnings("ignore")

import h5py
import platform

import numpy as np
import pandas as pd
from pathlib import Path

from pyldas.interface import LDAS_io
from pyldas.templates import get_template

def extract_smap_ldas_data():

    # Path specifications
    if platform.system() == 'Darwin':
        pointlist_file = '/Users/u0116961/data_sets/SMAP/4frederike/tileind_id_lat_lon_Dry_Chaco_tight.txt'
        smap_root = Path('/Users/u0116961/data_sets/SMAP/SPL2SMP.006')
        ldas_root_tc = Path('/Users/u0116961/data_sets/LDASsa_runs/GLOB_M36_7Thv_TWS_ensin_FOV0_M2/output/SMAP_EASEv2_M36_US/rc_out')
        ldas_root_data = Path('/Users/u0116961/data_sets/SMAP/4frederike')
        out_path = Path('/Users/u0116961/data_sets/SMAP/4frederike')
    else:
        pointlist_file = '/staging/leuven/stg_00024/MSC_TMP/frederike/RTM/RTM_CALI/tileind_id_lat_lon_Dry_Chaco_tight.txt'
        smap_root = Path('/staging/leuven/stg_00024/l_data/obs_satellite/SMAP/SPL2SMP.006')
        ldas_root_tc = Path('/scratch/leuven/314/vsc31402/output/GLOB_M36_7Thv_TWS_ensin_FOV0_M2/output/SMAP_EASEv2_M36_GLOB/rc_out')
        ldas_root_data = Path('/scratch/leuven/314/vsc31402/output/GLOB_M36_7Thv_TWS_ensin_FOV0_M2/output/SMAP_EASEv2_M36_GLOB/cat/ens_avg')
        out_path = Path('/staging/leuven/stg_00024/OUTPUT/alexg/4frederike')
    fbase = 'GLOB_M36_7Thv_TWS_ensin_FOV0_M2_run_1.ens_avg.ldas_tile_inst_out'

    # Get SMAP data path and observation dates
    files = sorted(smap_root.glob('**/*.h5'))
    dates = pd.to_datetime([str(f)[-29:-14] for f in files]).round('3h') # timedelta b/c wrongly assigned in LDAS!!

    # Setup LDAS interface and get tile_coord LUT
    io = LDAS_io()
    dtype, _, _ = get_template('xhourly_inst')
    tc_file = ldas_root_tc /  'GLOB_M36_7Thv_TWS_ensin_FOV0_M2.ldas_tilecoord.bin'
    tc = io.read_params('tilecoord', fname=tc_file)
    n_tiles = len(tc)

    # Extract only the relevant domain, as specified by GdL
    plist = pd.read_csv(pointlist_file, names=['tile_idx', 'tile_id', 'lat', 'lon'], sep='\t')
    tc = tc.reindex(plist.tile_idx.values)

    # Create empty array to be filled w. SMAP and LDAS data
    res_arr = np.full((len(dates), len(tc), 2), np.nan)

    ind_valid = [] # Keep only dates with valid data!
    for dt, (f, date) in enumerate(zip(files, dates)):
        print(f'Extracting date {dt} / {len(files)}...')
        with h5py.File(f, mode='r') as arr:
            qf = arr['Soil_Moisture_Retrieval_Data']['retrieval_qual_flag'][:]
            idx = np.where((qf == 0) | (qf == 8))
            row = arr['Soil_Moisture_Retrieval_Data']['EASE_row_index'][idx]
            col = arr['Soil_Moisture_Retrieval_Data']['EASE_column_index'][idx]
            sm = arr['Soil_Moisture_Retrieval_Data']['soil_moisture'][idx]

            rowcols_smap = [f'{r:03d}{c:03d}' for r, c in zip(row, col)]
            rowcols_list = [f'{r:03d}{c:03d}' for r, c in zip(tc.j_indg, tc.i_indg)]

            ind_dict_smap = dict((i, j) for j, i in enumerate(rowcols_smap))
            ind_dict_list = dict((i, j) for j, i in enumerate(rowcols_list))
            inter = set(rowcols_smap).intersection(set(rowcols_list))
            if len(inter) > 0:
                inds_smap = np.array([ind_dict_smap[x] for x in inter])
                inds_list = np.array([ind_dict_list[x] for x in inter])
                srt = np.argsort(inds_smap)

                fname = ldas_root_data / f'Y{date.year}' / f'M{date.month:02d}' / f'{fbase}.{date.strftime("%Y%m%d_%H%Mz.bin")}'
                if fname.exists():
                    data_ldas = io.read_fortran_binary(fname, dtype=dtype, length=n_tiles)
                    data_ldas.index += 1

                    ind_valid.append(dt)
                    res_arr[dt, inds_list[srt], 0] = sm[inds_smap[srt]]
                    res_arr[dt, inds_list[srt], 1] = data_ldas.reindex(tc.iloc[inds_list[srt]].index)['sm_surface'].values

    res_arr = res_arr[ind_valid, :, :]
    dates = dates[ind_valid]

    pd.Series(dates.to_julian_date(), index=dates).to_csv(out_path / 'dates.csv', header=None)
    np.save(out_path / 'soil_moisture_smap_ldas', res_arr)


if __name__=='__main__':
    extract_smap_ldas_data()