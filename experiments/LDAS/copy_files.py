

import os
import numpy as np

from pathlib import Path

def copy_files():

    # exps = 'DA_it532'
    exps = ['OL_it532', 'OL_it533', 'OL_it61', 'OL_it62']

    domain = 'SMAP_EASEv2_M36_US'

    root_mac = Path('/Users/u0116961/data_sets/LDASsa_runs')

    # root_vsc = Path('/scratch/leuven/320/vsc32046/output/TEST_RUNS')
    root_vsc = Path('/staging/leuven/stg_00024/OUTPUT/alexg')

    copy_ts = True
    copy_img = False

    replace_nc = False

    for exp in np.atleast_1d(exps):

        exp_root_mac = root_mac / ('US_M36_SMOS40_TB_MadKF_' + exp)
        exp_root_vsc = root_vsc / ('US_M36_SMOS40_TB_MadKF_' + exp) / 'output' / domain

        if not exp_root_mac.exists():
            Path.mkdir(exp_root_mac / 'ana', parents=True)
            Path.mkdir(exp_root_mac / 'cat')

        if not list(exp_root_mac.glob('rc_out')):
            os.system('scp -r vscg:' + str(exp_root_vsc / 'rc_out') + ' ' + str(exp_root_mac))

        if copy_ts:
            if (not list(exp_root_mac.glob('**/xhourly_timeseries.nc'))) | replace_nc:
                os.system('scp vscg:' + str(exp_root_vsc / 'cat' / 'xhourly_timeseries.nc') + ' ' + str(exp_root_mac / 'cat'))
            if (not list(exp_root_mac.glob('**/ObsFcstAna_timeseries.nc'))) | replace_nc:
                os.system('scp vscg:' + str(exp_root_vsc / 'ana' / 'ObsFcstAna_timeseries.nc') + ' ' + str(exp_root_mac / 'ana'))
            if (not list(exp_root_mac.glob('**/ObsFcstAnaEns_timeseries.nc'))) | replace_nc:
                os.system('scp vscg:' + str(exp_root_vsc / 'ana' / 'ObsFcstAnaEns_timeseries.nc') + ' ' + str(exp_root_mac / 'ana'))

        if copy_img:
            if (not list(exp_root_mac.glob('**/xhourly_images.nc'))) | replace_nc:
                os.system('scp vscg:' + str(exp_root_vsc / 'cat' / 'xhourly_images.nc') + ' ' + str(exp_root_mac / 'cat'))
            if (not list(exp_root_mac.glob('**/ObsFcstAna_images.nc'))) | replace_nc:
                os.system('scp vscg:' + str(exp_root_vsc / 'ana' / 'ObsFcstAna_images.nc') + ' ' + str(exp_root_mac / 'ana'))
            if (not list(exp_root_mac.glob('**/ObsFcstAnaEns_images.nc'))) | replace_nc:
                os.system('scp vscg:' + str(exp_root_vsc / 'ana' / 'ObsFcstAnaEns_images.nc') + ' ' + str(exp_root_mac / 'ana'))


if __name__=='__main__':
    copy_files()