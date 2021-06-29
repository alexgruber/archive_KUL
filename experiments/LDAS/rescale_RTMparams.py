
import pandas as pd
import numpy as np

from pathlib import Path
from netCDF4 import Dataset

from pyldas.interface import LDAS_io

from pyldas.templates import get_template

def create_M36_M09_lut():
    ''' Create a NN look-up table from the M09 to the M36 grid'''

    fout = '/Users/u0116961/data_sets/GEOSldas_runs/LUT_M36_M09_US.csv'

    fname36 = '/Users/u0116961/data_sets/GEOSldas_runs/NLv4_M36_US_SMAP_TB_OL.ldas_tilecoord.bin'
    fname09 = '/Users/u0116961/data_sets/GEOSldas_runs/US_M09_SMAP_OL.ldas_tilecoord.bin'

    io = LDAS_io(exp='US_M36_SMAP_OL')
    dtype, hdr, length = get_template('tilecoord')

    tc36 = io.read_fortran_binary(fname36, dtype, hdr=hdr, length=length, reg_ftags=True)
    tc09 = io.read_fortran_binary(fname09, dtype, hdr=hdr, length=length, reg_ftags=True)

    tc36['ind09']= -9999

    for idx, data in tc36.iterrows():
        print('%i / %i' % (idx, len(tc36)))
        tc36.loc[idx, 'ind09'] = np.argmin((tc09.com_lat-data.com_lat)**2 + (tc09.com_lon-data.com_lon)**2)

    tc36['ind09'].to_csv(fout)


def resample_RTMparams():

    # read NN-based LUT
    lut = pd.read_csv('/Users/u0116961/data_sets/GEOSldas_runs/LUT_M36_M09_US.csv', index_col=0)

    fin = '/Users/u0116961/data_sets/GEOSldas_runs/config/RTM_params/RTMParam_SMAP_L4SM_v004/SMAP_EASEv2_M09/mwRTM_param.nc4'
    fout = Path('/Users/u0116961/data_sets/GEOSldas_runs/config/RTM_params/RTMParam_SMAP_L4SM_v004/SMAP_EASEv2_M36/mwRTM_param_US.nc4')
    if not fout.parent.exists():
        Path.mkdir(fout.parent, parents=True)

    # Create new M36 netCDF file w. RTM parameters based on the M09 NNs
    with Dataset(fin, mode='r') as ds09:
        with Dataset(fout, mode='w') as ds36:
            ds36.createDimension('tile', len(lut))
            for var in ds09.variables.keys():
                print(var)
                ds36.createVariable(var, ds09[var].dtype,
                                    dimensions=('tile',),
                                    zlib=True)
                ds36[var].setncatts({'long_name': ds09[var].getncattr('long_name'),
                                     'units': ds09[var].getncattr('units')})
                ds36[var][:] = ds09[var][lut.ind09.values]



if __name__=='__main__':

    create_M36_M09_lut()
    resample_RTMparams()


