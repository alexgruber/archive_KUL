
import os
from pathlib import Path
from netCDF4 import Dataset

def rechunk():

    inpath = '/data_sets/GLEAM/v33b_forcing/'
    outpath = '/data_sets/GLEAM/v33b_forcing_ts_chunked/'

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    for f in list(Path(inpath).glob('**/*.nc')):

        n_days = 12 if (f.name.split('_')[0] == 'LF') else 365
        cmdBase = 'ncks -4 -L 4 --cnk_dmn time,%i --cnk_dmn lat,1 --cnk_dmn lon,1 ' % n_days
        cmd = cmdBase + str(f) + ' ' + outpath + f.name
        os.system(cmd)

if __name__ == '__main__':
    rechunk()




