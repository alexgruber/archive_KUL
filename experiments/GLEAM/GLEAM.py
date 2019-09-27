
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from copy import deepcopy
from multiprocessing import Pool

from myprojects.experiments.GLEAM.GLEAM_model import GLEAM
from myprojects.experiments.GLEAM.GLEAM_IO import output_ts
from myprojects.experiments.GLEAM.grid import get_valid_gpis


def main():

    parts = 20

    # get valid gpis within CONUS
    gpis_valid = get_valid_gpis(latmin=24., latmax=51., lonmin=-128., lonmax=-64.)

    # split gpis into subsets for parallel processing
    subs = (np.arange(parts + 1) * len(gpis_valid) / parts).astype('int')
    subs[-1] = len(gpis_valid)
    gpi_subs = [gpis_valid[subs[i]:subs[i+1]] for i in np.arange(parts)]

    p = Pool(parts)
    p.map(run, gpi_subs)

    # run(gpis_valid)


def run(gpis):

    outpath = '/data_sets/GLEAM/_output'

    # initialize model with specific configuration
    gleam = GLEAM()

    # initialize output variable
    out_vars = [ 'E', 'w1', 'w2', 'w3']
    res_template = dict()
    for var in out_vars:
        res_template[var] = np.full((len(gleam.period),gleam.nens),np.nan)

    # process grid cells
    for cnt, gpi in enumerate(gpis):

        # initialize model generator
        g = gleam.proc(gpi)

        res = deepcopy(res_template)
        for t,_ in enumerate(gleam.period):

            # model time step
            sv = next(g)

            # fill up output variable
            for var in out_vars:
                res[var][t, :] = sv[var][0,:,0,:]

                # TODO: implement state updating here

        # write output to disc
        output_ts(res, outpath, gpi, out_vars, gleam.period, gleam.nens)

        print('gpi %i finished (%i / %i).' % (gpi, cnt+1, len(gpis)))


if __name__=='__main__':

    main()