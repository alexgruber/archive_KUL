import numpy as np
import xarray as xr
import sys

from pathlib import Path
from collections import OrderedDict
from netCDF4 import Dataset, num2date, date2num

def init(period, path, nens, col, row):

    files = list((path).glob('**/*.nc'))
    datain = {}

    for cnt,f in enumerate(files):
        var = f.name.split('_')[0]
        with Dataset(f) as ds:
            if var == 'LF':
                forcingds = ds.variables[var][:, col[0], row[0]]
                forcingds = forcingds.reshape((1, forcingds.size, 1, 1))
                datain['lightning.' + var] = np.repeat(forcingds, nens, axis=3)
            else:
                if cnt == 0:
                    dates = num2date(ds['time'][:], units=ds['time'].units)
                    i_from = np.where(dates==period[0])[0]
                    i_to = np.where(dates==period[-1])[0]+1
                    if len(i_from) == 0:
                        i_from = 0
                    if len(i_to) == 0:
                        i_to = len(dates)+1
                forcingds = ds.variables[var][i_from[0]:i_to[0],col[0],row[0]]
                forcingds = forcingds.reshape((1, forcingds.size, 1, 1))
                datain['forcing.' + var] = np.repeat(forcingds, nens, axis=3)

    return datain


def init_static(file, nens, col, row):

    variables = ['VOD_Q01', 'VOD_Q50', 'VOD_Q99', 'mask_ICE', 'mask_DEC', 'mask_WAT',
                 'SM_crt', 'SM_wlp', 'SM_por', 'SM_res', 'SM_flc']

    staticin = {}

    with Dataset(file) as ds:
        for var in variables:
            staticin['static.' + var] = np.repeat(ds.variables[var][col,row].reshape(1,1,1,1), nens, axis=3)

    return staticin


def init_initial(*args):

    # either read single gpi for time series processing, or spatial subset for image processing
    if len(args) == 4:
        file, nens, col, row = args
    else:
        file, nens, latmin, latmax, lonmin, lonmax = args

    vars = ['w_frac_B', 'w_frac_H', 'w_frac_T', 'El_frac_B', 'El_frac_H', 'El_frac_T']

    # print('reading startup', file)

    initial = xr.open_dataset(file)
    initial = initial[vars]
    if len(args) == 4:
        initial = initial.isel(lat=row, lon=col)
    else:
        initial = initial.sel(lat=initial.lat.loc[(initial.lat >= latmin) & (initial.lat <= latmax)],
                              lon=initial.lon.loc[(initial.lon >= lonmin) & (initial.lon <= lonmax)])
    initial = initial.stack(z=('lon', 'lat'))

    for k in initial.keys():
        level = list(set(initial[k].dims) - set(['time', 'ens', 'z']))[0]
        initial[k] = initial[k].transpose('z', 'time', level, 'ens')

    initialin = {}

    for i in initial.data_vars:
        name = 'initial.' + (initial.data_vars[i].name)

        if (initial.dims['ens'] == 1 and nens > 1):

            # print("initial: repeating startup values across defined number of ensemble members")
            initialin[name] = np.repeat(initial.data_vars[i].values, nens, axis=3)

        elif (initial.dims['ens'] > 1 and initial.dims['ens'] != nens):

            sys.exit('initial: incompatible ensemble members between startup file and defined ensemble members')

        elif (initial.dims['ens'] > 1 and initial.dims['ens'] == nens):

            print("initial: startup file variable has same amount of ensemble members as defined")
            initialin[name] = initial.data_vars[i].values

    return initialin


# needs adapting to ensembles
def output(sv, mask, lon, lat, day, outpath):

    file = outpath / ('GLEAM_' + day.strftime('%Y_%m_%d') + '.nc')
    # print('writing output')

    ds = xr.Dataset()

    for k in sv.keys():
        tmp = np.zeros(shape=[mask.shape[0], sv[k].shape[1], sv[k].shape[2], sv[k].shape[3]])
        tmp.fill(np.nan)

        tmp[mask, :, :, :] = sv[k]
        tmp = tmp.reshape(lon.size, lat.size, sv[k].shape[1], sv[k].shape[2], sv[k].shape[3])

        if sv[k].shape[2] == 1:
            level = 'level1d'
        elif sv[k].shape[2] == 2:
            level = 'level2d'
        elif sv[k].shape[2] == 3:
            level = 'level3d'

        da = xr.DataArray(data=tmp, coords={'lon': lon, 'lat': lat, 'time': np.array([np.datetime64(day)]),
                          level: np.arange(1,sv[k].shape[2]+1), 'ens': np.arange(1,sv[k].shape[3]+1)},
                          dims=['lon', 'lat', 'time', level, 'ens'])
        da.name = k

        ds = xr.merge([ds, da])

    # compression level, quite slow
    comp = dict(zlib=True, complevel=0)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(file, mode='w', encoding=encoding)


def output_ts(res, outpath, gpi, out_vars, period, nens):

    outpath = Path(outpath)

    if not outpath.exists():
        outpath.mkdir(parents=True)

    fname = outpath / ('%i.nc' % gpi)

    timeunit = 'days since 1970-01-01 00:00:00 UTC'
    dates = date2num(period.to_pydatetime(), timeunit).astype('int32')

    dims = OrderedDict([('time', dates),
                        ('ens', np.arange(nens).astype('int8'))])

    chunks = [len(dates), nens]

    with Dataset(fname, mode='w') as ds:

        for dim,chunk in zip(dims,chunks):
            dtype = dims[dim].dtype
            ds.createDimension(dim, len(dims[dim]))
            ds.createVariable(dim, dtype,
                              dimensions=(dim,),
                              chunksizes=(chunk,),
                              zlib=True)
            ds.variables[dim][:] = dims[dim]

        ds.variables['time'].setncatts({'long_name': 'time',
                                        'units': timeunit})

        for var in out_vars:
            ds.createVariable(var, 'float32',
                              dimensions=list(dims.keys()),
                              chunksizes=chunks,
                              fill_value=-9999.,
                              zlib=True)

            ds.variables[var][:,:] = res[var][:,:]



def read_obs(file, var):

    print('reading observation', file)

    obs = xr.open_dataset(file)
    obs = obs[var]
    obs = obs.transpose('lon', 'lat')
    obs = obs.stack(z=('lon', 'lat'))
    obs = obs.transpose()
    obs = obs.expand_dims(['time', 'level', 'ens'], [1, 2, 3])

    for k in obs.keys():
        obs[k] = obs[k].transpose('z', 'time', 'level', 'ens')

    return obs