
import numpy as np
import pandas as pd

from cmath import rect

from pathlib import Path
from netCDF4 import Dataset, num2date

import seaborn as sns
import matplotlib.pyplot as plt

from myprojects.readers.ascat import HSAF_io

from scipy.optimize import minimize
from scipy.stats import pearsonr

def lin2db(x):
    return 10 * np.log10(x)

def db2lin(x):
    return 10 ** (x/10)

def WCM(SM, LAI, A, B, C, D):
    '''
    A ~ scattering albedo
    B ~ vegetation optical depth
    C, D ~ slope, curvature of soil moisture / backscatter relation
    '''

    theta = 40 * np.pi / 180.

    T2 = np.exp((-2 * B * LAI) / np.cos(theta))         # Canopy attenuation
    sig0s = db2lin(C + D * SM)                          # (linearized) soil backscatter
    sig0v = A * LAI * np.cos(theta) * (1 - T2)          # vegetation backscatter
    sig0 = lin2db(T2 * sig0s + sig0v)                   # Attenuated soil backscatter + vegetation backscatter

    return sig0

def WCM_dyn(SM, LAI, A1, A2, B, C, D):
    '''
    A ~ scattering albedo
    B ~ vegetation optical depth
    C, D ~ slope, curvature of soil moisture / backscatter relation
    '''

    theta = 40 * np.pi / 180.

    phase = -3.003542501988556
    c = rect(A2, phase)

    A = pd.Series(A1, index=SM.index)
    doys = A.index.dayofyear.values
    doys[A.index.is_leap_year & (doys > 59)] -= 1

    A.loc[:] += c.real * np.cos(1 * 2 * np.pi * doys / 365) + c.imag * np.sin(1 * 2 * np.pi * doys / 365)

    T2 = np.exp((-2 * B * LAI) / np.cos(theta))  # Canopy attenuation
    sig0s = db2lin(C + D * SM)  # (linearized) soil backscatter
    sig0v = A * LAI * np.cos(theta) * (1 - T2)  # vegetation backscatter
    sig0 = lin2db(T2 * sig0s + sig0v)  # Attenuated soil backscatter + vegetation backscatter

    return sig0


def read_data():

    i_lat = 750
    i_lon = 750

    ascat = HSAF_io()
    merra2 = Dataset('/Users/u0116961/data_sets/MERRA2/MERRA2_timeseries.nc4')

    with Dataset('/Users/u0116961/data_sets/DMP_COPERNICUS/DMP_COPERNICUS_timeseries.nc') as ds:
        time = pd.DatetimeIndex(num2date(ds['time'][:], units=ds['time'].units,
                                         only_use_python_datetimes=True, only_use_cftime_datetimes=False))
        dmp_ts = pd.DataFrame({'DMP': ds['DMP'][:, i_lat, i_lon]}, index=time)
        lat = ds['lat'][i_lat].data
        lon = ds['lon'][i_lon].data

    ind_lat = abs(merra2['lat'][:] - lat).argmin()
    ind_lon = abs(merra2['lon'][:] - lon).argmin()
    gpi_ascat = ascat.latlon2gpi(lat, lon)

    time = pd.DatetimeIndex(num2date(merra2['time'][:], units=merra2['time'].units,
                                     only_use_python_datetimes=True, only_use_cftime_datetimes=False))
    df = pd.DataFrame({'time': time,
                       'sm': merra2['SFMC'][:, ind_lat, ind_lon],
                       'DMP': dmp_ts.reindex(time).values.flatten() / 10,
                       'sig40_ascat': ascat.read(gpi_ascat, resample_time=True, var='sigma40').reindex(
                           time).values}, index=time)
    merra2.close()
    ascat.close()

    return df



def cost_function(args, df):
    A, B, C, D = args

    df['sig40_wcm'] = WCM(df['sm'], df['DMP'], A, B, C, D)

    return ((df['sig40_wcm'] - df['sig40_ascat'])**2).mean()


def cost_function_dyn(args, df):
    A1, A2, B, C, D = args

    df['sig40_wcm'] = WCM_dyn(df['sm'], df['DMP'], A1, A2, B, C, D)

    return ((df['sig40_wcm'] - df['sig40_ascat'])**2).mean()



def calibrate():

    A0 = 0.2
    A10 = 0.02
    A20 = 0.01
    B0 = 0.2
    C0 = -20
    D0 = 50

    df = read_data()

    params = minimize(cost_function, (A0, B0, C0, D0), args=(df,), method='Nelder-Mead', tol=1e-03)
    A, B, C, D = params.x
    print(params.x)
    df['sig40_wcm_static'] = WCM(df['sm'], df['DMP'], A, B, C, D)

    params = minimize(cost_function_dyn, (A10, A20, B0, C0, D0), args=(df,), method='Nelder-Mead', tol=1e-03)
    A1, A2, B, C, D = params.x
    print(params.x)
    df['sig40_wcm_dynamic'] = WCM_dyn(df['sm'], df['DMP'], A1, A2, B, C, D)

    plot(df, cols=['sig40_ascat', 'sig40_wcm_static', 'sig40_wcm_dynamic' ])


def plot(df, cols=None):

    xdf = df.dropna().copy()

    if cols is None:
        cols = xdf.columns[1::]

    print(xdf.corr())

    fontsize=12
    plt.figure(figsize=(18,11))
    sns.set_context('talk', font_scale=0.8)

    sns.lineplot(x='time', y='value', hue='variable', data=xdf.melt('time', cols))
    plt.gca().set_xlabel('')
    plt.gca().set_ylabel('')

    plt.show()


def calc_harmonic(Ser, n=1, cutoff=True):

    T = 365

    xSer = Ser.dropna().copy()
    doys = xSer.index.dayofyear.values

    # in leap years, subtract 1 for all days after Feb 28
    doys[xSer.index.is_leap_year & (doys>59)] -= 1

    A = np.ones((len(doys),2*n+1))

    for j in np.arange(n)+1:
        A[:,j] = np.cos(j * 2 * np.pi * doys / T)
        A[:,j+n] = np.sin(j * 2 * np.pi * doys / T)

    A = np.matrix(A)
    y = np.matrix(xSer.values).T
    try:
        x = np.array((A.T * A).I * A.T * y).flatten()
    except:
        x = np.full(2*n+1,np.nan)
    print(x)

    # x_t = x.copy()
    # for j in np.arange(n) + 1:
    #     x_t[j] = np.sqrt(x[j]**2 + x[j+n]**2)
    #     x_t[j+n] = np.arctan(x[j+n] / x[j])

    doys = np.arange(T)+1
    clim = pd.Series(index=np.arange(T)+1)
    clim[:] = x[0]
    for j in np.arange(n)+1:
        # clim[:] += x_t[j] * np.cos(j * 2 * np.pi * doys / T + x_t[j+n]) + x_t[j] * np.sin(j * 2 * np.pi * doys / T + x_t[j+n])
        clim[:] += x[j] * np.cos(j * 2 * np.pi * doys / T) + x[j+n] * np.sin(j * 2 * np.pi * doys / T)

    if (cutoff is True)&(len(clim.dropna()!=0)):
        p = np.nanpercentile(xSer.values, [5,95])
        clim[(clim<p[0])|(clim>p[1])] = np.nan

    return clim


def add_harmonic(df, offs, ampl, phase):

    h = pd.Series(offs, index=df.index)
    doys = h.index.dayofyear.values
    doys[h.index.is_leap_year & (doys > 59)] -= 1

    c = rect(ampl, phase)

    h.loc[:] += c.real * np.cos(1 * 2 * np.pi * doys / 365) + c.imag * np.sin(1 * 2 * np.pi * doys / 365)

    df['harmonic'] = h.values


if __name__=='__main__':
    #
    # df = read_data()
    #
    # clim = calc_harmonic(df['DMP'])
    # doys = df.index.dayofyear.values
    # doys[df.index.is_leap_year & (doys > 59)] -= 1
    # df['DMP_harm'] = clim[doys].values

    # offs = 3
    # ampl = 2
    # phase = -3.003542501988556
    # add_harmonic(df, offs, ampl, phase)

    # plot(df)

    calibrate()