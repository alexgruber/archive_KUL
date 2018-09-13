
import numpy as np
import pandas as pd


def calc_anomaly(Ser, method='harmonic', output='anomaly', longterm=False):

    if (output=='climatology')&(longterm is True):
        output = 'climSer'

    xSer = Ser.dropna().copy()
    if len(xSer) == 0:
        return xSer

    doys = xSer.index.dayofyear.values
    doys[xSer.index.is_leap_year & (doys > 59)] -= 1
    climSer = pd.Series(index=xSer.index)

    if not method in ['harmonic','mean','moving_average','ma']:
        print 'Unknown method: ' + method
        return climSer

    if longterm is True:
        if method=='harmonic':
            clim = calc_clim_harmonic(xSer)
        if method=='mean':
            clim = calc_clim_harmonic(xSer, n=0)
        if (method=='moving_average')|(method=='ma'):
            clim = calc_clim_moving_average(xSer)
        if output == 'climatology':
            return clim
        climSer[:] = clim[doys]

    else:
        years = xSer.index.year
        for yr in np.unique(years):
            if method == 'harmonic':
                clim = calc_clim_harmonic(xSer[years == yr])
            if method == 'mean':
                clim = calc_clim_harmonic(xSer[years == yr], n=0)
            if (method == 'moving_average') | (method == 'ma'):
                clim = calc_clim_moving_average(xSer[years == yr])
            climSer[years == yr] = clim[doys[years == yr]].values

    if output == 'climSer':
        return climSer

    return xSer - climSer


def calc_clim_harmonic(Ser, n=3, cutoff=True):
    """
    Calculates the mean seasonal cycle of a data set
    by fitting harmonics.
    (!! Leap years are not yet properly treated !!)

    Parameters
    ----------
    Ser : pd.Series w. DatetimeIndex
        Timeseries of which the climatology shall be calculated.
    n : int (optional)
        Number of harmonics that should be fitted.
        n=0 : long term mean
        n=1 : long term mean + annual cycle
        n=2 : long term mean + annual + half-annual cycle
        n=3 : long term mean + annual + half-annual + seasonal cycle
    cutoff : boolean
        If set, the climatology is not allowed to exceed the min/max of the original time series.

    Returns
    -------
    clim : pd.Series
        climatology of Ser (without leap days)
    """

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

    doys = np.arange(T)+1
    clim = pd.Series(index=np.arange(T)+1)
    clim[:] = x[0]
    for j in np.arange(n)+1:
        clim[:] += x[j] * np.cos(j * 2 * np.pi * doys / T) + x[j+n] * np.sin(j * 2 * np.pi * doys / T)

    if (cutoff is True)&(len(clim.dropna()!=0)):
        p = np.nanpercentile(xSer.values, [5,95])
        clim[(clim<p[0])|(clim>p[1])] = np.nan

    return clim

def calc_clim_moving_average(Ser, window_size=45, n_min=20, return_n=False):
    """
    Calculates the mean seasonal cycle as long-term mean within a moving average window.

    Parameters
    ----------
    Ser : pd.Series w. DatetimeIndex
        Timeseries of which the climatology shall be calculated.
    window_size : int
        Moving Average window size
    n_min : int
        Minimum number of data points to calculate average
    return_n : boolean
        If true, the number of data points over which is averaged is returned
    Returns
    -------
    clim : pd.Series
        climatology of Ser (without leap days)
    n_days : pd.Series
        the number of data points available within each window
    """

    xSer = Ser.dropna().copy()
    doys = xSer.index.dayofyear.values

    # in leap years, subtract 1 for all days after Feb 28
    doys[xSer.index.is_leap_year & (doys > 59)] -= 1

    clim_doys =  np.arange(365) + 1
    clim = pd.Series(index=clim_doys)
    n_data = pd.Series(index=clim_doys)

    for doy in clim_doys:

        # Avoid artifacts at start/end of year
        tmp_doys = doys.copy()
        if doy < window_size/2.:
            tmp_doys[tmp_doys > 365 - (np.ceil(window_size/2.)-doy)] -= 365
        if doy > 365 - (window_size/2. - 1):
            tmp_doys[tmp_doys < np.ceil(window_size/2.) - (365-doy)] += 365

        n_data[doy] = len(xSer[(tmp_doys >= doy - np.floor(window_size/2.)) & \
                               (tmp_doys <= doy + np.floor(window_size/2.))])

        if n_data[doy] >= n_min:
            clim[doy] = xSer[(tmp_doys >= doy - np.floor(window_size/2.)) & \
                             (tmp_doys <= doy + np.floor(window_size/2.))].values.mean()

    if return_n is False:
        return clim
    else:
        return clim, n_data



def calc_pentadal_mean(Ser, n_min=20, return_n=False):
    """
    Calculates the mean seasonal cycle as long-term mean within a 45 days moving average window
    for each pentad (Faster than "calc_clim_moving_average" because output only per pentad)

    Parameters
    ----------
    Ser : pd.Series w. DatetimeIndex
        Timeseries of which the climatology shall be calculated.
    n_min : int
        Minimum number of data points to calculate average
    return_n : boolean
        If true, the number of data points over which is averaged is returned
    Returns
    -------
    clim : pd.Series
        climatology of Ser (without leap days)
    n_days : pd.Series
        the number of data points available within each window
    """

    xSer = Ser.dropna().copy()
    doys = xSer.index.dayofyear.values

    # in leap years, subtract 1 for all days after Feb 28
    doys[xSer.index.is_leap_year & (doys > 59)] -= 1

    Ser_pentad = np.floor((doys - 1) / 5.) + 1

    pentads = np.arange(73) + 1
    clim = pd.Series(index=pentads)
    n_data = pd.Series(index=pentads)
    for p in pentads:
        tmp_pentad = Ser_pentad.copy()
        if p < 5:
            tmp_pentad[tmp_pentad > 10] -= 73
        if p > 69:
            tmp_pentad[tmp_pentad < 60] += 73
        n_data[p] = len(xSer[(tmp_pentad >= p - 4) & (tmp_pentad <= p + 4)])

        if n_data[p] >= n_min:
            clim[p] = xSer[(tmp_pentad >= p - 4) & (tmp_pentad <= p + 4)].values.mean()

    if return_n is False:
        return clim
    else:
        return clim, n_data

