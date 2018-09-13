
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from netCDF4 import Dataset


def plot_figure(img, lons, lats,
                llcrnrlat=-58.,
                urcrnrlat=78.,
                llcrnrlon=-172.,
                urcrnrlon=180.,
                cbrange=(0,1),
                cmap='jet_r',
                plot_cmap=True,
                fontsize=12):

    m = Basemap(projection='mill',
                    llcrnrlat=llcrnrlat,
                    urcrnrlat=urcrnrlat,
                    llcrnrlon=llcrnrlon,
                    urcrnrlon=urcrnrlon,
                    resolution='c')

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, img, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])

    if plot_cmap is True:
        cb = m.colorbar(im, "bottom", size="10%", pad="4%")
        for t in cb.ax.get_xticklabels():
             t.set_fontsize(fontsize)
        for t in cb.ax.get_yticklabels():
             t.set_fontsize(fontsize)




def calc_wght(df,cl):

    sensors = cl.split('_',2)

    err_act = df['err_'+sensors[1]]
    err_pass = df['err_'+sensors[2]]

    return err_pass / (err_act + err_pass)

def plot_weights_combined_v3():

    root = r"D:\work\esa_cci_sm" + '\\'
    errp_file = root + 'errp_combined_v3.csv'

    fout = r'I:\publications\2018_ag_cci_merging\images\weights.png'

    df = pd.read_csv(errp_file, index_col=0, dtype='float')
    df.index = df.index.values.astype('int64')
    df = df.loc[df.index>0,:]

    tags = ['class_ers_ssmi',
            'class_ers_tmi',
            'class_ers_amsre',
            'class_ascat_amsre',
            'class_ascat_amsre_windsat',
            'class_ascat_amsre_windsat_smos',
            'class_ascat_windsat_smos',
            'class_ascat_smos_amsr2']

    titles = ['ERS - SSM/I',
              'ERS - SSM/I + TMI',
              'ERS - AMSR-E',
              'ASCAT - AMSR-E',
              'ASCAT - AMSR-E + WindSat',
              'ASCAT - AMSR-E + Windsat + SMOS',
              'ASCAT - WINDSAT + SMOS',
              'ASCAT - SMOS + AMSR2']

    figsize = (10,12)

    f = plt.figure(figsize=figsize)

    lons = (np.arange(360 * 4) * 0.25) - 179.875
    lats = (np.arange(180 * 4) * 0.25) - 89.875
    lons, lats = np.meshgrid(lons, lats)

    for i,tag in enumerate(tags):

        plt.subplot(5,2,i+1)

        w_act = calc_wght(df,tag).copy()

        w_act[df[tag]==0] = np.nan
        w_act[df[tag]==1] = 0.0
        w_act[df[tag]==2] = 1.0
        w_act[df[tag]==3] = 0.5

        if tag == 'class_ers_tmi':
            w_act2 = calc_wght(df,'class_ers_ssmi').copy()
            w_act2[df['class_ers_ssmi'] == 2] = 1.0
            w_act[df[tag] == 2] = w_act2[df[tag] == 2]


        name = 'w_act_'+'_'.join(tag.split('_')[1::])
        w_act.name = name

        img = np.empty(lons.size,dtype='float32')
        img.fill(None)
        img[w_act.index.values] = w_act
        img_masked = np.ma.masked_invalid(img.reshape((180*4,360*4)))

        plot_figure(img_masked, lons, lats)
        plt.title(titles[i], fontsize=10)


    plt.subplot(5, 2, 9)
    img_masked = Dataset(r"D:\work\esa_cci_sm\ASCAT_AMSR2_blendingMap.nc")['blendingMap'][:].reshape(lons.shape).astype('float')
    img_masked[img_masked==2] = 0.
    img_masked[img_masked==3] = 0.5
    plot_figure(img_masked, lons, lats)
    plt.title('ASCAT - AMSR2 (ESA CCI SM v2)', fontsize=8)

    plt.subplot(5, 2, 10)
    img_masked = np.flipud(Dataset(r"D:\work\esa_cci_sm\ESACCI-SOILMOISTURE-MEAN_VOD_V01.1.nc")['vod'][:])
    plot_figure(img_masked, lons, lats, cmap='YlGn')
    plt.title('Mean VOD', fontsize=8)

    # plt.show()
    plt.tight_layout()

    f.savefig(fout, dpi=300)
    plt.close()

def plot_gapfilled_snr():

    fname = r"D:\work\esa_cci_sm\errp_combined_gapfilled.csv"

    fout = r'I:\publications\2018_ag_cci_merging\images\snr.png'

    df = pd.read_csv(fname, index_col=0, dtype='float')
    df.index = df.index.values.astype('int64')
    df = df.loc[df.index>0,:]

    mask = pd.read_csv("D:\work\esa_cci_sm\pointlist_Greenland_quarter.csv",index_col=0)
    df.drop(mask.index, errors='ignore', inplace=True)

    tags = ['snr_ers',
            'snr_ascat',
            'snr_ssmi',
            'snr_tmi',
            'snr_amsre',
            'snr_windsat',
            'snr_smos',
            'snr_amsr2']

    titles = ['ERS',
              'ASCAT',
              'SSM/I',
              'TMI',
              'AMSR-E',
              'WindSat',
              'SMOS',
              'AMSR2']

    figsize = (10,12)

    f = plt.figure(figsize=figsize)

    lons = (np.arange(360 * 4) * 0.25) - 179.875
    lats = (np.arange(180 * 4) * 0.25) - 89.875
    lons, lats = np.meshgrid(lons, lats)

    for i,tag in enumerate(tags):

        plt.subplot(4,2,i+1)

        img = np.full(lons.size, np.nan)
        img[df.index.values] = 10*np.log10(df[tag])
        img_masked = np.ma.masked_invalid(img.reshape((180*4,360*4)))

        plot_figure(img_masked, lons, lats, cbrange=[-9,9],plot_cmap=True)
        plt.title(titles[i], fontsize=12)

    # plt.show()
    plt.tight_layout()

    f.savefig(fout, dpi=300)
    plt.close()


def plot_gapfilled_snr_v5():

    fname = r"D:\work\esa_cci_sm\errp_v5\anom\errp_combined_gapfilled.csv"

    fout = r'I:\publications\2018_ag_cci_merging\images\v5_snr_anom.png'

    df = pd.read_csv(fname, index_col=0, dtype='float')
    df.index = df.index.values.astype('int64')
    df = df.loc[df.index>0,:]

    mask = pd.read_csv("D:\work\esa_cci_sm\pointlist_Greenland_quarter.csv",index_col=0)
    df.drop(mask.index, errors='ignore', inplace=True)

    tags = ['snr_ers',
            'snr_ascat',
            'snr_ssmi',
            'snr_tmi',
            'snr_amsre',
            'snr_windsat',
            'snr_smos',
            'snr_amsr2',
            'snr_smap']

    titles = ['ERS',
              'ASCAT',
              'SSM/I',
              'TMI',
              'AMSR-E',
              'WindSat',
              'SMOS',
              'AMSR2',
              'SMAP']

    figsize = (14,9)

    f = plt.figure(figsize=figsize)

    lons = (np.arange(360 * 4) * 0.25) - 179.875
    lats = (np.arange(180 * 4) * 0.25) - 89.875
    lons, lats = np.meshgrid(lons, lats)

    for i,tag in enumerate(tags):

        plt.subplot(3,3,i+1)

        img = np.full(lons.size, np.nan)
        img[df.index.values] = 10*np.log10(df[tag])
        img_masked = np.ma.masked_invalid(img.reshape((180*4,360*4)))

        plot_figure(img_masked, lons, lats, cbrange=[-9,9],plot_cmap=True)
        plt.title(titles[i], fontsize=12)

    # plt.show()
    plt.tight_layout()

    f.savefig(fout, dpi=300)
    plt.close()


if __name__=='__main__':
    plot_gapfilled_snr_v5()