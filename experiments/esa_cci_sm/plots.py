
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from netCDF4 import Dataset

def plot_locations():

    stats = pd.DataFrame.from_csv(r"D:\work\ESA_CCI_SM\stations.csv")

    # llcrnrlat = 24
    # urcrnrlat = 51
    # llcrnrlon = -128
    # urcrnrlon = -64

    llcrnrlat = -44
    urcrnrlat = 72
    llcrnrlon = -169
    urcrnrlon = 154

    figsize = (20, 9)

    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')

    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, resolution='c', )
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

    for net in np.unique(stats['network']):
        tmp = stats[stats['network']==net]
        xs, ys = m(tmp.lon.values, tmp.lat.values)

        plt.plot(xs, ys, 'o', markersize=12, markeredgecolor='black', label=net)

    plt.legend(fontsize=9,loc='lower left')

    plt.tight_layout()

    plt.show()
    # fname = r'D:\work\MadKF\CONUS\ismn_eval\station_locations.png'
    # plt.savefig(fname, dpi=f.dpi)
    # plt.close()

def boxplot_validation_result():

    res = pd.read_csv('/work/ESA_CCI_SM/validation.csv', index_col=0)
    r_ismn = pd.read_csv('/work/ESA_CCI_SM/ismn_r_true.csv', index_col=0)

    plt.figure(figsize=(17, 8))

    modes = ['ACTIVE', 'PASSIVE', 'COMBINED']
    freq = ['abs', 'anom']

    periods = ['p1', 'p2', 'p3', 'p4']
    xticks = ['2007-10-01\n2010-01-14',
              '2010-01-15\n2011-10-04',
              '2011-10-05\n2012-06-30',
              '2012-07-01\n2014-12-31']

    versions = ['v02.2', 'v03.3', 'v04.4']

    offsets = [-0.25, 0.0, 0.25]
    cols = ['lightblue', 'lightgreen', 'coral']
    fontsize = 14

    k = 0

    idx = np.empty(0)

    for f in freq:
        ylabel = 'Absolute' if f == 'abs' else 'Anomaly'
        # ylim = [-0.15, 1.1] if f == 'abs' else [-0.15, 1.1]
        # yoffs = -0.081 if f == 'abs' else - 0.063
        ylim = [-0.15, 1.1]
        yoffs = -0.105
        for m in modes:
            k += 1

            ax = plt.subplot(2, 3, k)
            plt.grid(color='k', linestyle='--', linewidth=0.25)
            data = list()
            ticks = list()
            pos = list()
            colors = list()
            texts = list()
            for i, (tick, p) in enumerate(zip(xticks,periods)):
                ticks.append(tick)

                n_stat = 0
                n_meas = 0
                for col, offs, v in zip(cols, offsets, versions):

                    tmp_r = res.loc[:, 'corr_' + m + '_' + v + '_' + p + '_' + f]
                    tmp_p = res.loc[:, 'p_' + m + '_' + v + '_' + p + '_' + f]
                    tmp_n = res.loc[:, 'n_' + m + '_' + v + '_' + p + '_' + f]
                    tmp_data = tmp_r[(tmp_p<=0.05)&(tmp_n>100)].dropna()
                    # tmp_data = tmp_r.dropna()

                    tmp_data = (tmp_data / r_ismn.loc[tmp_data.index, 'r_'+f]).dropna()
                    tmp_data[tmp_data>1.0] = 1.0

                    if len(tmp_data) > 0:
                        idx = np.append(idx, tmp_data.index.values)

                    print(f,m,tick[0:10],v,'%.2f' % tmp_data.median())
                    data.append(tmp_data.values)
                    pos.append(i + 1 + offs)
                    colors.append(col)

                    n_stat += len(tmp_data)
                    n_meas += np.mean(tmp_n[tmp_data.index])

                n_stat /= 3.
                n_meas /= 3.

                if np.isnan(n_meas):
                    n_meas = 0.

                texts.append('%i (%i)' % (int(round(n_stat)), int(round(n_meas))))

            box = ax.boxplot(data, whis=[10, 90], showfliers=False, positions=pos, widths=0.13, patch_artist=True)
            for patch, color in zip(box['boxes'], colors):
                patch.set(color='black', linewidth=2)
                patch.set_facecolor(color)
            for patch in box['medians']:
                patch.set(color='black', linewidth=2)
            for patch in box['whiskers']:
                patch.set(color='black', linewidth=1)
            # plt.figlegend((box['boxes'][0:4]), versions, 'lower right', fontsize=fontsize-2)

            if m == 'ACTIVE':
                plt.yticks(np.arange(-0.0,1.2,0.2), np.arange(-0.0,1.2,0.2), fontsize=fontsize)
                plt.ylabel(ylabel, fontsize=fontsize)
            else:
                plt.yticks(np.arange(-0.0,1.2,0.2), '')

            plt.xlim(0.5, len(ticks) + 0.5)
            plt.ylim(ylim)
            for i in np.arange(len(periods)):
                plt.axvline(i + 0.5, linewidth=1, color='k')

            if f == 'abs':
                ax.set_title(m, fontsize=fontsize + 2)
                plt.xticks(np.arange(len(periods)) + 1, '')
            else:
                # if m == 'ACTIVE':
                #     ax.set_title('Absolute', fontsize=fontsize + 2)
                # else:
                ax.set_title('', fontsize=fontsize + 2)
                plt.xticks(np.arange(len(periods)) + 1, ticks, fontsize=fontsize-2)


            plt.axhline(0, linestyle='--', color='black', linewidth=1)
            for x, t in zip(np.arange(len(periods)), texts):
                if len(t) == 9:
                    x += 0.74
                elif len(t) == 8:
                    x += 0.74
                elif len(t) == 7:
                    x += 0.75
                else:
                    x += 0.89
                plt.text(x,yoffs, t, fontsize=fontsize-2)

    # res.loc[np.unique(idx),['network','station','lat','lon']].to_csv(r'D:\work\ESA_CCI_SM\stations.csv')
    # for net in np.unique(res['network']):
    #     print(net)

    plt.tight_layout()
    plt.show()

def plot_figure(img, lons, lats,
                llcrnrlat=-58.,
                urcrnrlat=78.,
                llcrnrlon=-172.,
                urcrnrlon=180.,
                cbrange=(0,1),
                cmap='jet',
                plot_cmap=True,
                title='',
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
        cb = m.colorbar(im, "bottom", size="12%", pad="4%")
        for t in cb.ax.get_xticklabels():
             t.set_fontsize(fontsize)
        for t in cb.ax.get_yticklabels():
             t.set_fontsize(fontsize)

    offs = 4 - 5.2 * len(title)

    x, y = m(offs, -52)
    plt.text(x, y, title, fontsize=fontsize)



def calc_wght(df,cl):

    sensors = cl.split('_',2)

    err_act = df['err_'+sensors[1]]
    err_pass = df['err_'+sensors[2]]

    return err_pass / (err_act + err_pass)

def plot_weights_combined_v3():

    fout = '/Users/u0116961/Documents/publications/2019_ag_cci_merging/_rev1/images/v3_weights_new2.png'

    df = pd.read_csv('/work/_archive/esa_cci_sm/errp_combined_v3.csv', index_col=0, dtype='float')
    df.index = df.index.values.astype('int64')
    df = df.loc[df.index>0,:]

    mask = pd.read_csv('/work/_archive/esa_cci_sm/pointlist_Greenland_quarter.csv', index_col=0)
    df.drop(mask.index, errors='ignore', inplace=True)

    mask = Dataset('/data_sets/ESA_CCI_SM/ESA-CCI-SOILMOISTURE-LAND_AND_RAINFOREST_MASK-fv04.2.nc')
    df.drop(np.where(mask['rainforest'][:].flatten().data == 1)[0], errors='ignore', inplace=True)

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

    figsize = (8,12)

    fontsize = 10
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

        plot_figure(img_masked, lons, lats, plot_cmap = False, fontsize = 12)
        plt.title(titles[i], fontsize=fontsize)

    mask_gl = pd.read_csv('/work/_archive/esa_cci_sm/pointlist_Greenland_quarter.csv', index_col=0)
    mask_rf = Dataset('/data_sets/ESA_CCI_SM/ESA-CCI-SOILMOISTURE-LAND_AND_RAINFOREST_MASK-fv04.2.nc')

    plt.subplot(5, 2, 9)
    ds = Dataset('/work/_archive/esa_cci_sm/ASCAT_AMSR2_blendingMap.nc')['blendingMap'][:]
    ds.mask[mask_gl.index] = True
    img_masked = ds.reshape(lons.shape).astype('float')
    img_masked.mask[np.where(mask_rf['rainforest'][:] == 1)] = True

    img_masked[img_masked==2] = 0.
    img_masked[img_masked==3] = 0.5
    plot_figure(img_masked, lons, lats, fontsize = 12)
    plt.title('ASCAT - AMSR2 (ESA CCI SM v2)', fontsize=fontsize)

    plt.subplot(5, 2, 10)
    ds = np.flipud(Dataset('/work/_archive/esa_cci_sm/ESACCI-SOILMOISTURE-MEAN_VOD_V01.1.nc')['vod'][:]).flatten()
    ds.mask[mask_gl.index] = True
    img_masked = ds.reshape(lons.shape).astype('float')
    img_masked.mask[np.where(mask_rf['rainforest'][:] == 1)] = True
    plot_figure(img_masked, lons, lats, cmap='YlGn', fontsize = 12)
    plt.title('Mean VOD', fontsize=fontsize)

    plt.subplots_adjust(wspace=0.1, hspace=0)

    # plt.show()
    plt.tight_layout()

    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

def plot_gapfilled_snr():

    fname = '/work/_archive/esa_cci_sm/errp_combined_gapfilled.csv'

    fout = '/Users/u0116961/Documents/publications/2019_ag_cci_merging/_rev1/images/v4_snr_new.png'

    df = pd.read_csv(fname, index_col=0, dtype='float')
    df.index = df.index.values.astype('int64')
    df = df.loc[df.index>0,:]

    mask = pd.read_csv('/work/_archive/esa_cci_sm/pointlist_Greenland_quarter.csv',index_col=0)
    df.drop(mask.index, errors='ignore', inplace=True)

    mask = Dataset('/data_sets/ESA_CCI_SM/ESA-CCI-SOILMOISTURE-LAND_AND_RAINFOREST_MASK-fv04.2.nc')
    df.drop(np.where(mask['rainforest'][:].flatten().data==1)[0], errors='ignore', inplace=True)

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

        plot_figure(img_masked, lons, lats, cbrange=[-9,9], plot_cmap = (True if (i > 5) else False), title = titles[i], fontsize = 14)

        # plt.title(titles[i], fontsize=12)


    plt.subplots_adjust(wspace=0.1, hspace=0)

    # plt.show()
    plt.tight_layout()

    f.savefig(fout, dpi=300, bbox_inches='tight')
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

    plt.show()
    plt.tight_layout()

    # f.savefig(fout, dpi=300)
    # plt.close()


if __name__=='__main__':
    # plot_locations()
    # boxplot_validation_result()
    # plot_gapfilled_snr()
    plot_weights_combined_v3()