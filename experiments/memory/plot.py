
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from pyldas.grids import EASE2
from pyldas.interface import LDAS_io


def plot_ease_img(data,tag,
                  llcrnrlat=24,
                  urcrnrlat=51,
                  llcrnrlon=-128,
                  urcrnrlon=-64,
                  cbrange=(-20,20),
                  cmap='jet',
                  title='',
                  fontsize=16):

    grid = EASE2()
    lons,lats = np.meshgrid(grid.ease_lons, grid.ease_lats)

    ind_lat = data['row'].values.astype('int')
    ind_lon = data['col'].values.astype('int')
    img = np.full(lons.shape, np.nan)
    img[ind_lat,ind_lon] = data[tag]
    img_masked = np.ma.masked_invalid(img)

    m = Basemap(projection='mill',
                llcrnrlat=llcrnrlat,
                urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,
                urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

    im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="8%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    if title == '':
        title = tag.lower()
    plt.title(title,fontsize=fontsize)


res = pd.read_csv(r"D:\work\sm_memory\v2\result.csv", index_col=0)

plt.figure(figsize = (18,4))

# mode = 'anom'
# cbrange = [0,4]

mode = 'anom'
cbrange = [0,3]

# sensors = ['AMSR2','SMOS','SMAP','ASCAT','MERRA2']
sensors = ['ASCAT','AMSR2','MERRA2']

tags = ['tau_'+mode+'_'+ s for s in sensors]

for i,tag in enumerate(tags):

    plt.subplot(1,3,i+1)
    plot_ease_img(res, tag, cbrange=cbrange, title=sensors[i])

plt.tight_layout()
plt.show()




















