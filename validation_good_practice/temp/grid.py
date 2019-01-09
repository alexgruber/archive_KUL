
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from netCDF4 import Dataset

import numpy as np
import pandas as pd

from ease_grid.ease2_grid import EASE2_grid

# latmin = 25.217895
# latmax = 49.433758
# lonmin = -124.543569
# lonmax = -67.033195
#
# rowmin = 48     # equals lat MAX !!
# rowmax = 116    # equals lat MIN !!
# colmin = 148
# colmax = 302

class EASE2(EASE2_grid):

    def __init__(self):
        res = 36000
        map_scale = 36032.220840584
        super(EASE2, self).__init__(res, map_scale=map_scale)

res = pd.read_csv(r"D:\data_sets\ASCAT\warp5_grid\pointlist_warp_conus.csv", index_col=0)

smos = pd.read_csv(r"D:\data_sets\SMOS_L2\smos_grid.txt", delim_whitespace=True, names=['gpi','lon','lat','alt','wf'])

smos_lat = smos['lat'].values
smos_lon = smos['lon'].values
smos_gpi = smos['gpi'].values

res = pd.read_csv(r"D:\data_sets\EASE2_grid\grid_lut_old.csv", index_col=0)
res['smos_lat'] = np.nan
res['smos_lon'] = np.nan
res['smos_gpi'] = -9999

for idx, data in res.iterrows():

    print idx
    r = np.sqrt((smos_lat - data['ease2_lat'])**2 + (smos_lon - data['ease2_lon'])**2)
    ind = np.where((r - r.min())<0.0001)[0][0]

    res.loc[idx,'smos_lat'] = smos_lat[ind]
    res.loc[idx,'smos_lon'] = smos_lon[ind]
    res.loc[idx,'smos_gpi'] = smos_gpi[ind]

res.to_csv(r"D:\data_sets\EASE2_grid\grid_lut.csv", float_format="%.6f")

#
# llcrnrlat = 24
# urcrnrlat = 51
# llcrnrlon = -128
# urcrnrlon = -64
#
# m = Basemap(projection='mill',
#             llcrnrlat=llcrnrlat,
#             urcrnrlat=urcrnrlat,
#             llcrnrlon=llcrnrlon,
#             urcrnrlon=urcrnrlon,
#             resolution='c')
# m.drawcoastlines()
# m.drawcountries()
# m.drawstates()
#
#
# res = pd.read_csv(r"D:\data_sets\EASE2\grid_lut.csv", index_col=0)
# lon = res['ease2_lon'].values
# lat = res['ease2_lat'].values
#
# x,y = m(lon,lat)
#
# plt.scatter(x,y, c=np.arange(len(x)))
#
# plt.show()
