

import pandas as pd
import numpy as np

from pathlib import Path

from pyldas.interface import GEOSldas_io
#
# import torch
# import torch.nn as nn
# torch.manual_seed(1)



#
# io_ofa = GEOSldas_io('ObsFcstAna', exp='NLv4_M36_US_OL_Pcorr')
# # io_cat = GEOSldas_io('SMAP_L4_SM_gph', exp='NLv4_M36_US_OL_Pcorr')
#
#
# lat, lon = 48.206665, -100.257308 # North Dacota (good)
# col, row = io_ofa.grid.lonlat2colrow(lon, lat)
#
# ts_obs = io_ofa.timeseries['obs_obs'].isel(lat=row,lon=col).to_pandas()
# ts_fcst = io_ofa.timeseries['obs_fcst'].isel(lat=row,lon=col).to_pandas()
#
#
#
#

# ts_cat = io_cat.timeseries.isel(lat=row,lon=col).to_dataframe()

# Required inputs:
# WCSF - Surface moisture content
# TPSURF - Surface temperature
# LAI

# SWE
# Tair

# Only under snow-free / non-frozen conditions!

