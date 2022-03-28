
import sys
import getpass

# This needs to point to the directory where pyldas is located (intended for the HPC)
uid = getpass.getuser()
sys.path.append('/data/leuven/' + uid[3:6] + '/' + uid + '/python')

from myprojects.experiments.machine_learning.copernicus_dmp_prediction.reformat import reformat

# experiment name and parameters to be converted to netCDF need to be passed when calling this script!
path_in = sys.argv[1]
path_out = sys.argv[2]

reformat(path_in, path_out)

