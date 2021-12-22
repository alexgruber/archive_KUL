import sys
import getpass

# This needs to point to the directory where pyldas is located (intended for the HPC)
uid = getpass.getuser()
sys.path.append('/data/leuven/' + uid[3:6] + '/' + uid + '/python')

from myprojects.publications.deforestation_paper.preprocessing import reformat_SMOS_IC

reformat_SMOS_IC()