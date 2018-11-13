
import os
import numpy as np
import pandas as pd

def walk_up_folder(path, depth=1):
    """ Walk up a specific number of sub-directories """
    _cur_depth = 0
    while _cur_depth < depth:
        path = os.path.dirname(path)
        _cur_depth += 1
    return path


def remove_fields(nparray, names):
    """ Remove fields field from a structured numpy array """

    fields = list(nparray.dtype.names)
    for name in names:
        if name in fields:
            fields.remove(name)
    return nparray[fields]


def find_files(path,searchstr):
    """ Recursive file search with a given search string """

    res = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.find(searchstr) != -1:
                res.append(os.path.join(root, f))

    if len(res) == 0:
        print 'No files found which contain: "' + searchstr + '".'
        return None
    elif len(res) == 1:
        return res[0]
    else:
        return np.array(res)

def merge_files():

    path = r'D:\work\MadKF\synthetic_experiment'
    files = find_files(path,'.csv')

    result = pd.DataFrame()
    for f in files:
        tmp = pd.read_csv(f, index_col=0)
        result = result.append(tmp)

    result.to_csv(path + '\\result.csv')

if __name__=='__main__':
    merge_files()
