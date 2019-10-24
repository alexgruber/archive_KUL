

import pandas as pd

from pathlib import Path

def merge_files(path, delete=False):

    path = Path(path)

    files = path.glob('**/*.csv')
    fname = path / 'result.csv'

    result = pd.DataFrame()
    for f in files:
        tmp = pd.read_csv(f, index_col=0)
        result = result.append(tmp)

    result.sort_index().to_csv(fname, float_format='%0.3f')

    if (delete is True) & fname.exists():
        for f in files:
            f.unlink()

if __name__=='__main__':

    path = '/work/GLEAM/perturbation_correction'
    merge_files(path)

