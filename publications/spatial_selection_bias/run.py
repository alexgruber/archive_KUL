import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
from pathos.multiprocessing import ProcessPool
from itertools import repeat

from myprojects.publications.spatial_selection_bias.interface import io
from myprojects.readers.insitu import ISMN_io
from myprojects.functions import merge_files

from pytesmo.df_metrics import ubrmsd, pearsonr
from pytesmo.metrics import ecol

def run(n_procs=1, insitu=True):

    ext = '_insitu' if insitu else '_global'
    res_path = Path(r"H:\work\publications\spatial_selection_bias\results")
    if not res_path.exists():
        Path.mkdir(res_path, parents=True)

    part = np.arange(n_procs) + 1
    parts = repeat(n_procs, n_procs)

    if n_procs > 1:
        with ProcessPool(n_procs) as p:
            if insitu:
                p.map(main_insitu, part, parts)
            else:
                p.map(main_global, part, parts)
    else:
        if insitu:
            main_insitu(1, 1)
        else:
            main_global(1, 1)

    merge_files(res_path, pattern=f'result{ext}_*.csv', fname=f'result{ext}.csv', delete=True)

def main_insitu(part, parts):

    ds_sat = io()
    ds_ins = ISMN_io()

    lut = pd.read_csv(r"D:\data_sets\auxiliary\lut_ease_025_dgg.csv", index_col=0)

    idx = []
    cells = []
    for i, row in ds_ins.list.iterrows():
        tmp = lut[(lut.ease_col==row.ease_col)&(lut.ease_row==row.ease_row)]
        if len(tmp) > 0:
            idx += [i]
            cells += [tmp['cell_025'].values[0]]
    ds_ins.list = ds_ins.list.loc[idx,:]
    ds_ins.list['cell'] = cells
    ds_ins.list.sort_values('cell', inplace=True)

    # Split station list for parallelization
    subs = (np.arange(parts + 1) * len(ds_ins.list) / parts).astype('int')
    subs[-1] = len(ds_ins.list)
    start = subs[part - 1]
    end = subs[part]
    ds_ins.list = ds_ins.list.iloc[start:end, :]

    res_path = Path(r"H:\work\publications\spatial_selection_bias\results")
    result_file = res_path / ('result_insitu_%i.csv' % part)

    for cnt, (meta, ts_insitu) in enumerate(ds_ins.iter_stations(surface_depth=0.1)):
        print(f'{cnt} / {len(ds_ins.list)}')
        if len(ts_insitu) < 25:
            continue

        idx = lut[(lut.ease_col==meta.ease_col)&(lut.ease_row==meta.ease_row)].index[0]
        df = ds_sat.read_df(idx)
        ts_insitu = ts_insitu.resample('1D').nearest()['sm_surface']
        ts_insitu.name = "ISMN"
        df = pd.concat((df, ts_insitu), axis=1).dropna()
        if len(df) < 25:
            continue

        aux = ds_sat.read_smap_aux(idx)

        r, p = pearsonr(df)
        r = pd.Series(r, index=[f'R_{key}' for key in r._fields])
        p = pd.Series(p, index=[f'p_{key}' for key in p._fields])

        ec_gldas = pd.Series(ecol(df.drop(['ISMN'], axis='columns'), correlated=[['AMSR2','SMAP']]))
        ec_ismn = pd.Series(ecol(df.drop(['GLDAS'], axis='columns'), correlated=[['AMSR2','SMAP']]))

        ec_gldas.index = [f'{name}_ref_GLDAS' for name in ec_gldas.index.values]
        ec_ismn.index = [f'{name}_ref_ISMN' for name in ec_ismn.index.values]

        res = pd.DataFrame(pd.concat([meta,r,p,ec_gldas,ec_ismn,aux]),columns=[meta.name]).transpose()
        res['len'] = len(df)
        md = ds_ins.io[meta.network][meta.station].metadata
        res['lc_2010'] = md['lc_2010'][1]
        res['climate_KG'] = md['climate_KG'][1]
        res['clay_fraction'] = md['clay_fraction'][1] if md['clay_fraction'][0] == 'clay_fraction' else md['clay_fraction'][0][1]
        res['sand_fraction'] = md['sand_fraction'][1] if md['sand_fraction'][0] == 'sand_fraction' else md['sand_fraction'][0][1]
        res['organic_carbon'] = md['organic_carbon'][1] if md['organic_carbon'][0] == 'organic_carbon' else md['organic_carbon'][0][1]
        res['elevation'] = md['elevation'][1]

        if not result_file.exists():
            res.to_csv(result_file, float_format='%0.4f')
        else:
            res.to_csv(result_file, float_format='%0.4f', mode='a', header=False)

    ds_sat.close()
    ds_ins.close()

def main_global(part, parts):

    ds_sat = io()

    lut = pd.read_csv(r"D:\data_sets\auxiliary\lut_ease_025_dgg.csv", index_col=0)

    # Split station list for parallelization
    subs = (np.arange(parts + 1) * len(lut) / parts).astype('int')
    subs[-1] = len(lut)
    start = subs[part - 1]
    end = subs[part]
    lut = lut.iloc[start:end, :]

    res_path = Path(r"H:\work\publications\spatial_selection_bias\results")
    result_file = res_path / ('result_global_%i.csv' % part)

    for cnt, (idx, meta) in enumerate(lut.iterrows()):
        print(f'{cnt} / {len(lut)}')

        df = ds_sat.read_df(idx).dropna()
        if len(df) < 25:
            continue

        try:
            aux = ds_sat.read_smap_aux(idx)

            r, p = pearsonr(df)
            r = pd.Series(r, index=[f'R_{key}' for key in r._fields])
            p = pd.Series(p, index=[f'p_{key}' for key in p._fields])

            ec = pd.Series(ecol(df, correlated=[['AMSR2','SMAP']]))

            res = pd.DataFrame(pd.concat([meta,r,p,ec,aux]),columns=[meta.name]).transpose()
            res['len'] = len(df)

            if not result_file.exists():
                res.to_csv(result_file, float_format='%0.4f')
            else:
                res.to_csv(result_file, float_format='%0.4f', mode='a', header=False)
        except:
            print(f'error {idx}')

    ds_sat.close()

if __name__=='__main__':
    run(8, insitu=False)