import warnings
warnings.filterwarnings("ignore")

import os

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

import matplotlib.pyplot as plt


def download(date=None):

    if date:
        t = datetime.strptime(date, '%Y-%m-%d')
    else:
        t = datetime.now() - timedelta(days=1)

    fname_in = 'https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-%i-%02i-%02i.xlsx' % (t.year, t.month, t.day)
    fname_out = '/Users/u0116961/data_sets/COVID19/%i_%02i_%02i.xlsx' %  (t.year, t.month, t.day)

    if os.path.isfile(fname_out):
        print('File already exists.')
        return

    cmd = 'wget ' + fname_in + ' -O ' + fname_out

    os.system(cmd)


def read_data(date=None, mode='cases', countries=None, normalized=True):

    if date:
        t = datetime.strptime(date, '%Y-%m-%d')
    else:
        t = datetime.now() - timedelta(days=1)

    fname = '/Users/u0116961/data_sets/COVID19/%i_%02i_%02i.xlsx' %  (t.year, t.month, t.day)
    if not os.path.exists(fname):
        download(date)

    if normalized:
        pop_dens = {'Italy': 206, 'Germany': 240, 'Belgium': 383, 'Austria': 109, 'Croatia': 73, 'Netherlands': 508,
                              'United_States_of_America': 40, 'Sweden': 22, 'United_Kingdom': 274, 'Nigeria': 215}
        pop = pd.read_csv('/Users/u0116961/data_sets/COVID19/population_stats.csv', index_col=0, header=2)['2018'] / 1e6
        idx_list = pop.index.tolist()
        idx = idx_list.index('United States')
        idx_list[idx] = 'United_States_of_America'
        idx = idx_list.index('United Kingdom')
        idx_list[idx] = 'United_Kingdom'
        pop.index = idx_list

    df = pd.read_excel(fname, index_col=0)
    df.columns.values[5] = 'Countries'
    df.columns.values[df.columns == 'Cases'] = 'cases'
    df.columns.values[df.columns == 'Deaths'] = 'deaths'
    df.index.name = ''

    idx = df.index.unique().sort_values()
    if not countries:
        countries = np.sort(np.unique(df['Countries']))

    data = pd.DataFrame(index=idx, columns=countries)
    growth = pd.DataFrame(index=idx, columns=countries)

    for country in countries:
        data.loc[:, country] = df[df['Countries'] == country].reindex(idx)[mode.lower()]
        if normalized:
            # data.loc[:, country] /= (pop[country] * pop_dens[country])
            data.loc[:, country] /= (pop[country])
    data.replace(np.nan, 0.0, inplace=True)

    # calculate cummulative values
    for i in range(1, len(idx)):
        data.iloc[i, :] += data.iloc[i - 1, :]
        growth.iloc[i, :] = (data.iloc[i, :] / data.iloc[i - 1, :])
    growth.replace(np.inf, np.nan, inplace=True)

    return data, growth


def calc_stats(growth, countries=None, date_from=None, date_to=None, output=False, mean_bias=0.0):

    if not countries:
        countries = growth.columns
    if not date_from:
        date_from = growth.index[0]
    else:
        date_from = pd.to_datetime(date_from)
    if not date_to:
        date_to = growth.index[-1]
    else:
        date_to = pd.to_datetime(date_to)

    xgrowth = growth.copy()
    xgrowth[xgrowth>1.7] = 1.7

    # for ctr in countries:
    growth_mean = np.nanmean(xgrowth.loc[date_from:date_to, countries]) - mean_bias
    std = np.nanstd(xgrowth.loc[date_from:date_to, countries])
    growth_min = growth_mean - 1*std
    growth_max = growth_mean + 1*std

    if output:
        print('Growth rate statistics:')
        print('Overall:'.ljust(13,' ') + 'Mean: %.3f     Std.dev:  %.3f' % (growth_mean, std))
        for ctr in countries:
            ctr_mean = np.nanmean(xgrowth.loc[date_from:date_to, ctr])
            ctr_std = np.nanstd(xgrowth.loc[date_from:date_to, ctr])
            print((ctr +':').ljust(13,' ') + 'Mean: %.3f     Std.dev:  %.3f' % (ctr_mean, ctr_std))
        print('')

    return growth_min, growth_mean, growth_max


def make_predictions(data, growth, countries, stat_start, date_from, date_to, mean_bias=0.0, output=False):

    if not date_from:
        date_from = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    growth_min, growth_mean, growth_max = calc_stats(growth, countries, date_from=stat_start, mean_bias=mean_bias, output=output)

    idx = pd.date_range(start=date_from, end=date_to)
    cols = [ctr + '_min' for ctr in countries] + \
           [ctr + '_mean' for ctr in countries] + \
           [ctr + '_max' for ctr in countries]

    preds_data = pd.DataFrame(columns=cols, index=idx)
    preds_data[[ctr + '_min' for ctr in countries]] = data.loc[date_from].values.reshape(1,-1).repeat(len(idx), axis=0) * (np.full(len(idx),growth_min)**np.arange(len(idx))).reshape(-1,1)
    preds_data[[ctr + '_mean' for ctr in countries]] = data.loc[date_from].values.reshape(1,-1).repeat(len(idx), axis=0) * (np.full(len(idx),growth_mean)**np.arange(len(idx))).reshape(-1,1)
    preds_data[[ctr + '_max' for ctr in countries]] = data.loc[date_from].values.reshape(1,-1).repeat(len(idx), axis=0) * (np.full(len(idx),growth_max)**np.arange(len(idx))).reshape(-1,1)

    preds_growth = pd.DataFrame(index=idx, columns=countries)
    preds_growth['mean'] = growth_mean
    preds_growth['min'] = growth_min
    preds_growth['max'] = growth_max

    return preds_data, preds_growth

def make_predictions_per_country(data, growth, countries, stat_start, date_from, date_to, mean_bias=0.0, output=False):

    if not date_from:
        date_from = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    idx = pd.date_range(start=date_from, end=date_to)
    cols = [ctr + '_mean' for ctr in countries]
    preds_data = pd.DataFrame(columns=cols, index=idx)

    for ctr in countries:
        growth_min, growth_mean, growth_max = calc_stats(growth, ctr, date_from=stat_start, mean_bias=mean_bias, output=output)
        preds_data[ctr + '_mean'] = data.loc[date_from, ctr].repeat(len(idx), axis=0) * (np.full(len(idx),growth_mean)**np.arange(len(idx)))

    return preds_data, None

def print_time_to_threshold(preds, countries, threshold):

    print('%i cases reached:' % threshold)
    for ctr in countries:
        t_min = preds[preds[ctr + '_max'] >= threshold]
        t_max = preds[preds[ctr + '_min'] >= threshold]
        t_min = t_min.index[0].strftime('%Y-%m-%d') if len(t_min) > 0 else '?'
        t_max = t_max.index[0].strftime('%Y-%m-%d') if len(t_max) > 0 else '?'
        print((ctr+':').ljust(13,' ') + t_min + ' - ' + t_max)
    print('')


def plot(yscale='log', countries=None):

    if not countries:
        countries = ['Italy', 'Germany', 'Belgium', 'Austria', 'Croatia', 'Netherlands']

    normalized = True

    t = datetime.now()

    stat_start = (t - timedelta(days=7)).strftime('%Y-%m-%d')

    pred_start = (t - timedelta(days=7)).strftime('%Y-%m-%d')
    pred_end = (t + timedelta(days=180)).strftime('%Y-%m-%d')

    plot_start = (t - timedelta(days=28)).strftime('%Y-%m-%d')
    plot_end = (t + timedelta(days=14)).strftime('%Y-%m-%d')

    out_file = '/Users/u0116961/data_sets/COVID19/plot_' + yscale + '.png'

    data_cases, growth_cases = read_data(countries=countries, mode='cases', normalized=False)
    data_deaths, growth_deaths = read_data(countries=countries, mode='deaths', normalized=False)

    data_cases_norm, _ = read_data(countries=countries, mode='cases', normalized=True)
    data_deaths_norm, _ = read_data(countries=countries, mode='deaths', normalized=True)

    preds_data_cases, _ = make_predictions_per_country(data_cases, growth_cases, countries, stat_start, pred_start, pred_end)
    preds_data_cases_norm, _ = make_predictions_per_country(data_cases_norm, growth_cases, countries, stat_start, pred_start, pred_end)

    preds_data_deaths, _ = make_predictions_per_country(data_deaths, growth_deaths, countries, stat_start, pred_start, pred_end)
    preds_data_deaths_norm, _= make_predictions_per_country(data_deaths_norm, growth_deaths, countries, stat_start, pred_start, pred_end)

    # print_time_to_threshold(preds_data_cases, countries, 10000)
    # print_time_to_threshold(preds_data_cases, countries, 60000)

    f = plt.figure(figsize=(24,12))

    fontsize = 12
    markersize = 6
    ylim_growth_cases = (0.99,1.06)
    ylim_growth_deaths = (0.99,1.05)

    ylim_data_cases = (2e3, 8e6) if yscale == 'log' else (0, 350000)
    ylim_data_deaths = (8e1, 2e5) if yscale == 'log' else (0, 45000)

    ylim_data_cases_norm = (5e2, 2e4) if yscale == 'log' else (0, 7000)
    ylim_data_deaths_norm = (2e1, 1e3) if yscale == 'log' else (0, 700)

    # population density normalized
    # ylim_data_cases_norm = (1e3, 2e5) if yscale == 'log' else (-500, 100000)
    # ylim_data_deaths_norm = (1e2, 2e4) if yscale == 'log' else (-100, 1000)

    # ----------------------------------------------------------------------
    # --- Growth rate ---
    plt.subplot(2,3,1)
    growth_cases.loc[plot_start::,:].plot(ax=plt.gca(),fontsize=fontsize, linewidth=2.5, marker='o', markersize=markersize-1)
    colors = [line.get_color() for line in plt.gca().lines]
    # preds_growth_cases['mean'].plot(ax=plt.gca(), legend=False, color='black', linestyle='--', linewidth=1)
    # plt.fill_between(preds_growth_cases.index, preds_growth_cases['min'].values, preds_growth_cases['max'].values, alpha=0.15, color='black')
    plt.title('Growth rate cases',fontsize=fontsize+2)
    plt.axhline(1, color='black', linestyle='-', linewidth=1.5)
    plt.xlim(plot_start, plot_end)
    plt.ylim(ylim_growth_cases)
    plt.gca().set_xticks([])

    plt.subplot(2,3,4)
    growth_deaths.loc[plot_start::,:].plot(ax=plt.gca(),fontsize=fontsize, linewidth=2.5, marker='o', markersize=markersize-1,legend=False)
    colors = [line.get_color() for line in plt.gca().lines]
    # preds_growth_deaths['mean'].plot(ax=plt.gca(), legend=False, color='black', linestyle='--', linewidth=1)
    # plt.fill_between(preds_growth_deaths.index, preds_growth_deaths['min'].values, preds_growth_deaths['max'].values, alpha=0.15, color='black')
    plt.title('Growth rate deaths',fontsize=fontsize+2)
    plt.axhline(1, color='black', linestyle='-', linewidth=1.5)
    plt.xlim(plot_start, plot_end)
    plt.ylim(ylim_growth_deaths)

    # ----------------------------------------------------------------------
    # --- Predictions ---
    plt.subplot(2, 3, 2)
    data_cases.loc[plot_start:, :].plot(ax=plt.gca(),fontsize=fontsize, linewidth=2.5, marker='o', markersize=markersize, linestyle='-',legend=False)
    preds_data_cases.loc[:,[ctr + '_mean' for ctr in countries]].plot(ax=plt.gca(),legend=False, color=colors,fontsize=fontsize, linewidth=1, linestyle='--')
    # for ctr, col in zip(countries, colors):
    #     plt.fill_between(preds_data_cases.index, preds_data_cases[ctr + '_min'].values.astype('float'), preds_data_cases[ctr + '_max'].values.astype('float'), alpha=0.15, color=col)
    plt.title('Total cases',fontsize=fontsize+2)
    plt.yscale(yscale)
    plt.xlim(plot_start, plot_end)
    plt.ylim(ylim_data_cases)
    plt.gca().set_xticks([])

    plt.subplot(2, 3, 5)
    data_deaths.loc[plot_start:, :].plot(ax=plt.gca(),fontsize=fontsize, linewidth=2.5, marker='o', markersize=markersize, linestyle='-',legend=False)
    preds_data_deaths.loc[:,[ctr + '_mean' for ctr in countries]].plot(ax=plt.gca(),legend=False, color=colors,fontsize=fontsize, linewidth=1, linestyle='--')
    # for ctr, col in zip(countries, colors):
    #     plt.fill_between(preds_data_deaths.index, preds_data_deaths[ctr + '_min'].values.astype('float'), preds_data_deaths[ctr + '_max'].values.astype('float'), alpha=0.15, color=col)
    plt.title('Total deaths' ,fontsize=fontsize+2)
    plt.yscale(yscale)
    plt.xlim(plot_start, plot_end)
    plt.ylim(ylim_data_deaths)

    # ----------------------------------------------------------------------
    # --- Predictions normalized---
    plt.subplot(2, 3, 3)
    data_cases_norm.loc[plot_start:, :].plot(ax=plt.gca(), fontsize=fontsize, linewidth=2.5, marker='o', markersize=markersize, linestyle='-', legend=False)
    preds_data_cases_norm.loc[:, [ctr + '_mean' for ctr in countries]].plot(ax=plt.gca(), legend=False, color=colors,fontsize=fontsize, linewidth=1, linestyle='--')
    # for ctr, col in zip(countries, colors):
    #     plt.fill_between(preds_data_cases.index, preds_data_cases[ctr + '_min'].values.astype('float'), preds_data_cases[ctr + '_max'].values.astype('float'), alpha=0.15, color=col)
    plt.title('Total cases / 1 million population', fontsize=fontsize + 2)
    plt.yscale(yscale)
    plt.xlim(plot_start, plot_end)
    plt.ylim(ylim_data_cases_norm)
    plt.gca().set_xticks([])

    plt.subplot(2, 3, 6)
    data_deaths_norm.loc[plot_start:, :].plot(ax=plt.gca(), fontsize=fontsize, linewidth=2.5, marker='o', markersize=markersize, linestyle='-', legend=False)
    preds_data_deaths_norm.loc[:, [ctr + '_mean' for ctr in countries]].plot(ax=plt.gca(), legend=False, color=colors, fontsize=fontsize, linewidth=1, linestyle='--')
    # for ctr, col in zip(countries, colors):
    #     plt.fill_between(preds_data_deaths.index, preds_data_deaths[ctr + '_min'].values.astype('float'), preds_data_deaths[ctr + '_max'].values.astype('float'), alpha=0.15, color=col)
    plt.title('Total deaths / 1 million population', fontsize=fontsize + 2)
    plt.yscale(yscale)
    plt.xlim(plot_start, plot_end)
    plt.ylim(ylim_data_deaths_norm)

    # plt.show()
    f.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()


if __name__=='__main__':

    countries_select = ['Italy', 'Germany', 'Belgium', 'Austria', 'United_Kingdom', 'Netherlands', 'United_States_of_America', 'Sweden', 'Croatia']

    # countries_select = ['Italy', 'Belgium', 'Austria', 'Croatia', 'Germany', 'Spain', 'Switzerland', 'France', 'Sweden', 'United_Kingdom', 'Hungary', 'Ireland', 'Norway', 'Poland', 'Portugal', 'Netherlands', 'Bulgaria', 'Serbia', 'Bosnia_and_Herzegovina']

    for yscale in ['log']:
        plot(countries=countries_select, yscale=yscale)




