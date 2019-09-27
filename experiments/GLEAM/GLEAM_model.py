
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import timedelta, date

from myprojects.experiments.GLEAM.GLEAM_IO import init, init_static, init_initial, output_ts
from myprojects.experiments.GLEAM.config import config

from pygleam.GLEAM_I import calculate_I_GLEAM
from pygleam.GLEAM_RF import calculate_RF
from pygleam.GLEAM_pRad import calculate_pRAD
from pygleam.GLEAM_Inf import calculate_Inf
from pygleam.GLEAM_readInit import read_initial
from pygleam.GLEAM_sub import calc_sublimation
from pygleam.GLEAM_SWB import update_SM
from pygleam.GLEAM_S import calculate_S_GLEAM
from pygleam.GLEAM_Ep import calculate_Ep_GLEAM, post_Ep
from pygleam.GLEAM_Ea import calculate_Ea
from pygleam.GLEAM_Edis import distribute_E

class GLEAM(object):

    def __init__(self):

        lats = np.arange(-89.875, 90, 0.25)[::-1]
        lons = np.arange(-179.875, 180, 0.25)
        lons, lats = np.meshgrid(lons, lats)
        self.gpis = np.arange(lons.size).reshape(lons.shape)

        # read model configuration
        self.force_path, self.static_file, self.startup_file, \
        self.timeres, self.period, \
        self.nens, \
        self.fAE, self.alpha, self.ldepth = config()


    def proc(self, gpi):

        row, col = np.where(self.gpis == gpi)

        # read forcing data + static variables + startup file for specified grid cell
        datainm = init(self.period, self.force_path, self.nens, col, row)
        datainm.update(init_static(self.static_file, self.nens, col, row))
        datainm.update(init_initial(self.startup_file, self.nens, col, row))

        sv = {}

        for iday,day in enumerate(self.period):

            # print("running GLEAM for", day.strftime("%Y-%m-%d-%H"))
            month = int(day.strftime("%m"))-1

            # read initial conditions, needs to be modified to read from startup file.
            if day == self.period[0]:
                sv['El_frac_B'], sv['El_frac_H'], sv['El_frac_T'], sv['w_frac_B'], sv['w_frac_H'], \
                sv['w_frac_T'], datainm = read_initial(datainm)

                # initialise snow sublimation, here 0, add reading Es from startup file.
                sv['Es'] = np.zeros(shape=sv['El_frac_B'].shape)
                sv['Es'] = np.where(datainm['static.mask_WAT'], np.nan, sv['Es'])

            sv['Ds'] = datainm['forcing.SWE'][:, [iday], :, :]

            if day == self.period[0]:
                sv['Ds0'] = datainm['forcing.SWE'][:, [iday], :, :]
                sv['Dsm0'] = datainm['forcing.SWE'][:, [iday], :, :]

            # snow depth and snow mask
            sv['Ds0'] = np.where(datainm['static.mask_ICE'] > 0, 99, sv['Ds0'])
            sv['Dsm0'] = np.where(datainm['static.mask_ICE'] > 0, 99, sv['Dsm0'])
            sv['Ds'] = np.where(datainm['static.mask_ICE'] > 0, 99, sv['Ds'])
            sv['Ds'] = np.where(sv['Ds'] < 0, 0, sv['Ds'])

            # Smooth snow depth
            sv['Dsa'] = np.maximum(sv['Ds'], sv['Ds0'])

            # Save for the next time step
            sv['Ds0'] = sv['Ds']

            # snow mask
            sv['Msnow'] = np.logical_or(sv['Dsa'] >= 10,  datainm['static.mask_ICE'] > 0)

            # modify land cover fractions
            sv['frac_B'], sv['frac_H'], sv['frac_T'], sv['frac_W'] = \
                calculate_RF(datainm, iday)

            # radiation partitioning
            sv['ae_frac_B'], sv['ae_frac_H'], sv['ae_frac_T'], sv['ae_frac_S'], sv['ae_frac_W'] = \
                calculate_pRAD(self.fAE, datainm, iday, pRad=False)

            # compute Interception
            sv['I'] = calculate_I_GLEAM(P=datainm['forcing.P'][:,[iday],:,:], frac=sv['frac_T'],
                                 LF = datainm['lightning.LF'][:,[month],:,:])

            # mask snow covered areas
            sv['I'] = np.where(sv['Msnow'], 0, sv['I'])

            # compute infiltration
            # area covered with > 1 cm snow = snowfall)
            sv['Ps'] = sv['Msnow'] * datainm['forcing.P'][:, [iday], :, :]

            # Calculate rainfall[mm] (snow < --> no rain)
            datainm['forcing.P'][:, [iday], :, :] = datainm['forcing.P'][:, [iday], :, :] - sv['Ps']

            # Estimate snow depth[mm] = initial depth (yesterday) + snowfall
            # today - sublimation yesterday
            sv['Dsm'] = sv['Dsm0'] + sv['Ps'] - sv['Es']

            # Calculate snow melt[mm] (this part will infiltrate; assumption:
            # if the modelled cumulative snow depth exceeds the observed
            # snow depth, there is melting)
            sv['Fsn'] = sv['Dsm'] - sv['Dsa']
            sv['Fsn'] = np.where(sv['Fsn'] < 0, 0, sv['Fsn'])

            # Assumption: (1) there is no snowmelt in glaciers(we do assume
            # there is sublimation) and (2) if the temperature is below - 10C
            sv['Fsn'] = np.where(datainm['static.mask_ICE'] > 0, 0, sv['Fsn'])
            sv['Fsn'] = np.where(datainm['forcing.AT'][:,[iday],:,:] < -10, 0, sv['Fsn'])

            # Calculate new snow depth [mm] and save variable for tomorrow
            sv['Dsm'] = sv['Dsm'] - sv['Fsn']
            sv['Dsm0'] = sv['Dsm']

            # infiltration function
            sv['Inf_frac_B'], sv['Inf_frac_H'], sv['Inf_frac_T'] =\
                calculate_Inf(P=datainm['forcing.P'][:,[iday],:,:], I=sv['I'], frac_T=sv['frac_T'], Q=0, Fsn=sv['Fsn'], FFb=0)

            # compute soil water balance
            sv['w_frac_T'] = update_SM(sv['w_frac_T'], sv['El_frac_T'], sv['Inf_frac_T'], self.ldepth,
                       datainm['static.SM_wlp'], datainm['static.SM_flc'], datainm['static.SM_por'])

            sv['w_frac_H'] = update_SM(sv['w_frac_H'], sv['El_frac_H'], sv['Inf_frac_H'],
                       self.ldepth[:, :, 0:2, :], datainm['static.SM_wlp'][:, :, 0:2, :],
                       datainm['static.SM_flc'][:, :, 0:2, :], datainm['static.SM_por'][:, :, 0:2, :])

            sv['w_frac_B'] = update_SM(sv['w_frac_B'], sv['El_frac_B'], sv['Inf_frac_B'],
                       self.ldepth[:, :, 0:1, :], datainm['static.SM_res'][:, :, 0:1, :],
                       datainm['static.SM_flc'][:, :, 0:1, :], datainm['static.SM_por'][:, :, 0:1, :])

            # aggregate to pixel level
            # where is this done?!

            # calculate vegetation stress
            sv['S'], sv['S_frac_T'], sv['S_frac_H'], sv['S_frac_B'] = \
               calculate_S_GLEAM(datainm['forcing.VOD'][:,[iday],:,:], datainm['static.VOD_Q01'], datainm['static.VOD_Q99'],
               np.concatenate([sv['w_frac_T'], sv['w_frac_H'], sv['w_frac_B']], axis=2),
               datainm['static.SM_wlp'], datainm['static.SM_crt'], datainm['static.SM_res'],
               np.concatenate([sv['frac_T'], sv['frac_H'], sv['frac_B']], axis=2))

            # Calculate potential evaporation(Priestley - Taylor) and snow sublimation

            # Comment on the Priestley and Taylor coefficient(alpha):
            # Values of alpha around 1.26(Priestley and Taylor, 1972) are
            # often chosen as representative for most ecosystems.Forests,
            # however, are often subjected to lower values of potential
            # transpiration due to the more conservative behavior of tree
            # stomata(see e.g. Kelliger et al., 1993,
            # Shuttleworth and Calder, 1979, Teuling et al., 2010). In GLEAM,
            # the alpha for tall vegetation is calculated as the average of the
            # findings by McNaughton and Black(1973), Shuttleworth et al.
            # (1984), Viswanadham et al.(1991), Diawara et al.(1991), and
            # Eaton et al.(2001)(last two found in Komatsu et al., 2005)
            # these studies report values for forest stands during periods of
            # no rainfall(i.e.no interception loss) and potential evaporation
            # (i.e.no evaporative stress). This results in an alpha value of
            # 0.97 for tall vegetation, with a 0.08 standard deviation from
            # the study sample.

            # Sublimation of ice and snow, Pa in kPa
            sv['Es'] = calc_sublimation(datainm['forcing.AT'][:,[iday],:,:], Pa=101.3, F=sv['ae_frac_S'],
                                        alpha=self.alpha[4], timeres=self.timeres)

            # Apply snow mask
            sv['Es'] = np.where(sv['Es'] > self.timeres / 6, self.timeres / 6, sv['Es']) # very high values sometimes! (check why)
            sv['Es'] = sv['Msnow'] * sv['Es']

            # Set limits
            sv['Es'] = np.where(sv['Es'] > sv['Dsm'], sv['Dsm'], sv['Es']) # no more sublimation than available snow

            # potential evaporation
            sv['Ep_frac_B'] = \
                 calculate_Ep_GLEAM(datainm['forcing.NR'][:,[iday],:,:], datainm['forcing.AT'][:,[iday],:,:],
                sv['ae_frac_B'], self.alpha[2], self.timeres, pRad=False, SWu=None, albedo=None, frac=None)

            sv['Ep_frac_H'] = \
                 calculate_Ep_GLEAM(datainm['forcing.NR'][:,[iday],:,:], datainm['forcing.AT'][:,[iday],:,:],
                sv['ae_frac_H'], self.alpha[1], self.timeres, pRad=False, SWu=None, albedo=None, frac=None)

            sv['Ep_frac_T'] = \
                 calculate_Ep_GLEAM(datainm['forcing.NR'][:,[iday],:,:], datainm['forcing.AT'][:,[iday],:,:],
                sv['ae_frac_T'], self.alpha[0], self.timeres, pRad=False, SWu=None, albedo=None, frac=None)

            sv['E_frac_W'] = \
                 calculate_Ep_GLEAM(datainm['forcing.NR'][:,[iday],:,:], datainm['forcing.AT'][:,[iday],:,:],
                sv['ae_frac_W'], self.alpha[3], self.timeres, pRad=False, SWu=None, albedo=None, frac=None)

            # Process negative evaporation( = condensation): there is no 'stress'
            # for condensation and condensation is 26 % of the radiative term
            # (Priestley and Taylor, 1972)

            # postprocess potential evaporation
            sv['Ep_frac_B'], sv['Ep_frac_H'], sv['Ep_frac_T'], sv['E_frac_W'], sv['Ep'], sv['S_frac_B'],\
            sv['S_frac_H'], sv['S_frac_T'], sv['Es'] = \
            post_Ep(sv['Ep_frac_B'], sv['Ep_frac_H'], sv['Ep_frac_T'], sv['E_frac_W'], sv['S_frac_B'],sv['S_frac_H'],
                           sv['S_frac_T'], sv['frac_B'], sv['frac_H'], sv['frac_T'], sv['frac_W'], Es=sv['Es'], alpha=self.alpha)

            # calculate actual evaporation
            sv['Ea_frac_B'], sv['Ea_frac_H'], sv['Ea_frac_T'] = \
                calculate_Ea(sv['Ep_frac_B'], sv['Ep_frac_H'], sv['Ep_frac_T'], sv['S_frac_B'],
                             sv['S_frac_H'], sv['S_frac_T'], sv['frac_T'], sv['I'])

            # take evaporation from soil layers
            sv['El_frac_B'], sv['El_frac_H'], sv['El_frac_T'] = \
                distribute_E(sv['Ea_frac_T'], sv['Ea_frac_H'], sv['Ea_frac_B'], sv['w_frac_T'],
                             sv['w_frac_H'], sv['w_frac_B'], self.ldepth, datainm['static.SM_wlp'], datainm['static.SM_res'])


            # Distribute condensation (all water enters the first layer)
            ef1 = sv['El_frac_T'][:, :, [0], :]
            eh1 = sv['El_frac_H'][:, :, [0], :]

            ef1 = np.where(sv['Ea_frac_T'] < 0, sv['Ea_frac_T'], ef1)
            eh1 = np.where(sv['Ea_frac_H'] < 0, sv['Ea_frac_H'], eh1)

            sv['El_frac_T'][:, :, [0], :] = ef1
            sv['El_frac_H'][:, :, [0], :] = eh1
            sv['El_frac_B'] = np.where(sv['Ea_frac_B'] < 0, sv['Ea_frac_B'], sv['El_frac_B'])

            # Set evaporation equal to zero for snow - covered areas
            x = sv['El_frac_T'][:,:,[0],:]
            x = np.where(sv['Msnow'], 0, x)
            sv['El_frac_T'][:,:,[0], :] = x

            x = sv['El_frac_T'][:,:,[1], :]
            x = np.where(sv['Msnow'], 0, x)
            sv['El_frac_T'][:,:,[1], :] = x

            x = sv['El_frac_T'][:,:,[2], :]
            x = np.where(sv['Msnow'], 0, x)
            sv['El_frac_T'][:,:,[2], :] = x

            x = sv['El_frac_H'][:,:,[0], :]
            x = np.where(sv['Msnow'], 0, x)
            sv['El_frac_H'][:,:,[0], :] = x

            x = sv['El_frac_H'][:,:,[1], :]
            x = np.where(sv['Msnow'], 0, x)
            sv['El_frac_H'][:,:, [1], :] = x

            sv['El_frac_B'] = np.where(sv['Msnow'], 0, sv['El_frac_B'])

            # Recalculate the total actual evaporation per land cover fraction
            # taking into account the root - zone soil moisture content

            sv['Ea_frac_T'] = np.sum(sv['El_frac_T'], axis=2)
            sv['Ea_frac_H'] = np.sum(sv['El_frac_H'], axis=2)

            sv['Ea_frac_T'] = np.expand_dims(sv['Ea_frac_T'], axis=2)
            sv['Ea_frac_H'] = np.expand_dims(sv['Ea_frac_H'], axis=2)
            sv['Ea_frac_B'] = sv['El_frac_B']

            # Add sublimation to actual evaporation
            sv['Ea_frac_T'] = sv['Ea_frac_T'] + sv['Es']
            sv['Ea_frac_H'] = sv['Ea_frac_H'] + sv['Es']
            sv['Ea_frac_B'] = sv['Ea_frac_B'] + sv['Es']

            # is this all aggregation stuff which can be added to a separate function?
            # Calculation of total actual evaporation at the grid cell
            sv['E_frac_W'] = np.where(np.logical_and(sv['Msnow'], np.isnan(sv['Es'])), sv['Es'], sv['E_frac_W'])
            sv['E_frac_W'] = np.where(sv['E_frac_W'] < 0, 0, sv['E_frac_W'])

            sv['E'] = sv['Ea_frac_T'] * sv['frac_T'] + sv['Ea_frac_H'] * sv['frac_H'] + sv['Ea_frac_B'] * \
                      sv['frac_B'] + sv['E_frac_W'] * sv['frac_W'] + sv['I']
            sv['E'] = np.where(np.isnan(sv['E']), sv['E_frac_W'], sv['E'])

            # Aggregating soil moisture per layer to pixel scale
            sv['w1'] = (sv['frac_T'] * sv['w_frac_T'][:, :, [0], :] + sv['frac_H'] * sv['w_frac_H'][:, :, [0], :]
                        + sv['frac_B'] * sv['w_frac_B']) / \
                       ( sv['frac_T'] + sv['frac_H'] + sv['frac_B'])

            sv['w2'] = (sv['frac_T'] * sv['w_frac_T'][:, :, [1], :] + sv['frac_H'] * sv['w_frac_H'][:, :, [1], :]) / \
                       (sv['frac_T'] + sv['frac_H'])

            ww2 = (sv['w_frac_T'][:, :, [1], :] + sv['w_frac_T'][:, :, [0], :]) / 2

            sv['w2'] = np.where((sv['frac_T'] + sv['frac_H'] == 0), ww2, sv['w2'])
            sv['w3'] = sv['w_frac_T'][:, :, [2], :]

            # Calculating volume of water for each fraction and the total volume of water
            vB = sv['w_frac_B'] * self.ldepth[:, :, [0], :]

            vH = sv['w_frac_H'][:, :, [0], :] * self.ldepth[:, :, [0], :] + sv['w_frac_H'][:,:,[1], :] * self.ldepth[:, :, [1], :]
            vTC = sv['w_frac_T'][:, :, [0], :] * self.ldepth[:, :, [0], :] + sv['w_frac_T'][:, :, [1], :] * \
                  self.ldepth[:, :, [1], :] + sv['w_frac_T'][:, :, [2], :] * self.ldepth[:, :, [2], :]

            vT = (sv['frac_B'] * vB + sv['frac_H'] * vH + sv['frac_T'] * vTC) / (sv['frac_B'] + sv['frac_H'] + sv['frac_T'])

            # Calculating the total depth of the soil (weighted average per fraction)
            dT = (sv['frac_B'] * self.ldepth[:, :, [0], :] + sv['frac_H'] *
               np.sum(self.ldepth[:, :, [1], :], axis=2, keepdims=True) +
               sv['frac_T'] * np.sum(self.ldepth, axis=2, keepdims=True)) / (sv['frac_B'] + sv['frac_H'] + sv['frac_T'])

            # Calculating the root - zone soil moisture
            sv['wr'] = vT / dT

            # Calculating the evaporative fraction at the pixel level
            # Calculation of latent heat of vaporization
            # make this into function.

            lambdax = 1.91846 * np.power((datainm['forcing.AT'][:,[iday],:,:] + 273.15) /
                                         ((datainm['forcing.AT'][:,[iday],:,:] + 273.15) - 33.91), 2)

            # Calculation AE at the pixel level
            sv['AE'] = sv['frac_B'] * sv['ae_frac_B'] + sv['frac_H'] * sv['ae_frac_H'] + sv['frac_T'] * \
                       sv['ae_frac_T'] + sv['frac_W'] * sv['ae_frac_W']

            # Conversion E to W / m2
            E_wpsm = sv['E'] * lambdax * 1e6 / (3600 * 24)

            # Calculation evaporative fraction
            sv['EF'] = E_wpsm / sv['AE']

            # Mask data
            # maskout = np. Msnow | MiceP | Msea | AE < 0 | AE_frac_B < 0 | AE_frac_H < 0 |
            # AE_frac_T < 0 | AE_frac_W < 0 | E < 0 | Ea_frac_B < 0 | Ea_frac_H < 0 |
            # Ea_frac_T < 0 | E_frac_W < 0 | EF < 0
            # EF(mask) = NaN

            # Set maximum
            sv['EF'] = np.where(sv['EF'] > 1, 1, sv['EF'])

            yield sv

