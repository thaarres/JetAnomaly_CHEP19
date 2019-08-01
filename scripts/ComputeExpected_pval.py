from __future__ import print_function
import argparse

Mjj_selection = 1000.
vae_loss = 'mae'
SM_eff = 1e-2
binning=[50, Mjj_selection, 3000]

parser = argparse.ArgumentParser()
parser.add_argument ('--nBSM', type=str, help='Name of the BSM model', default='AtoZZZ')
parser.add_argument ('--xsecBSM', type=float, help='BSM model xsec in pb', default = 0.1)
parser.add_argument ('--lumi', type=float, help='Luminosity in pb^-1', default=100)
args = parser.parse_args()

name_BSM = args.nBSM
xsec_BSM = args.xsecBSM

lumi = args.lumi
N_asymov = 10000
# N_asymov = int(1 / (1 - erf(5.5/np.sqrt(2))))
N_exp_per_xsec = 300

def computeDiscriminatingVar(x):
    out = x[:,-2]/x[:,1]+x[:,-1]/x[:,6]
    out *= 1e5
    return out

import h5py, os, sys, glob, datetime, time
sys.path.append('../lib')
from glob import glob
import numpy as np
import scipy as sp
from scipy.special import erf

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import joblib

dnd = []

gbr_cut = joblib.load('../models/gbr_cut_trained_20190731.joblib')

def getSelection(x_Mjj, x_loss):
    cut = gbr_cut.predict(np.reshape(x_Mjj, (-1,1)))
    return x_loss > cut

def HypotesisTets(h_a_bin_content, h_r_bin_content, eff, N_asymov = 10000, max_N_asymov = 1e8):
    nu = eff*h_r_bin_content/(1-eff)

    probs_obs = sp.stats.poisson.pmf(h_a_bin_content.astype(np.int), nu)
    probs_obs = np.where(probs_obs < 1e-10, np.full_like(probs_obs, 1e-10), probs_obs)
    s_obs = np.sum(-np.log(probs_obs), axis=-1)
    print('S obs:', s_obs)

    N_worse = 0
    N_tot = 0
    loops = 0
    while N_worse < 25 and N_tot < max_N_asymov:
        loops += 1
        if loops > 1 and loops%10 == 0:
            print(N_tot, N_worse)
        if loops == 10:
            print('Increasing by a factor 5 the number of asymov per loop')
            N_asymov *=5
        o_asymov = np.random.poisson(nu, (N_asymov, nu.shape[0]))
        probs = sp.stats.poisson.pmf(o_asymov, nu)
        probs = np.where(probs < 1e-10, np.full_like(probs, 1e-10), probs)
        nll = -np.log(probs)
        s_asymov = np.sum(nll, axis=-1)

        N_worse += np.sum(s_asymov > s_obs)
        N_tot += N_asymov

        if max_N_asymov/N_tot < 25 and (N_worse * (max_N_asymov/N_tot) < 25):
            print('Will never have enough stat - giving up.')
            p_val = max(1, N_worse)/float(N_tot)
            return p_val

    print('Test stat reached after {} loops'.format(loops))

    p_val = max(1, N_worse)/float(N_tot)
    return p_val


sample_loc = {'qcd':'qcd_dEta_signalregion_results.h5',
              'AtoZZZ':'AtoHZ_to_ZZZ_13TeV_PU40_results.h5',
              'GtoWW':'RSGraviton_WW_NARROW_13TeV_PU40_results.h5',
              'GtoBtt':'RSGraviton_tt_BROAD_13TeV_PU40_results.h5',
              'GtoNtt':'RSGraviton_tt_NARROW_13TeV_PU40_results.h5'
             }

sample_nGenEvts = {'qcd': 418*10000,
                   'AtoZZZ': 100*1000,
                   'GtoWW': 100*1000,
                   'GtoBtt': 96*1000,
                   'GtoNtt': 96*1000
                  }

sample_xsec = {'qcd': 8734.0 #pb
              }


data = {}
sample_eff = {}
for n, fname in sample_loc.iteritems():
    f = h5py.File('../data/vae_'+vae_loss+'/'+fname, 'r')
    x = np.array(f.get("results"))

    # apply the dijet mass cut
    sel_Mjj = x[:,0] > Mjj_selection
    x = x[sel_Mjj]
    dVar = computeDiscriminatingVar(x)
    sel = getSelection(x[:,0], dVar)
    x = np.column_stack((x, dVar, sel))

    sample_eff[n] = float(x.shape[0])/sample_nGenEvts[n]

    dt = [(str(s), '<f4') for s in list(f.get("labels")) + ['dVar']]
    dt += [('sel', '?')]
    data[n] = np.array(list(zip(*x.T)), dtype=dt)


print('xsec BSM {:.1e} pb'.format(xsec_BSM))
lumi_text = '{:.1f} fb^{{-1}} (14 TeV), '.format(1e-3*lumi) + name_BSM + ' ({:.2} pb)'.format(xsec_BSM)
SM_samples = ['qcd']

p_val_test = []
for i_exp in range(N_exp_per_xsec):
    print(i_exp)
    d_obs = np.zeros((0,2))

    sample_xsec[name_BSM] = xsec_BSM
    for n in SM_samples + [name_BSM]:
        nExpEvts = lumi*sample_xsec[n]*sample_eff[n]
        nEvts = np.random.poisson(nExpEvts)
        if data[n]['mJJ'].shape[0] < nEvts:
            print('[WARNING] ' + n + ' re-use factor = {:.2f}'.format(float(nEvts)/data[n]['mJJ'].shape[0]))
        evtsIdx = np.random.randint(0, data[n]['mJJ'].shape[0], size=(nEvts,))

        d_aux = np.column_stack((data[n]['mJJ'][evtsIdx], data[n]['sel'][evtsIdx]))
        d_obs = np.concatenate((d_obs, d_aux))

    h_a_bin_content, _ = np.histogram(d_obs[:, 0][d_obs[:,1].astype(np.bool)], bins=binning[0], range=(binning[1], binning[2]))
    h_r_bin_content, _ = np.histogram(d_obs[:, 0][np.logical_not(d_obs[:,1].astype(np.bool))], bins=binning[0], range=(binning[1], binning[2]))

    p_val = HypotesisTets(h_a_bin_content, h_r_bin_content, SM_eff, N_asymov = N_asymov)
    p_val_test.append(p_val)

aux = [lumi, xsec_BSM] + list(np.percentile(p_val_test, [2.5, 16, 50, 84, 97.5]))
aux = np.array(aux)
print(aux)
out_name = '../data/ModelIndepAnalysis/pVal_'
out_name += name_BSM + '{:1.2e}'.format(xsec_BSM) + 'pb' + '_L{:.0f}pb-1.npy'.format(lumi)
np.save(out_name, aux)
