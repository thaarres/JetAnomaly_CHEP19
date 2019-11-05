import numpy as np
import scipy as sp
from scipy import stats
import uproot
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy.special import erf

from tqdm import tqdm
from multiprocessing import Process, Manager, Value
matplotlib.rcParams.update({'font.size': 18})
import h5py, os, sys, glob, datetime, time
import subprocess


def HypotesisTets(h_a_bin_content, h_r_bin_content, eff=0.01, N_asymov = 10000, max_N_asymov = 1e9, show_hist=True):
    
    scale = np.sum(h_a_bin_content)/np.sum(h_r_bin_content)
    #nu = eff*h_r_bin_content/(1-eff)
    nu = scale*h_r_bin_content

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
            factor = int(25/max(1, N_worse))
            print('Increasing by a factor {} the number of asymov per loop'.format(factor))
            N_asymov *= factor
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
    print("p_val {} from {} toys MC".format(p_val, N_tot))  
    if show_hist:
        plt.figure()
        binContent, _, _ = plt.hist(s_asymov, label='Distribution assuming eff={:.1f}%'.format(100*eff))
        plt.plot([s_obs, s_obs], [0,np.max(binContent)], label='Observed')
        plt.legend(loc='best')
        plt.xlabel('Test statistic')
        plt.ylabel('Entries')

    return p_val
from hist_data import h_a, h_r

pval_dict = {}
for xsec in h_a:
    pval_dict[xsec] = []
    for i_exp in range(10):
        print("-------------------------")
        print("Cross section: {}".format(xsec))
        pval_dict[xsec].append(HypotesisTets(np.asarray(h_a[xsec][i_exp]), np.asarray(h_r[xsec][i_exp]), N_asymov = 10000, show_hist=False))

print(pval_dict)
