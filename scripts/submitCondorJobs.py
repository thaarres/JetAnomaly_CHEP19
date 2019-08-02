#!/usr/bin/env python
import os, sys, subprocess, re
from glob import glob
import argparse
import subprocess
import time, datetime
import numpy as np

#____________________________________________________________________________________________________________
### processing the external os commands
def processCmd(cmd, quite = 0):
    status, output = subprocess.getstatusoutput(cmd)
    if (status !=0 and not quite):
        print('Error in processing command:\n   ['+cmd+']')
        print('Output:\n   ['+output+'] \n')
    return output

#_____________________________________________________________________________________________________________
#example line: python scripts/submitCondorJobs.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument ('--lumi', type=float, help='Luminosity in pb^-1', default=100)
    parser.add_argument ('--maxtime', help='Max wall run time [s=seconds, m=minutes, h=hours, d=days]', default='8h')
    args = parser.parse_args()

    time_scale = {'s':1, 'm':60, 'h':60*60, 'd':60*60*24}
    maxRunTime = int(args.maxtime[:-1]) * time_scale[args.maxtime[-1]]

    for nBSM in ['AtoZZZ', 'GtoWW', 'GtoBtt', 'GtoNtt']:
        for xsecBSM in np.logspace(-1.2, 1.2, 15):
            subname = 'sub{}{:1.2e}.sub'.format(nBSM, xsecBSM)
            fsub = open(subname, 'w')
            fsub.write('executable    = /afs/cern.ch/user/o/ocerri/cernbox/JetAnomaly/scripts/condorJob.sh')
            fsub.write('\n')
            exec_args = '{} {} {}'.format(nBSM, xsecBSM, args.lumi)
            fsub.write('arguments     = ' + exec_args)
            fsub.write('\n')
            fsub.write('output        = {}/tmp/out/{}_{:1.2e}pb.$(ClusterId).$(ProcId).out'.format(os.environ['HOME'], nBSM, xsecBSM))
            fsub.write('\n')
            fsub.write('error         = {}/tmp/out/{}_{:1.2e}pb.$(ClusterId).$(ProcId).err'.format(os.environ['HOME'], nBSM, xsecBSM))
            fsub.write('\n')
            fsub.write('log           = {}/tmp/out/{}_{:1.2e}pb.$(ClusterId).$(ProcId).log'.format(os.environ['HOME'], nBSM, xsecBSM))
            fsub.write('\n')
            fsub.write('+MaxRuntime   = '+str(maxRunTime))
            fsub.write('\n')
            fsub.write('queue 1')
            fsub.write('\n')
            fsub.close()

            print('Submitting job ' + subname)
            output = os.system('condor_submit ' + subname)
            print('Job submitted')
            os.system('mv ' + subname +' ' + os.environ['HOME']+'/tmp/cfg/'+subname)
