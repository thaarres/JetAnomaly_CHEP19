#!/bin/bash
source /afs/cern.ch/user/o/ocerri/.bash_profile

cd /afs/cern.ch/user/o/ocerri/cernbox/JetAnomaly/scripts

python ComputeExpected_pval.py -nBSM $1 -xsecBSM $2 --lumi $3
