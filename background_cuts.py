import numpy as np
import sys
import awkward as ak
sys.path.append('/sbnd/app/users/brindenc/mypython') #My utils path
from bc_utils.nueutils import pic as nuepic
from bc_utils.nueutils import plotters as nueplotters
from bc_utils.utils import pic,plotters
import uproot
from time import time
import numpy as np
import uproot3
import os

#Constants/parameters
E_threshold = 0.02 #GeV energy threshold for visible hadron
data_dir = '/pnfs/sbnd/persistent/users/brindenc/analyze_sbnd/nue/v09_43_00/data'
fname = 'nuecc__postcut.root'
fname_signal = 'nue__postcut.root'

#Load files
#nue_tree = uproot.open(f'{data_dir}/nue__precut.root:NuEScat/Event;1')
nuecc_tree = uproot.open(f'{data_dir}/nuecc__precut.root:NuEScat/Event;1')

#Genie keys (only using genie for now)
keys = nuecc_tree.keys()
genie_keys = [key for key in keys if 'genie' in key]
genie_keys.extend(['run','subrun','event','ccnc_truth'])

#Get awkward arrays
#nue_ak = nue_tree.arrays(genie_keys)
nuecc_ak = nuecc_tree.arrays(genie_keys)







