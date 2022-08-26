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
fname = 'nuecc__precut.root'
fname_signal = 'nue__precut.root'

start = time()

#Load files
#nue_tree = uproot.open(f'{data_dir}/nue.root:NuEScat/Event;1')
nuecc_tree = uproot.open(f'{data_dir}/nuecc.root:NuEScat/Event;1')

#Genie keys (only using genie for now)
genie_keys = [key for key in nuecc_tree.keys() if 'genie' in key]
genie_keys.extend(['run','subrun','event','ccnc_truth'])

#Get awkward arrays
#nue_ak = nue_tree.arrays(genie_keys)
nuecc_ak = nuecc_tree.arrays(genie_keys)

#Make nuecc cuts
cut1 = nuepic.cut_pdg_event(nuecc_ak,211,E_threshold=E_threshold) #Pions
cut2 = nuepic.cut_pdg_event(cut1,2212,E_threshold=E_threshold) #Protons
cut3 = nuepic.cut_ccnc_event(cut2) #Keep only cc events

#Convert to awkward0 to save
nuecc_ak0 = ak.to_awkward0(cut3)

end = time() #Time it
pic.print_stars()
print(f'Cuts took {end-start:.2f}s. Reduced events from {len(nuecc_ak)} to {len(cut3)} events for nue background\n')


#Save to root file
start = time()

file = uproot3.recreate(f'{fname}')
file['genie'] = uproot3.newtree({
  'genie_no_primaries':np.int32,
  'genie_primaries_pdg':uproot3.newbranch(np.dtype('f8'),size='n'),
  'genie_Eng':uproot3.newbranch(np.dtype('f8'),size='n'),
  'genie_Px':uproot3.newbranch(np.dtype('f8'),size='n'),
  'genie_Py':uproot3.newbranch(np.dtype('f8'),size='n'),
  'genie_Pz':uproot3.newbranch(np.dtype('f8'),size='n'),
  'genie_P':uproot3.newbranch(np.dtype('f8'),size='n'),
  'genie_status_code':uproot3.newbranch(np.dtype('f8'),size='n'),
  'genie_mass':uproot3.newbranch(np.dtype('f8'),size='n'),
  'genie_trackID':uproot3.newbranch(np.dtype('f8'),size='n'),
  'genie_ND':uproot3.newbranch(np.dtype('f8'),size='n'),
  'genie_mother':uproot3.newbranch(np.dtype('f8'),size='n'),
  'run':np.int32,
  'subrun':np.int32,
  'event':np.int32,
  'ccnc_truth':np.int32
}) #Make tree
file['genie'].extend({
  'genie_no_primaries':nuecc_ak0[:]['genie_no_primaries'],
  'genie_primaries_pdg':nuecc_ak0[:]['genie_primaries_pdg'],
  'genie_Eng':nuecc_ak0[:]['genie_Eng'],
  'genie_Px':nuecc_ak0[:]['genie_Px'],
  'genie_Py':nuecc_ak0[:]['genie_Py'],
  'genie_Pz':nuecc_ak0[:]['genie_Pz'],
  'genie_P':nuecc_ak0[:]['genie_P'],
  'genie_status_code':nuecc_ak0[:]['genie_status_code'],
  'genie_mass':nuecc_ak0[:]['genie_mass'],
  'genie_trackID':nuecc_ak0[:]['genie_trackID'],
  'genie_ND':nuecc_ak0[:]['genie_ND'],
  'genie_mother':nuecc_ak0[:]['genie_mother'],
  'run':nuecc_ak0[:]['run'],
  'subrun':nuecc_ak0[:]['subrun'],
  'event':nuecc_ak0[:]['event'],
  'ccnc_truth':nuecc_ak0[:]['ccnc_truth'],
  'n':nuecc_ak0[:]['genie_primaries_pdg'].counts
}) #Push data to branches
file.close()#Save file

end = time()
os.system(f'mv {fname} {data_dir}/')
print(f'Saved file in {data_dir}/ {end-start:.2f}s')
#nue_ak.to_pickle(f'{data_dir}/nue.pkl')








