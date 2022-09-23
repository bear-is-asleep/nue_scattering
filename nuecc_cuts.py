import sys
sys.path.append('/sbnd/app/users/brindenc/mypython') #My utils path
from bc_utils.CAFana import pic as CAFpic
from bc_utils.utils import pic
from time import time
import numpy as np
import pandas as pd
import uproot
import seaborn as sns


#Constants/parameters
pot1 = 15e16
pot2 = 20e20
SEED = 42
n = 5 #number of random universes
ntrk = 10 #Size of shower confusion matrix
nshw = 10 #Size of track confusion matrix
E_threshold = 0.021 #GeV energy threshold for visible hadron (ArgoNeut)
E_threshold_exotic = E_threshold #Set same for now
#DATA_DIR = '/pnfs/sbnd/persistent/users/brindenc/analyze_sbnd/nue/v09_43_00/data'
DATA_DIR = '../test_fcl/'
savename = 'nuecc.pkl'
savename_Np = 'nueccNp.pkl'
#savename_signal = 'nue.pkl'
exotic_hadrons = [3112,321,4122,3222] #Make cuts on these?
print_results = False
Etheta1 = 0.003 #Gev rad^2
Etheta2 = 0.004 #Gev rad^2
Etheta3 = 0.001 #Gev rad^2
razzlecutoff = 0.6 #best score required to be electron

t1 = time()

#Load files

#nu + e
# nue_tree1 = uproot.open(f'{DATA_DIR}CAFnue1.root:recTree;1')
# CAFkeys = nue_tree1.keys()

# mcnu_keys = [key for key in CAFkeys if CAFpic.mcnuprefix in key]
# shw_keys = [key for key in CAFkeys if CAFpic.shwprefix in key]
# trk_keys = [key for key in CAFkeys if CAFpic.trkprefix in key]
# reco_keys = [key for key in CAFkeys if CAFpic.recoprefix in key]

# nue_mc,nue_mc_indeces = CAFpic.get_df(nue_tree1,mcnu_keys)
# nue_shw,nue_shw_indeces = CAFpic.get_df(nue_tree1,shw_keys)
# nue_trk,nue_trk_indeces = CAFpic.get_df(nue_tree1,trk_keys)
# nue_reco,_ = CAFpic.get_df(nue_tree1,reco_keys)

# nue_mc = [df.sort_index() for df in nue_mc]
# nue_shw = [df.sort_index() for df in nue_shw]
# nue_trk = [df.sort_index() for df in nue_trk]
# nue_reco = [df.sort_index() for df in nue_reco]

#nuecc
nuecc_tree1 = uproot.open(f'{DATA_DIR}CAFnuecc1.root:recTree;1')
CAFkeys = nuecc_tree1.keys()

mcnu_keys = [key for key in CAFkeys if CAFpic.mcnuprefix in key]
shw_keys = [key for key in CAFkeys if CAFpic.shwprefix in key]
trk_keys = [key for key in CAFkeys if CAFpic.trkprefix in key]
reco_keys = [key for key in CAFkeys if CAFpic.recoprefix in key]

nuecc_mc = CAFpic.get_df(nuecc_tree1,mcnu_keys)
nuecc_nreco = CAFpic.get_df(nuecc_tree1,[f'{CAFpic.recoprefix}nshw',f'{CAFpic.recoprefix}ntrk',f'{CAFpic.recoprefix}nstub']) #Make seperate tree to count reco objects
nuecc_reco = CAFpic.get_df(nuecc_tree1,reco_keys)

nuecc_mc = [df.sort_index() for df in nuecc_mc]
#nuecc_shw = [df.sort_index() for df in nuecc_shw]
#nuecc_trk = [df.sort_index() for df in nuecc_trk]
nuecc_nreco = nuecc_nreco.sort_index()
nuecc_reco = [df.sort_index() for df in nuecc_reco]

#Find indeces with proper keys, since we have multiple dfs, we should find index with key we're interested
iscc_nuecc_index = CAFpic.find_index_with_key(nuecc_mc,f'{CAFpic.mcnuprefix}iscc') #same as truth pdg
#nshw_nuecc_index = CAFpic.find_index_with_key(nuecc_reco,f'{CAFpic.recoprefix}nshw') #same as ntrk
razzle_nuecc_index = CAFpic.find_index_with_key(nuecc_reco,f'{CAFpic.shwprefix}razzle.electronScore')
pdg_nuecc_index = CAFpic.find_index_with_key(nuecc_mc,f'{CAFpic.primprefix}pdg')
#shwdir_nuecc_index = CAFpic.find_index_with_key() f'{CAFpic.shwprefix}dir.y'
#dazzle_nuecc_index = CAFpic.find_index_with_key(nuecc_trk,f'{CAFpic.trkprefix}dazzle.electronScore')

#POT + event info
pots = nuecc_tree1.arrays('rec.hdr.pot',library='pd')
pot_nuecc = float(pots.iloc[0]) #Get POT
#pot_nue = float(nuecc_tree1.arrays('rec.hdr.pot',library='np')) #Get POT
indeces_nuecc = nuecc_mc[0].index.drop_duplicates()
events_nuecc = len(indeces_nuecc)
#events_nue = len(nue_mc[0].index.drop_duplicates())

#Initialize universe study - truth level
columns = ['Precut','nc','nu+e','pion','proton']
columns.extend(['pdg '+str(x) for x in exotic_hadrons])
columns.extend(['Etheta2 cut1','Etheta2 cut2','Etheta2 cut3'])
columns.extend(['POT'])
events_cut = np.zeros((n,len(columns))) #Display number of events cut for each step

#Initialize universe study - reco level
columns_reco = ['Precut','showers','tracks',f'razzle{razzlecutoff}']
columns_reco.extend(['Etheta2 cut1','Etheta2 cut2','Etheta2 cut3'])
columns_reco.extend(['POT'])
events_cut_reco = np.zeros((n+2,len(columns_reco))) #Display number of events cut for each step
shwconfusion = np.zeros((n+2,nshw,nshw)) #Confusion matrices for number of showers
trkconfusion = np.zeros((n+2,ntrk,ntrk)) #Confusion matrices for number of tracks

seeds = np.arange(n) #Get random seeds

for seed in seeds: #Iterate over n random universes, with different seed each time
  #Get pot normalized event list
  nuecc_drop_indeces = CAFpic.get_pot_normalized_indeces(indeces_nuecc,pot1,pot_nuecc,seed=seed)

  #nuecc
  nuecc_mc_pot1 = [CAFpic.get_df_dropindeces(df,nuecc_drop_indeces) for df in nuecc_mc] #1st ele has most info
  #nuecc_shw_pot1 = [CAFpic.get_df_dropindeces(df,nuecc_drop_indeces) for df in nuecc_shw] #1st ele has most info
  #nuecc_trk_pot1 = [CAFpic.get_df_dropindeces(df,nuecc_drop_indeces) for df in nuecc_trk] #1st ele has most info
  nuecc_reco_pot1 = [CAFpic.get_df_dropindeces(df,nuecc_drop_indeces) for df in nuecc_reco] #1st ele has most info
  nuecc_nreco_pot1 = CAFpic.get_df_dropindeces(nuecc_nreco,nuecc_drop_indeces)
  
  #nue
  # nue_mc_pot1 = [CAFpic.get_pot_normalized_df(df,pot1,pot_nue,events_nue,seed=seed)[0] for df in nue_mc] #1st ele has most info
  # nue_shw_pot1 = [CAFpic.get_pot_normalized_df(df,pot1,pot_nue,events_nue,seed=seed)[0] for df in nue_shw] #0th ele has most info
  # nue_trk_pot1 = [CAFpic.get_pot_normalized_df(df,pot1,pot_nue,events_nue,seed=seed)[0] for df in nue_trk] #0th ele has most info
  # nue_reco_pot1 = [CAFpic.get_pot_normalized_df(df,pot1,pot_nue,events_nue,seed=seed)[0] for df in nue_reco] #1st ele has most info

  #temp names for ease

  t2 = time()

  #print(f'Load time {t2-t1:.2f} s')

  #Make nuecc cuts
  
  #TRUTH LEVEL

  events_cut[seed,0] = nuecc_mc_pot1[0].index.drop_duplicates().shape[0]

  cut1 = nuecc_mc_pot1[iscc_nuecc_index][nuecc_mc_pot1[iscc_nuecc_index].loc[:,f'{CAFpic.mcnuprefix}iscc'] == 1] #Keep only cc events
  cut1_indeces = cut1.index.drop_duplicates()
  events_cut[seed,1] = cut1_indeces.shape[0]
  nuecc_mc_pot1 = [CAFpic.get_df_keepindeces(df,cut1_indeces) for df in nuecc_mc]

  cut2,_ = CAFpic.true_nue(nuecc_mc_pot1[pdg_nuecc_index]) #Drop nue scattering events
  events_cut[seed,2] = cut2.index.drop_duplicates().shape[0]

  cut3,_ = CAFpic.cut_pdg_event(cut2,211,E_threshold=E_threshold) #Pions
  events_cut[seed,3] = cut3.index.drop_duplicates().shape[0]

  cut4,_ = CAFpic.cut_pdg_event(cut3,2212,E_threshold=E_threshold) #Protons
  events_cut[seed,4] = cut4.index.drop_duplicates().shape[0]

  #Exotic particle cuts
  cut5 = cut4.copy() #Set temporary df
  for j,pdg in enumerate(exotic_hadrons): #this will iteravily cut hadrons
    temp_cut,_ = CAFpic.cut_pdg_event(cut5,pdg,E_threshold=E_threshold_exotic)
    cut5 = temp_cut.copy()
    events_cut[seed,j+5] = cut5.index.drop_duplicates().shape[0]
  #DO NOT REASSIGN J IT WILL MESS UP INDEXING!!
  #Print events cut
  if print_results:
    print(f'Precut events {nuecc_mc_pot1[0].index.drop_duplicates().shape[0]}')
    print(f'Cut non cc events {cut1.index.drop_duplicates().shape[0]}')
    print(f'Cut nu+e events {cut2.index.drop_duplicates().shape[0]}')
    print(f'Cut pion events {cut3.index.drop_duplicates().shape[0]}')
    print(f'Cut proton events {cut4.index.drop_duplicates().shape[0]}')
    for i,pdg in enumerate(exotic_hadrons):
      print(f'Cut pdg {pdg} events {events_cut[seed,i+5]}')
  
  #Calc thetae and Etheta^2
  nuecc_1 = CAFpic.calc_thetat(cut5) #Use momentum method
  nuecc_2 = CAFpic.calc_Etheta(nuecc_1)
  #Etheta cuts
  #Calc cuts
  nuecc_cut1,_ = CAFpic.make_cuts(nuecc_2,Etheta=Etheta1)
  events_cut[seed,j+6] = CAFpic.number_events(nuecc_cut1)

  nuecc_cut2,_ = CAFpic.make_cuts(nuecc_2,Etheta=Etheta2)
  events_cut[seed,j+7] = CAFpic.number_events(nuecc_cut2)

  nuecc_cut3,_ = CAFpic.make_cuts(nuecc_2,Etheta=Etheta3)
  events_cut[seed,j+8] = CAFpic.number_events(nuecc_cut3)

  #POT
  events_cut[seed,j+9] = pot1

  #RECO LEVEL

  events_cut_reco[seed,0] = nuecc_nreco_pot1.index.drop_duplicates().shape[0]

  #Get confusion matrices
  shwconfusion[seed,:,:] = CAFpic.get_shw_confusion_matrix(nuecc_nreco_pot1,nuecc_mc_pot1[pdg_nuecc_index],n=nshw)
  trkconfusion[seed,:,:] = CAFpic.get_trk_confusion_matrix(nuecc_nreco_pot1,nuecc_mc_pot1[pdg_nuecc_index],n=ntrk)
  #stubconfusion?

  #Cuts
  cut1,_ = CAFpic.cut_nshws(nuecc_nreco_pot1,1) #Keep only events with one reconstructed shower
  events_cut_reco[seed,1] = cut1.index.drop_duplicates().shape[0]

  cut2,_ = CAFpic.cut_ntrks(cut1,0) #Keep only events with one reconstructed shower
  events_cut_reco[seed,2] = cut2.index.drop_duplicates().shape[0]
  cut2_indeces = cut2.index.drop_duplicates()
  nuecc_reco_pot1 = [CAFpic.get_df_keepindeces(df,cut2_indeces) for df in nuecc_reco]

  cut3 = CAFpic.cut_razzlescore(nuecc_reco_pot1[razzle_nuecc_index],cutoff=razzlecutoff)
  events_cut_reco[seed,3] = cut3.index.drop_duplicates().shape[0]

  #Update dataframe for etheta cut
  #cut3_indeces = cut3.index.drop_duplicates()
  #nuecc_reco_pot1 = [CAFpic.get_df_keepindeces(df,cut3_indeces) for df in nuecc_reco]
  #nuecc_0 = nuecc_reco_pot1[]

  #Calc thetae and Etheta^2
  nuecc_1 = CAFpic.calc_thetat(cut3,return_key=f'{CAFpic.shwprefix}thetal',px_key=f'{CAFpic.shwprefix}dir.x',
    py_key=f'{CAFpic.shwprefix}dir.y',pz_key=f'{CAFpic.shwprefix}dir.z') #Use direction method
  nuecc_2 = CAFpic.calc_Etheta(nuecc_1,return_key=f'{CAFpic.shwprefix}Etheta2',E_key=f'{CAFpic.shwprefix}bestplane_energy',
    theta_l_key=f'{CAFpic.shwprefix}thetal')

  #Etheta2 cuts
  nuecc_cut1,_ = CAFpic.make_reco_cuts(nuecc_2,Etheta=Etheta1)
  events_cut_reco[seed,4] = CAFpic.number_events(nuecc_cut1)

  nuecc_cut2,_ = CAFpic.make_reco_cuts(nuecc_2,Etheta=Etheta2)
  events_cut_reco[seed,5] = CAFpic.number_events(nuecc_cut2)

  nuecc_cut3,_ = CAFpic.make_reco_cuts(nuecc_2,Etheta=Etheta3)
  events_cut_reco[seed,6] = CAFpic.number_events(nuecc_cut3)
    
#Save cut information
events_cut[-2,:] = np.std(events_cut,axis=0) #standard deviation
events_cut[-1,:] = np.mean(events_cut,axis=0) #mean
cuts_df = pd.DataFrame(events_cut,columns=columns)
cuts_df.to_csv(f'truth_cuts.csv')

events_cut_reco[-2,:] = np.std(events_cut_reco,axis=0) #standard deviation
events_cut_reco[-1,:] = np.mean(events_cut_reco,axis=0) #mean
cutsreco_df = pd.DataFrame(events_cut_reco,columns=columns_reco)
cutsreco_df.to_csv(f'reco_cuts.csv')

#Save confusion matrix info
for i in range(nshw):
  for j in range(nshw):
    shwconfusion[-2,i,j] = np.std(shwconfusion[:-2,i,j]) #Mean value of ith jth comp, exclude last two since these are for std,mean
    shwconfusion[-1,i,j] = np.mean(shwconfusion[:-2,i,j]) #Mean value of ith jth comp, exclude last two since these are for std,mean 

#sns.heatmap(shwconfusion[n], annot=True)
np.savetxt('shwconfusion_std.csv',shwconfusion[-2],delimiter=',')
np.savetxt('shwconfusion_mean.csv',shwconfusion[-1],delimiter=',')

for i in range(ntrk):
  for j in range(ntrk):
    trkconfusion[-2,i,j] = np.std(trkconfusion[:-2,i,j]) #Mean value of ith jth comp, exclude last two since these are for std,mean
    trkconfusion[-1,i,j] = np.mean(trkconfusion[:-2,i,j]) #Mean value of ith jth comp, exclude last two since these are for std,mean 

np.savetxt('trkconfusion_std.csv',trkconfusion[-2],delimiter=',')
np.savetxt('trkconfusion_mean.csv',trkconfusion[-1],delimiter=',')








