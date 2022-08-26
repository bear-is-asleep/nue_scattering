import sys
sys.path.append('/sbnd/app/users/brindenc/mypython') #My utils path
from bc_utils.nueutils import pic as nuepic
from bc_utils.utils import pic
from time import time
import numpy as np
import pandas as pd

#Constants/parameters
E_threshold = 0.021 #GeV energy threshold for visible hadron (ArgoNeut)
E_threshold_exotic = E_threshold #Set same for now
data_dir = '/pnfs/sbnd/persistent/users/brindenc/analyze_sbnd/nue/v09_43_00/data'
savename = 'nuecc.pkl'
savename_Np = 'nueccNp.pkl'
savename_signal = 'nue.pkl'
exotic_hadrons = [3112,321,4122,3222] #Make cuts on these?

t1 = time()

#Load files
nue_df,_ = nuepic.get_genie_df(f'{data_dir}/nue.root','NuEScat','Event')
nuecc_df,_ = nuepic.get_genie_df(f'{data_dir}/nuecc.root','NuEScat','Event')
#nuecc_df = pd.read_pickle('data/nuecc.pkl') #temp test

#Sort indeces
nue_df = nue_df.sort_index()
nuecc_df = nuecc_df.sort_index()

t2 = time()
pic.print_stars()

print(f'Load time {t2-t1:.2f} s')

#Make nuecc cuts
events_cut = np.zeros(len(exotic_hadrons)+5) #Display number of events cut for each step
columns = ['Precut','nc','nu+e','pion','proton']
columns.extend(['pdg '+str(x) for x in exotic_hadrons])
events_cut[0] = nuecc_df.index.drop_duplicates().shape[0]

cut1 = nuecc_df[nuecc_df['ccnc_truth'] == 0] #Keep only cc events
events_cut[1] = cut1.index.drop_duplicates().shape[0]

cut2 = nuepic.find_hadron_activity(cut1,drop=True) #Cuts for nu e events seen in background
events_cut[2] = cut2.index.drop_duplicates().shape[0]

cut3 = nuepic.cut_pdg_event(cut2,211,E_threshold=E_threshold) #Pions
events_cut[3] = cut3.index.drop_duplicates().shape[0]

cut4 = nuepic.cut_pdg_event(cut3,2212,E_threshold=E_threshold) #Protons
events_cut[4] = cut4.index.drop_duplicates().shape[0]




#Exotic particle cuts
cut5 = cut4.copy() #Set temporary df
for i,pdg in enumerate(exotic_hadrons): #this will iteravily cut hadrons
  temp_cut = nuepic.cut_pdg_event(cut5,pdg,E_threshold=E_threshold_exotic)
  cut5 = temp_cut.copy()
  events_cut[i+5] = cut5.index.drop_duplicates().shape[0]

#Print events cut
print(f'Precut events {nuecc_df.index.drop_duplicates().shape[0]}')
print(f'Cut non cc events {cut1.index.drop_duplicates().shape[0]}')
print(f'Cut nu+e events {cut2.index.drop_duplicates().shape[0]}')
print(f'Cut pion events {cut3.index.drop_duplicates().shape[0]}')
print(f'Cut proton events {cut4.index.drop_duplicates().shape[0]}')
for i,pdg in enumerate(exotic_hadrons):
  print(f'Cut pdg {pdg} events {events_cut[i+5]}')

#Save cut information
cuts_df = pd.DataFrame(events_cut,index=columns)
#cuts_df.to_csv(f'{data_dir}/nuecc_cuts.csv')
cuts_df.to_csv(f'data/nuecc_cuts.csv')


t3 = time()
print(f'\nCut time {t3-t2:.2f} s')

#Drop at rest electrons (initial state)
nue_0 = nuepic.drop_initial_e(nue_df) #Get rid of initial electron in dataframe

#Add electron count info, just in case 
#nuecc_1 = nuepic.get_electron_count(cut4)
nuecc_1 = nuepic.get_electron_count(cut4) #Select cut stage
nueccNp_1 = nuepic.get_electron_count(cut3) #Select cut stage
nue_1 = nuepic.get_electron_count(nue_0)

#Get background and scat types
nuecc_1.loc[:,'background_type'] = 0
nue_2 = nuepic.get_scat_type(nue_1)

#Calc thetae and Etheta^2
nuecc_2 = nuepic.calc_thetat(nuecc_1,method=1) #Use momentum method
nueccNp_2 = nuepic.calc_thetat(nueccNp_1,method=1) #Use momentum method
#nuecc_3 = nuepic.calc_thetave(nuecc_2,return_key='theta_ve')
nuecc_4 = nuepic.calc_Etheta(nuecc_2)
nueccNp_4 = nuepic.calc_Etheta(nueccNp_2)
#nuecc_4.loc[:,'E_theta_ve'] = nuecc_4.loc[:,'genie_Eng']*nuecc_4.loc[:,'theta_ve']**2
nue_3 = nuepic.calc_thetat(nue_2,method=1) #Use momentum method
#nue_4 = nuepic.calc_thetave(nue_3,return_key='theta_ve')
nue_5 = nuepic.calc_Etheta(nue_3)
#nue_5.loc[:,'E_theta_ve'] = nue_5.loc[:,'genie_Eng']*nue_5.loc[:,'theta_ve']**2

t4 = time()
print(f'Calc and organize new vars. time {t4-t3:.2f} s')

#Save dataframes
#nuecc_3.to_pickle(f'{data_dir}/{savename}')
nuecc_4.to_pickle(f'data/{savename}')
nueccNp_4.to_pickle(f'data/{savename_Np}')
#nue_4.to_pickle(f'{data_dir}/{savename_signal}')
nue_5.to_pickle(f'data/{savename_signal}')





