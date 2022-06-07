import sys
sys.path.append('/sbnd/app/users/brindenc/mypython') #My utils path
from bc_utils.nueutils import pic as nuepic
from bc_utils.utils import pic
from time import time

#Constants/parameters
E_threshold = 0.02 #GeV energy threshold for visible hadron
data_dir = '/pnfs/sbnd/persistent/users/brindenc/analyze_sbnd/nue/v09_43_00/data'
savename = 'nuecc.pkl'
savename_signal = 'nue.pkl'

t1 = time()

#Load files
nue_df,_ = nuepic.get_genie_df(f'{data_dir}/nue.root','NuEScat','Event')
nuecc_df,_ = nuepic.get_genie_df(f'{data_dir}/nuecc.root','NuEScat','Event')
t2 = time()
pic.print_stars()

print(f'Load time {t2-t1:.2f} s')

#Make nuecc cuts
cut1 = nuepic.cut_pdg_event(nuecc_df,211,E_threshold=E_threshold) #Pions
cut2 = nuepic.cut_pdg_event(cut1,2212,E_threshold=E_threshold) #Protons

#Exotic particle cuts
pdgs_seen = [3112,321,4122,3222] #Make cuts on these?
cut3 = nuepic.find_hadron_activity(cut2,drop=True) #Cuts for nu e events seen in background
cut4 = cut3[cut3['ccnc_truth'] == 0] #Keep only cc events

print(f'Precut events {nuecc_df.index.drop_duplicates().shape[0]}')
print(f'Cut pion events {cut1.index.drop_duplicates().shape[0]}')
print(f'Cut proton events {cut2.index.drop_duplicates().shape[0]}')
print(f'Cut nu+e events {cut3.index.drop_duplicates().shape[0]}')
print(f'Cut non cc events {cut4.index.drop_duplicates().shape[0]}')

t3 = time()
print(f'Cut time {t3-t2:.2f} s')

#Drop at rest electrons (initial state)
nue_0 = nuepic.drop_initial_e(nue_df) #Get rid of initial electron in dataframe

#Add electron count info, just in case 
#nuecc_1 = nuepic.get_electron_count(cut4)
nuecc_1 = nuepic.get_electron_count(cut3) #Include cc and nc events
nue_1 = nuepic.get_electron_count(nue_0)

#Get background and scat types
nuecc_1.loc[:,'background_type'] = 0
nue_2 = nuepic.get_scat_type(nue_1)

#Calc thetae and Etheta^2
nuecc_2 = nuepic.calc_thetat(nuecc_1,method=1) #Use momentum method
nuecc_3 = nuepic.calc_Etheta(nuecc_2)
nue_3 = nuepic.calc_thetat(nue_2,method=1) #Use momentum method
nue_4 = nuepic.calc_Etheta(nue_3)

t4 = time()
print(f'Calc and organize new vars. time {t4-t3:.2f} s')

#Save dataframes
#nuecc_3.to_pickle(f'{data_dir}/{savename}')
nuecc_3.to_pickle(f'data/{savename}')
#nue_4.to_pickle(f'{data_dir}/{savename_signal}')
nue_4.to_pickle(f'data/{savename_signal}')





