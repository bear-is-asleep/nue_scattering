import sys
sys.path.append('/sbnd/app/users/brindenc/mypython') #My utils path
from bc_utils.nueutils import pic as nuepic
from bc_utils.nueutils import plotters as nueplotters
from bc_utils.utils import pic,plotters
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from datetime import date

#Important stuff!
plotters.plot_stuff()
day = date.today().strftime("%Y_%m_%d")
energy_folder = f'energy_{day}'
thetae_folder = f'thetae_{day}'
cuts_folder = f'cuts_{day}'

#Constants/parameters
data_dir = '/pnfs/sbnd/persistent/users/brindenc/analyze_sbnd/nue/v09_43_00/data'
#data_dir = 'data'
pot1 = 6.6e20
pot2 = 10e20
SEED = 69420
Etheta1 = 0.03 #Gev rad^2
Etheta2 = 0.04 #Gev rad^2


#Load data
t1 = time()
nuecc = pd.read_pickle(f'data/nuecc.pkl')
nue = pd.read_pickle(f'data/nue.pkl')
#nuecc = pd.read_pickle(f'{data_dir}/nuecc.pkl')
#nue = pd.read_pickle(f'{data_dir}/nue.pkl')
#pot_nuecc = nuepic.get_pot(f'{data_dir}/nuecc.root','NuEScat')
#pot_nue = nuepic.get_pot(f'{data_dir}/nue.root','NuEScat')
pot_nuecc = nuepic.get_pot(f'{data_dir}/nuecc.root','NuEScat')
pot_nue = nuepic.get_pot(f'{data_dir}/nue.root','NuEScat')
nue_df,_ = nuepic.get_genie_df(f'{data_dir}/nue.root','NuEScat','Event')
nuecc_df,_ = nuepic.get_genie_df(f'{data_dir}/nuecc.root','NuEScat','Event')

#POT per event
nue_pot_per_event = pot_nue/nue_df.index.drop_duplicates().shape[0]
nuecc_pot_per_event = pot_nuecc/nuecc_df.index.drop_duplicates().shape[0]



#Get normalized dfs
events_nuecc = len(nuecc.index.drop_duplicates())
events_nue = len(nue.index.drop_duplicates())
back_pot1,_ = nuepic.get_pot_normalized_df(nuecc,pot1,pot_nuecc,events_nuecc,seed=SEED,pot_per_event=nuecc_pot_per_event)
nue_pot1,_ = nuepic.get_pot_normalized_df(nue,pot1,pot_nue,events_nue,seed=SEED,pot_per_event=nue_pot_per_event)
back_pot2,_ = nuepic.get_pot_normalized_df(nuecc,pot2,pot_nuecc,events_nuecc,seed=SEED,pot_per_event=nuecc_pot_per_event)
nue_pot2,_ = nuepic.get_pot_normalized_df(nue,pot2,pot_nue,events_nue,seed=SEED,pot_per_event=nue_pot_per_event)

#Get s/b
sb0 = nuepic.get_signal_background(nue_pot1,back_pot1)
#sb0 = nuepic.get_signal_background(nue,nuecc)

#Make precut plots
ax,fig = nueplotters.hist_scatback(nue_pot1,back_pot1,'genie_Eng',sb0,alpha=0.6,
          title=r'$e^-$ Energy Distribution 6.6 $\times$10$^{20}$ POT'+'\nPrecut',xlabel='Energy [GeV]',ylabel='')
plotters.save_plot('E_pot1__precut',folder_name=energy_folder)
ax,fig = nueplotters.hist_scatback(nue_pot1,back_pot1,'theta_t',sb0,alpha=0.6,
          title=r'$\theta_e$ Distribution 6.6 $\times$10$^{20}$ POT'+'\n Precut',xlabel=r'$\theta _e$ [rad]',ylabel='',
          nbins=100)
ax.set_xlim([-0.1,0.5])
plotters.save_plot('E_pot1__precut',folder_name=thetae_folder)
ax,fig = nueplotters.hist_scatback(nue_pot1,back_pot1,'E_theta^2',sb0,alpha=0.6,
          title=r'$E_e \theta_e^2$ Distribution 6.6 $\times$10$^{20}$ POT'+'\nPrecut',xlabel=r'$E_e \theta_e^2$ [GeV rad$^2$]',ylabel='')
plotters.save_plot('Ethetae_pot1__precut',folder_name=cuts_folder)

#Calc cuts
back_pot1_cut1 = nuepic.make_cuts(back_pot1,Etheta=Etheta1)
back_pot1_cut2 = nuepic.make_cuts(back_pot1,Etheta=Etheta2)
back_pot2_cut1 = nuepic.make_cuts(back_pot2,Etheta=Etheta1)
back_pot2_cut2 = nuepic.make_cuts(back_pot2,Etheta=Etheta2)

nue_pot1_cut1 = nuepic.make_cuts(nue_pot1,Etheta=Etheta1)
nue_pot1_cut2 = nuepic.make_cuts(nue_pot1,Etheta=Etheta2)
nue_pot2_cut1 = nuepic.make_cuts(nue_pot2,Etheta=Etheta1)
nue_pot2_cut2 = nuepic.make_cuts(nue_pot2,Etheta=Etheta2)

#Get new s/b
sb1_pot1_cut1 = nuepic.get_signal_background(nue_pot1_cut1,back_pot1_cut1)
sb1_pot1_cut2 = nuepic.get_signal_background(nue_pot1_cut2,back_pot1_cut2)
sb1_pot2_cut1 = nuepic.get_signal_background(nue_pot2_cut1,back_pot2_cut1)
sb1_pot2_cut2 = nuepic.get_signal_background(nue_pot2_cut2,back_pot2_cut2)

print(sb0,sb1_pot1_cut1,sb1_pot1_cut2)

#Make new plots here


