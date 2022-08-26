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
enu_folder = f'enu_{day}'

#Constants/parameters
data_dir = '/pnfs/sbnd/persistent/users/brindenc/analyze_sbnd/nue/v09_43_00/data'
#data_dir = 'data'
pot1 = 6.6e20
pot2 = 10e20
SEED = 42
Etheta1 = 0.003 #Gev rad^2
Etheta2 = 0.004 #Gev rad^2
Etheta3 = 0.001
cut1_str = r'$E_e \theta _e ^2 <$ 0.003 [GeV rad$^2$]'
cut2_str = r'$E_e \theta _e ^2 <$ 0.004 [GeV rad$^2$]'
cut3_str = r'$E_e \theta _e ^2 <$ 0.001 [GeV rad$^2$]'
bw_e = 0.1
bw_t = 0.02
bw_et = 0.0005
bw_nu = 0.1
alpha = 1
stacked = True


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
#nue_pot_per_event = pot_nue/nue_df.index.drop_duplicates().shape[0]
#nuecc_pot_per_event = pot_nuecc/nuecc_df.index.drop_duplicates().shape[0]
nue_pot_per_event = -1 #Temporary value to use raw pot info method
nuecc_pot_per_event = -1



#Get normalized dfs
events_nuecc = len(nuecc.index.drop_duplicates())
events_nue = len(nue.index.drop_duplicates())
back_pot1,_ = nuepic.get_pot_normalized_df(nuecc,pot1,pot_nuecc,events_nuecc,seed=SEED,pot_per_event=nuecc_pot_per_event)
nue_pot1,_ = nuepic.get_pot_normalized_df(nue,pot1,pot_nue,events_nue,seed=SEED,pot_per_event=nue_pot_per_event)
back_pot2,_ = nuepic.get_pot_normalized_df(nuecc,pot2,pot_nuecc,events_nuecc,seed=SEED,pot_per_event=nuecc_pot_per_event)
nue_pot2,_ = nuepic.get_pot_normalized_df(nue,pot2,pot_nue,events_nue,seed=SEED,pot_per_event=nue_pot_per_event)

#Get s/b
sb1 = nuepic.get_signal_background(nue_pot1,back_pot1)
sb2 = nuepic.get_signal_background(nue_pot2,back_pot2)
#sb0 = nuepic.get_signal_background(nue,nuecc)

#Make precut plots
# ax,fig = nueplotters.hist_scatback(nue_pot1,back_pot1,'genie_Eng',sb0,alpha=alpha,
#           title=r'$e^-$ Energy Distribution 6.6 $\times$10$^{20}$ POT'+'\nPrecut',xlabel='Energy [GeV]',
#           ylabel='',bw=bw_e)
# plotters.save_plot('E_pot1__precut',folder_name=energy_folder)
# ax,fig = nueplotters.hist_scatback(nue_pot1,back_pot1,'theta_t',sb0,alpha=alpha,
#           title=r'$\theta_e$ Distribution 6.6 $\times$10$^{20}$ POT'+'\n Precut',xlabel=r'$\theta _e$ [rad]',ylabel='',
#           bw=bw_t)
# plotters.save_plot('theta_pot1__precut',folder_name=thetae_folder)
# ax.set_xlim([-0.01,0.5])
# plotters.save_plot('theta_pot1__precut_zoom',folder_name=thetae_folder)
# ax,fig = nueplotters.hist_scatback(nue_pot1,back_pot1,'E_theta^2',sb0,alpha=alpha,
#           title=r'$E_e \theta_e^2$ Distribution 6.6 $\times$10$^{20}$ POT'+'\nPrecut',xlabel=r'$E_e \theta_e^2$ [GeV rad$^2$]',ylabel='',
#           bw=0.1)
# plotters.save_plot('Ethetae_pot1__precut',folder_name=cuts_folder)
# ax,fig = nueplotters.hist_scatback(nue_pot1,back_pot1,'E_theta^2',sb0,alpha=alpha,
#           title=r'$E_e \theta_e^2$ Distribution 6.6 $\times$10$^{20}$ POT'+'\nPrecut',xlabel=r'$E_e \theta_e^2$ [GeV rad$^2$]',ylabel='',
#           bw=bw_et)
# ax.set_xlim([0,0.05])
# plotters.save_plot('Ethetae_pot1__precut_zoom',folder_name=cuts_folder)

#Calc cuts
back_pot1_cut1 = nuepic.make_cuts(back_pot1,Etheta=Etheta1)
back_pot1_cut2 = nuepic.make_cuts(back_pot1,Etheta=Etheta2)
back_pot1_cut3 = nuepic.make_cuts(back_pot1,Etheta=Etheta3)
back_pot2_cut1 = nuepic.make_cuts(back_pot2,Etheta=Etheta1)
back_pot2_cut2 = nuepic.make_cuts(back_pot2,Etheta=Etheta2)
back_pot2_cut3 = nuepic.make_cuts(back_pot2,Etheta=Etheta3)

nue_pot1_cut1 = nuepic.make_cuts(nue_pot1,Etheta=Etheta1)
nue_pot1_cut2 = nuepic.make_cuts(nue_pot1,Etheta=Etheta2)
nue_pot1_cut3 = nuepic.make_cuts(nue_pot1,Etheta=Etheta3)
nue_pot2_cut1 = nuepic.make_cuts(nue_pot2,Etheta=Etheta1)
nue_pot2_cut2 = nuepic.make_cuts(nue_pot2,Etheta=Etheta2)
nue_pot2_cut3 = nuepic.make_cuts(nue_pot2,Etheta=Etheta3)

#Get new s/b
sb1_pot1_cut1 = nuepic.get_signal_background(nue_pot1_cut1,back_pot1_cut1)
sb1_pot1_cut2 = nuepic.get_signal_background(nue_pot1_cut2,back_pot1_cut2)
sb1_pot1_cut3 = nuepic.get_signal_background(nue_pot1_cut3,back_pot1_cut3)
sb1_pot2_cut1 = nuepic.get_signal_background(nue_pot2_cut1,back_pot2_cut1)
sb1_pot2_cut2 = nuepic.get_signal_background(nue_pot2_cut2,back_pot2_cut2)
sb1_pot2_cut3 = nuepic.get_signal_background(nue_pot2_cut3,back_pot2_cut3)

#Make arrays for data
Ethetas = [Etheta1,Etheta2,Etheta3]
pots = [pot1,pot2]
potstrs = [r'6.6 $\times$10$^{20}$ POT',r'10$^{21}$ POT']
sb_precuts = [sb1,sb2]
sbs = [[sb1_pot1_cut1,sb1_pot2_cut1],[sb1_pot1_cut2,sb1_pot2_cut2],[sb1_pot1_cut3,sb1_pot2_cut3]]
nues = [[nue_pot1_cut1,nue_pot2_cut1],[nue_pot1_cut2,nue_pot2_cut2],[nue_pot1_cut3,nue_pot2_cut3]]
backs = [[back_pot1_cut1,back_pot2_cut1],[back_pot1_cut2,back_pot2_cut2],[back_pot1_cut3,back_pot2_cut3]]
nue_precuts = [nue_pot1,nue_pot2]
back_precuts = [back_pot1,back_pot2]

#Make postcut plots
for i in range(3):#3
  for j in range(2):#2
    sb_precut = sb_precuts[j]
    sb = sbs[i][j]
    nue = nues[i][j]
    back = backs[i][j]
    Etheta=Ethetas[i]
    savestr = f'cut{i}_pot{j}'
    potstr = potstrs[j]
    nue_precut = nue_precuts[j]
    back_precut = back_precuts[j]

    #Precut plots
    if i == 0:
      ax,fig = nueplotters.hist_scatback(nue_precut,back_precut,'genie_Eng',sb_precut,alpha=alpha,
                title=r'$E_e$ Distribution '+potstr+'\n'+'Precut',
                xlabel='Energy [GeV]',ylabel='',bw=bw_e,stacked=stacked,edgecolor='black')
      plotters.save_plot(f'E_pot{j}__precut',folder_name=energy_folder)
      ax,fig = nueplotters.hist_scatback(nue_precut,back_precut,'theta_t',sb_precut,alpha=alpha,
                title=r'$\theta_e$ Distribution '+potstr+'\nPrecut',
                xlabel=r'$\theta _e$ [rad]',ylabel='',bw=bw_t,stacked=stacked,edgecolor='black')
      plotters.save_plot(f'thetae_pot{j}__precut',folder_name=thetae_folder)
      ax.set_xlim([-0.01,0.5])
      plotters.save_plot(f'thetae_pot{j}__precut_zoom',folder_name=thetae_folder)
      plt.close()
      ax,fig = nueplotters.hist_scatback(nue_precut,back_precut,'E_theta^2',sb_precut,alpha=alpha,
                title=r'$E_e \theta_e^2$ Distribution '+potstr+'\nPrecut',
                xlabel=r'$E_e \theta_e^2$ [GeV rad$^2$]',ylabel='',
                bw=0.1,stacked=stacked,edgecolor='black')
      plotters.save_plot(f'Ethetae_pot{j}__precut',folder_name=cuts_folder)
      plt.close()
      ax,fig = nueplotters.hist_scatback(nue_precut,back_precut,'E_theta^2',sb_precut,alpha=alpha,
                title=r'$E_e \theta_e^2$ Distribution '+potstr+'\nPrecut',
                xlabel=r'$E_e \theta_e^2$ [GeV rad$^2$]',ylabel='',
                bw=bw_et*2,stacked=stacked,edgecolor='black')
      ax.set_xlim([0,0.015])
      plotters.save_plot(f'Ethetae_pot{j}__precut_zoom',folder_name=cuts_folder)
      plt.close()


    #Postcut plots
    ax,fig = nueplotters.hist_scatback(nue,back,'genie_Eng',sb,alpha=alpha,
              title=r'$E_e$ Distribution '+potstr+'\n'+r'$E_e \theta_e^2 <$'+f'{Etheta}',
              xlabel='Energy [GeV]',ylabel='',bw=bw_e,stacked=stacked,edgecolor='black')
    plotters.save_plot(f'E_{savestr}',folder_name=energy_folder)
    plt.close()
    ax,fig = nueplotters.hist_scatback(nue,back,'theta_t',sb,alpha=alpha,
              title=r'$\theta_e$ Distribution '+potstr+'\n'+r'$E_e \theta_e^2 <$'+f'{Etheta}',
              xlabel=r'$\theta _e$ [rad]',ylabel='',bw=bw_t,stacked=stacked,edgecolor='black')
    ax.set_xlim([-0.01,0.5])
    plotters.save_plot(f'thetae_{savestr}',folder_name=thetae_folder)
    plt.close()
    ax,fig = nueplotters.hist_scatback(nue,back,'E_theta^2',sb,alpha=alpha,
              title=r'$E_e \theta_e^2$ Distribution '+potstr+'\n'+r'$E_e \theta_e^2 <$'+f'{Etheta}',
              xlabel=r'$E_e \theta_e^2$ [GeV rad$^2$]',ylabel='',
              bw=bw_et,stacked=stacked,edgecolor='black')
    ax.set_xlim([None,Etheta+bw_et])
    plotters.save_plot(f'Ethetae_{savestr}',folder_name=cuts_folder)
    plt.close()
    ax,fig = nueplotters.hist_scatback(nue,back,'genie_Eng',sb=sb,alpha=alpha,
                title=r'$E_\nu$ Distribution '+potstr+'\n'+r'$E\theta ^2 <$ '+f'{Etheta}',
                xlabel=r'$E_\nu$ [GeV]',ylabel='',
                bw=bw_nu,stacked=stacked,pdgs=[12,14],include_background=True,status_code=0,
                edgecolor='black')
    ax.set_xlim([0,4])
    plotters.save_plot(f'Enu_{savestr}',folder_name=enu_folder)

#print(sb0,sb1_pot1_cut1,sb1_pot1_cut2)

#Make new plots here



