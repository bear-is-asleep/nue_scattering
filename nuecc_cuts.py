#!/sbnd/data/users/brindenc/.local/bin/python3.9
import sys
sys.path.append('/sbnd/app/users/brindenc/mypython') #My utils path
from bc_utils.CAFana import pic as CAFpic
from bc_utils.CAFana import plotters as CAFplotters
from bc_utils.utils import pic,plotters
from time import time
import numpy as np
import pandas as pd
import uproot
import nuecc #Contains parameters
import matplotlib.pyplot as plt
plt.style.use(['science','no-latex'])


#Constants/parameters
pot1 = nuecc.pot1
pot2 = 20e20
SEED = nuecc.SEED
n = nuecc.n #number of random universes
ntrk = 50 #Size of shower confusion matrix
nshw = 50 #Size of track confusion matrix
E_threshold = 0.021 #GeV energy threshold for visible hadron (ArgoNeut)
E_threshold_exotic = E_threshold #Set same for now
DATA_DIR = nuecc.DATA_DIR
#DATA_DIR = '../test_fcl/'
savename = 'nuecc.pkl'
savename_Np = 'nueccNp.pkl'
savename_signal = 'nue.pkl'
exotic_hadrons = [] #Make cuts on these? 3112,321,4122,3222
print_results = False
Etheta1 = 0.003 #Gev rad^2
Etheta2 = 0.004 #Gev rad^2
Etheta3 = 0.001 #Gev rad^2
razzlecutoff = 0.6 #best score required to be electron
suffix = '_full' #Add to end of all files to distinguish between test and full sample
truthstudy = True
recostudy = True

t1 = time()

#Load files

#Write down CAF keys we want to use
nreco_keys = nuecc.nreco_keys
shw_keys = nuecc.shw_keys
mcnu_keys = nuecc.mcnu_keys
mcprim_keys = nuecc.mcprim_keys

#nu + e
nue_tree1 = uproot.open(f'{DATA_DIR}CAFnue1{suffix}.root:recTree;1')

nue_mc = CAFpic.get_df(nue_tree1,mcnu_keys)
nue_mcprim = CAFpic.get_df(nue_tree1,mcprim_keys)
nue_nreco = CAFpic.get_df(nue_tree1,nreco_keys) #Make seperate tree to count reco objects
nue_shw = CAFpic.get_df(nue_tree1,shw_keys)

#nuecc
nuecc_tree1 = uproot.open(f'{DATA_DIR}CAFnuecc1{suffix}.root:recTree;1')

nuecc_mc = CAFpic.get_df(nuecc_tree1,mcnu_keys)
nuecc_mcprim = CAFpic.get_df(nuecc_tree1,mcprim_keys)
nuecc_nreco = CAFpic.get_df(nuecc_tree1,nreco_keys) #Make seperate tree to count reco objects
nuecc_shw = CAFpic.get_df(nuecc_tree1,shw_keys)

#POT + event info
pots = nuecc_tree1.arrays('rec.hdr.pot',library='pd')
pot_nuecc = pots.values.sum() #Get POT
indeces_nuecc = nuecc_mc.index.drop_duplicates()
events_nuecc = len(indeces_nuecc)

pots = nue_tree1.arrays('rec.hdr.pot',library='pd')
pot_nue = pots.values.sum() #Get POT
indeces_nue = nue_mc.index.drop_duplicates()
events_nue = len(indeces_nue)

#Initialize universe study - truth level
columns = ['Precut','Fiducial\nVolume','nc','nu+e','pion','proton']
columns.extend(['pdg '+str(x) for x in exotic_hadrons])
columns.extend([r'$E\theta^2 < $ '+f'{Etheta1} '+'\n'+r'[Gev rad$^2$]',
  r'$E\theta^2 < $ '+f'{Etheta2} '+'\n'+r'[Gev rad$^2$]',r'$E\theta^2 < $ '+f'{Etheta3} '+'\n'+r'[Gev rad$^2$]'])
columns.extend(['POT'])
events_cut = np.full((2,n+2,len(columns)),0.) #Display number of events cut for each step

#Initialize universe study - reco level
columns_reco = ['Precut','Fiducial\nVolume',r'$n_{shw} = 1$',r'$n_{trk} = 0$',r'$E_{reco}>200$'+'\n[MeV]',f'e- razzle > {razzlecutoff}']
columns_reco.extend([r'$E\theta^2 < $ '+f'{Etheta1} '+'\n'+r'[Gev rad$^2$]',
  r'$E\theta^2 < $ '+f'{Etheta2} '+'\n'+r'[Gev rad$^2$]',r'$E\theta^2 < $ '+f'{Etheta3} '+'\n'+r'[Gev rad$^2$]'])
columns_reco.extend(['POT'])
events_cut_reco = np.zeros((2,n+2,len(columns_reco))) #Display number of events cut for each step

#Initialize universe study - confusion matrices
shwconfusion = np.zeros((2,n+2,nshw,nshw)) #Confusion matrices for number of showers
trkconfusion = np.zeros((2,n+2,ntrk,ntrk)) #Confusion matrices for number of tracks

shwconfusion_cut1 = np.zeros((2,n+2,nshw,nshw)) #Confusion matrices for number of showers
trkconfusion_cut1 = np.zeros((2,n+2,ntrk,ntrk)) #Confusion matrices for number of tracks

#Constants for plotting
labels = [r'$\nu+e$',r'$\nu_e$']
erecobins = np.arange(0,4,0.1)
ethetabins = np.arange(0,0.005,0.001)
thetabins = np.arange(0,0.4,0.05)


seeds = np.arange(n) #Get random seeds

t2 = time()

print(f'Load time {t2-t1:.2f} s')
pic.print_stars()

for seed in seeds: #Iterate over n random universes, with different seed each time
  #Time each iteration
  t3 = time()

  #Get pot normalized event list
  nuecc_drop_indeces = CAFpic.get_pot_normalized_indeces(indeces_nuecc,pot1,pot_nuecc,seed=SEED*seed)
  nue_drop_indeces = CAFpic.get_pot_normalized_indeces(indeces_nue,pot1,pot_nue,seed=SEED*seed)
  #print('got indeces')

  #nuecc
  nuecc_nreco_pot1 = CAFpic.get_df_dropindeces(nuecc_nreco,nuecc_drop_indeces).sort_index()
  nuecc_mc_pot1 = CAFpic.get_df_dropindeces(nuecc_mc,nuecc_drop_indeces).sort_index()
  nuecc_mcprim_pot1 = CAFpic.get_df_dropindeces(nuecc_mcprim,nuecc_drop_indeces).sort_index()
  nuecc_shw_pot1 = CAFpic.get_df_dropindeces(nuecc_shw,nuecc_drop_indeces).sort_index()

  #nue
  nue_nreco_pot1 = CAFpic.get_df_dropindeces(nue_nreco,nue_drop_indeces).sort_index()
  nue_mc_pot1 = CAFpic.get_df_dropindeces(nue_mc,nue_drop_indeces).sort_index()
  nue_mcprim_pot1 = CAFpic.get_df_dropindeces(nue_mcprim,nue_drop_indeces).sort_index()
  nue_shw_pot1 = CAFpic.get_df_dropindeces(nue_shw,nue_drop_indeces).sort_index()
  
  t6 = time()
  print(f'Time to drop indeces seed {seed*SEED}: {t6-t3:.2f} (s)')

  #TRUTH LEVEL
  if truthstudy:
    truthcut_iter = 0 #add one after each cut

    #Precut
    precut_signal = nue_mc_pot1.index.drop_duplicates().shape[0] #Precut signal events
    events_cut[0,seed,truthcut_iter] = nuecc_mc_pot1.index.drop_duplicates().shape[0]
    events_cut[1,seed,truthcut_iter] = precut_signal
    truthcut_iter+=1

    #FV containment - truth interaction vertex
    #print(f'FV{time()-t6}')
    nuecc_fvindeces = CAFpic.FV_cut(nuecc_mc_pot1)
    events_cut[0,seed,truthcut_iter] = len(nuecc_fvindeces)
    nue_fvindeces = CAFpic.FV_cut(nue_mc_pot1)
    events_cut[1,seed,truthcut_iter] = len(nue_fvindeces)
    truthcut_iter+=1

    #NC Cut
    #print(f'NC{time()-t6}')
    #Precut plots
    if seed < 2:
      ax,_ = CAFplotters.back_sig_hist([nue_mc_pot1,nuecc_mc_pot1],labels,f'{CAFpic.mcnuprefix}iscc',precut_signal,xlabel='GENIE CC',bins=2,alpha=0.8)
      plotters.save_plot(f'cc_precut{seed}')
      ccmin,ccmax = ax.get_ylim()

    #Do cuts
    nuecc_mc_pot1 = CAFpic.get_df_keepindeces(nuecc_mc_pot1,nuecc_fvindeces)
    cut1 = nuecc_mc_pot1[nuecc_mc_pot1.loc[:,f'{CAFpic.mcnuprefix}iscc'] == 1] #Keep only cc events
    ncevent = nuecc_mc_pot1[nuecc_mc_pot1.loc[:,f'{CAFpic.mcnuprefix}iscc'] == 0].index[0] #Get single nc event
    print(f'nuecc nc ind: {ncevent}')
    cut1_indeces = cut1.index.drop_duplicates()
    events_cut[0,seed,truthcut_iter] = cut1_indeces.shape[0]
    nuecc_mcprim_pot1 = CAFpic.get_df_keepindeces(nuecc_mcprim_pot1,cut1_indeces)

    nue_mcprim_pot1 = CAFpic.get_df_keepindeces(nue_mcprim_pot1,nue_fvindeces)
    events_cut[1,seed,truthcut_iter] = len(nue_fvindeces)
    truthcut_iter+=1

    if seed < 2:
      ax,_ = CAFplotters.back_sig_hist([nue_mc_pot1,cut1],labels,f'{CAFpic.mcnuprefix}iscc',precut_signal,xlabel='GENIE CC',bins=2,alpha=0.8)
      #ax.text(0.2,40,r"Don't drop $\nu+e$ CC events")
      ax.set_ylim([ccmin,ccmax])
      plotters.save_plot(f'cc_postcut{seed}')

    #nu e scattering cut
    #print(f'nue{time()-t6}')
    cut2,_ = CAFpic.true_nue(nuecc_mcprim_pot1) #Drop nue scattering events
    events_cut[0,seed,truthcut_iter] = cut2.index.drop_duplicates().shape[0]
    events_cut[1,seed,truthcut_iter] = len(nue_fvindeces)
    truthcut_iter+=1

    #Charged pion
    #print(f'pi{time()-t6}')
    cut3,_ = CAFpic.cut_pdg_event(cut2,211,E_threshold=E_threshold) #Pions
    events_cut[0,seed,truthcut_iter] = cut3.index.drop_duplicates().shape[0]
    events_cut[1,seed,truthcut_iter] = len(nue_fvindeces)
    truthcut_iter+=1

    #Proton
    #print(f'proton{time()-t6}')
    cut4,_ = CAFpic.cut_pdg_event(cut3,2212,E_threshold=E_threshold) #Protons
    events_cut[0,seed,truthcut_iter] = cut4.index.drop_duplicates().shape[0]
    events_cut[1,seed,truthcut_iter] = len(nue_fvindeces)
    truthcut_iter+=1

    #Exotic particle cuts
    cut5 = cut4.copy() #Set temporary df
    for j,pdg in enumerate(exotic_hadrons): #this will iteravily cut hadrons
      temp_cut,_ = CAFpic.cut_pdg_event(cut5,pdg,E_threshold=E_threshold_exotic)
      cut5 = temp_cut.copy()
      events_cut[0,seed,truthcut_iter] = cut5.index.drop_duplicates().shape[0]
      events_cut[1,seed,truthcut_iter] = len(nue_fvindeces)
      truthcut_iter+=1

    #Print events cut
    if print_results:
      print(f'Precut events {nuecc_mc_pot1[0].index.drop_duplicates().shape[0]}')
      print(f'Cut non cc events {cut1.index.drop_duplicates().shape[0]}')
      print(f'Cut nu+e events {cut2.index.drop_duplicates().shape[0]}')
      print(f'Cut pion events {cut3.index.drop_duplicates().shape[0]}')
      print(f'Cut proton events {cut4.index.drop_duplicates().shape[0]}')
      for i,pdg in enumerate(exotic_hadrons):
        print(f'Cut pdg {pdg} events {events_cut[0,seed,i+5]}')
    
    #Calc thetae and Etheta^2
    #print(f'Calc{time()-t6}')
    nuecc_1 = CAFpic.calc_thetat(cut5) #Use momentum method
    nuecc_2 = CAFpic.calc_Etheta(nuecc_1)
    nuecc_e = nuecc_2[abs(nuecc_2.loc[:,f'{CAFpic.primprefix}pdg']) == 11]


    nue_1 = CAFpic.calc_thetat(nue_mcprim_pot1) #Use momentum method
    nue_2 = CAFpic.calc_Etheta(nue_1)
    nue_e = nue_2[abs(nue_2.loc[:,f'{CAFpic.primprefix}pdg']) == 11]
    
    #Etheta cuts
    #Calc cuts
    if seed < 2:
      ax,etheta_zoomout_bins = CAFplotters.back_sig_hist([nue_e,nuecc_e],labels,f'{CAFpic.primprefix}Etheta2',precut_signal,xlabel=r'$E_e\theta_e^2$ [GeV rad$^2$]',alpha=0.8)
      ethetamin, ethetamax = ax.get_xlim()
      ethetamin_y, ethetamax_y = ax.get_ylim()
      plotters.save_plot(f'etheta1_precut{seed}')
      ax,_ = CAFplotters.back_sig_hist([nue_e,nuecc_e],labels,f'{CAFpic.primprefix}Etheta2',precut_signal,xlabel=r'$E_e\theta_e^2$ [GeV rad$^2$]',
        annotate=False,alpha=0.8,bins=ethetabins)
      ax.set_xlim([0,0.005])
      plotters.save_plot(f'etheta1_precut_zoom{seed}')
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_e,nuecc_e],labels,f'{CAFpic.primprefix}genE',precut_signal,xlabel=r'$E_e$ [GeV]',alpha=0.8,
        bins=erecobins)
      plotters.save_plot(f'e1_precut{seed}')
      emin, emax = ax.get_xlim()
      emin_y,emax_y = ax.get_ylim()
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_e,nuecc_e],labels,f'{CAFpic.primprefix}thetal',precut_signal,xlabel=r'$\theta_e$ [rad]',alpha=0.8)
      thetamin, thetamax = ax.get_xlim()
      thetamin_y,thetamax_y = ax.get_ylim()
      plotters.save_plot(f'theta1_precut{seed}')
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_e,nuecc_e],labels,f'{CAFpic.primprefix}thetal',precut_signal,xlabel=r'$\theta_e$ [rad]',alpha=0.8,
        bins=thetabins)
      plotters.save_plot(f'theta1_precut_zoom{seed}')
      plt.close()
    print('nuecc')
    nuecc_cut1,_ = CAFpic.make_cuts(nuecc_2,Etheta=Etheta1)
    events_cut[0,seed,truthcut_iter] = CAFpic.number_events(nuecc_cut1)
    print('nue')
    nue_cut1,_ = CAFpic.make_cuts(nue_2,Etheta=Etheta1)
    events_cut[1,seed,truthcut_iter] = CAFpic.number_events(nue_cut1)
    truthcut_iter+=1

    #Get just electrons
    nue_cute = nue_cut1[abs(nue_cut1.loc[:,f'{CAFpic.primprefix}pdg']) == 11]
    nuecc_cute = nuecc_cut1[abs(nuecc_cut1.loc[:,f'{CAFpic.primprefix}pdg']) == 11]

    if seed < 2:
      ax,_ = CAFplotters.back_sig_hist([nue_cute,nuecc_cute],labels,f'{CAFpic.primprefix}Etheta2',precut_signal,xlabel=r'$E_e\theta_e^2$ [GeV rad$^2$]',alpha=0.8,
        bins=ethetabins)
      #ax.set_xlim([ethetamin,ethetamax])
      plotters.save_plot(f'etheta1_cut{seed}')
      ax.set_ylim([ethetamin_y,ethetamax_y])
      plotters.save_plot(f'etheta1_cut_zoom{seed}')
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_cute,nuecc_cute],labels,f'{CAFpic.primprefix}genE',precut_signal,xlabel=r'$E_e$ [GeV]',alpha=0.8,
        bins=erecobins)
      ax.set_xlim([emin,emax])
      plotters.save_plot(f'e1_cut{seed}')
      ax.set_ylim([emin_y,emax_y])
      plotters.save_plot(f'e1_cut_zoomy{seed}')
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_cute,nuecc_cute],labels,f'{CAFpic.primprefix}thetal',precut_signal,xlabel=r'$\theta_e$ [rad]',alpha=0.8,
        bins=thetabins)
      ax.set_xlim([thetamin,thetamax])
      plotters.save_plot(f'theta1_cut{seed}')
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_cute,nuecc_cute],labels,f'{CAFpic.primprefix}thetal',precut_signal,xlabel=r'$\theta_e$ [rad]',alpha=0.8,
        bins=np.arange(0,0.4,0.05))
      plotters.save_plot(f'theta1_cut_zoom{seed}')
      ax.set_ylim([thetamin_y,thetamax_y])
      plotters.save_plot(f'theta1_cut_zoomxy{seed}')
      plt.close()

    print('nuecc')
    nuecc_cut2,_ = CAFpic.make_cuts(nuecc_2,Etheta=Etheta2)
    events_cut[0,seed,truthcut_iter] = CAFpic.number_events(nuecc_cut2)
    print('nue')
    
    nue_cut2,_ = CAFpic.make_cuts(nue_2,Etheta=Etheta2)
    events_cut[1,seed,truthcut_iter] = CAFpic.number_events(nue_cut2)
    truthcut_iter+=1

    #Get just electrons
    nue_cute = nue_cut2[abs(nue_cut2.loc[:,f'{CAFpic.primprefix}pdg']) == 11]
    nuecc_cute = nuecc_cut2[abs(nuecc_cut2.loc[:,f'{CAFpic.primprefix}pdg']) == 11]

    if seed < 2:
      ax,_ = CAFplotters.back_sig_hist([nue_cute,nuecc_cute],labels,f'{CAFpic.primprefix}Etheta2',precut_signal,xlabel=r'$E_e\theta_e^2$ [GeV rad$^2$]',alpha=0.8,
        bins=ethetabins)
      #ax.set_xlim([ethetamin,ethetamax])
      plotters.save_plot(f'etheta2_cut{seed}')
      ax.set_ylim([ethetamin_y,ethetamax_y])
      plotters.save_plot(f'etheta2_cut_zoom{seed}')
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_cute,nuecc_cute],labels,f'{CAFpic.primprefix}genE',precut_signal,xlabel=r'$E_e$ [GeV]',alpha=0.8,
        bins=erecobins)
      ax.set_xlim([emin,emax])
      plotters.save_plot(f'e2_cut{seed}')
      ax.set_ylim([emin_y,emax_y])
      plotters.save_plot(f'e2_cut_zoomy{seed}')
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_cute,nuecc_cute],labels,f'{CAFpic.primprefix}thetal',precut_signal,xlabel=r'$\theta_e$ [rad]',alpha=0.8,
        bins=erecobins)
      ax.set_xlim([thetamin,thetamax])
      plotters.save_plot(f'theta2_cut{seed}')
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_cute,nuecc_cute],labels,f'{CAFpic.primprefix}thetal',precut_signal,xlabel=r'$\theta_e$ [rad]',alpha=0.8,
        bins=np.arange(0,0.4,0.05))
      plotters.save_plot(f'theta2_cut_zoom{seed}')
      ax.set_ylim([thetamin_y,thetamax_y])
      plotters.save_plot(f'theta2_cut_zoomxy{seed}')
      plt.close()

    print('nuecc')

    nuecc_cut3,_ = CAFpic.make_cuts(nuecc_2,Etheta=Etheta3)
    events_cut[0,seed,truthcut_iter] = CAFpic.number_events(nuecc_cut3)
    print('nue')
    
    nue_cut3,_ = CAFpic.make_cuts(nue_2,Etheta=Etheta3)
    events_cut[1,seed,truthcut_iter] = CAFpic.number_events(nue_cut3)
    truthcut_iter+=1

    #Get just electrons
    nue_cute = nue_cut3[abs(nue_cut3.loc[:,f'{CAFpic.primprefix}pdg']) == 11]
    nuecc_cute = nuecc_cut3[abs(nuecc_cut3.loc[:,f'{CAFpic.primprefix}pdg']) == 11]

    if seed < 2:
      ax,_ = CAFplotters.back_sig_hist([nue_cute,nuecc_cute],labels,f'{CAFpic.primprefix}Etheta2',precut_signal,xlabel=r'$E_e\theta_e^2$ [GeV rad$^2$]',alpha=0.8,
        bins=ethetabins)
      #ax.set_xlim([ethetamin,ethetamax])
      plotters.save_plot(f'etheta3_cut{seed}')
      ax.set_ylim([ethetamin_y,ethetamax_y])
      plotters.save_plot(f'etheta3_cut_zoom{seed}')
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_cute,nuecc_cute],labels,f'{CAFpic.primprefix}genE',precut_signal,xlabel=r'$E_e$ [GeV]',alpha=0.8,
        bins=erecobins)
      ax.set_xlim([emin,emax])
      plotters.save_plot(f'e3_cut{seed}')
      ax.set_ylim([emin_y,emax_y])
      plotters.save_plot(f'e3_cut_zoomy{seed}')
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_cute,nuecc_cute],labels,f'{CAFpic.primprefix}thetal',precut_signal,xlabel=r'$\theta_e$ [rad]',alpha=0.8)
      ax.set_xlim([thetamin,thetamax])
      plotters.save_plot(f'theta3_cut{seed}')
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_cute,nuecc_cute],labels,f'{CAFpic.primprefix}thetal',precut_signal,xlabel=r'$\theta_e$ [rad]',alpha=0.8,
        bins=np.arange(0,0.4,0.05))
      plotters.save_plot(f'theta3_cut_zoom{seed}')
      ax.set_ylim([thetamin_y,thetamax_y])
      plotters.save_plot(f'theta3_cut_zoomxy{seed}')
      plt.close()

    t4 = time()
    print(f'Truth cuts seed {seed*SEED}: {t4-t3:.1f} (s)')

    #POT
    events_cut[:,seed,truthcut_iter] = pot1

  #RECO LEVEL
  if recostudy:
    recocut_iter = 0 #add one to this each time a cut is made
    t4 = time()

    #Revamp modified dataframes
    #nuecc
    nuecc_mcprim_pot1 = CAFpic.get_df_dropindeces(nuecc_mcprim,nuecc_drop_indeces)
    
    #nue
    nue_mcprim_pot1 = CAFpic.get_df_dropindeces(nue_mcprim,nue_drop_indeces)

    #Precut
    events_cut_reco[0,seed,recocut_iter] = nuecc_nreco_pot1.index.drop_duplicates().shape[0]
    events_cut_reco[1,seed,recocut_iter] = nue_nreco_pot1.index.drop_duplicates().shape[0]
    recocut_iter+=1

    #Get confusion matrices
    if seed == 999:
      shwconfusion[0,seed,:,:] = CAFpic.get_shw_confusion_matrix(nuecc_nreco_pot1,nuecc_mcprim_pot1,n=nshw)
      trkconfusion[0,seed,:,:] = CAFpic.get_trk_confusion_matrix(nuecc_nreco_pot1,nuecc_mcprim_pot1,n=ntrk)
      
      shwconfusion[1,seed,:,:] = CAFpic.get_shw_confusion_matrix(nue_nreco_pot1,nue_mcprim_pot1,n=nshw)
      trkconfusion[1,seed,:,:] = CAFpic.get_trk_confusion_matrix(nue_nreco_pot1,nue_mcprim_pot1,n=ntrk)
    #stubconfusion?

    #FV containment - truth interaction vertex
    nuecc_fvindeces = CAFpic.FV_cut(nuecc_mc_pot1)
    events_cut_reco[0,seed,recocut_iter] = len(nuecc_fvindeces) #Get nuecc_fvindeces from truth level cuts
    nue_fvindeces = CAFpic.FV_cut(nue_mc_pot1)
    events_cut_reco[1,seed,recocut_iter] = len(nue_fvindeces) #Get nue_fvindeces from truth level cuts
    recocut_iter+=1

    #Modify nreco values to keep only FV indeces
    nuecc_nreco_pot1 = CAFpic.get_df_keepindeces(nuecc_nreco_pot1,nuecc_fvindeces)
    nue_nreco_pot1 = CAFpic.get_df_keepindeces(nue_nreco_pot1,nue_fvindeces)


    #Number showers
    if seed < 2:
      ax,_ = CAFplotters.back_sig_hist([nue_nreco_pot1,nuecc_nreco_pot1],labels,f'{CAFpic.recoprefix}nshw',precut_signal,xlabel=r'$n_{shw}$',alpha=0.8,
        bins=[0,1,2,3,4,5])
      nshw_miny,nshw_maxy = ax.get_ylim()
      ax.set_xlim([0,5])
      plotters.save_plot(f'nshw_precut{seed}')
      plt.close()

    cut1,_ = CAFpic.cut_nshws(nuecc_nreco_pot1.loc[nuecc_fvindeces],1) #Keep only events with one reconstructed shower
    events_cut_reco[0,seed,recocut_iter] = cut1.index.drop_duplicates().shape[0]
    
    cut1_nue,_ = CAFpic.cut_nshws(nue_nreco_pot1.loc[nue_fvindeces],1) #Keep only events with one reconstructed shower
    events_cut_reco[1,seed,recocut_iter] = cut1_nue.index.drop_duplicates().shape[0]
    recocut_iter+=1

    if seed < 2:
      ax,_ = CAFplotters.back_sig_hist([cut1_nue,cut1],labels,f'{CAFpic.recoprefix}nshw',precut_signal,xlabel=r'$n_{shw}$',alpha=0.8,
        bins=[0,1,2,3,4,5])
      ax.set_xlim([0,5])
      plotters.save_plot(f'nshw_cut{seed}')
      ax.set_ylim([nshw_miny,nshw_maxy])
      plotters.save_plot(f'nshw_cut_zoomy{seed}')
      plt.close()

    #Number tracks
    if seed < 2:
      ax,_ = CAFplotters.back_sig_hist([cut1_nue,cut1],labels,f'{CAFpic.recoprefix}ntrk',precut_signal,xlabel=r'$n_{trk}$',alpha=0.8,
        bins=[0,1,2,3,4,5])
      ntrk_miny,ntrk_maxy = ax.get_ylim()
      ax.set_xlim([0,5])
      plotters.save_plot(f'ntrk_precut{seed}')
      plt.close()

    cut2,_ = CAFpic.cut_ntrks(cut1,0) #Keep only events with one reconstructed shower
    events_cut_reco[0,seed,recocut_iter] = cut2.index.drop_duplicates().shape[0]
    cut2_indeces = cut2.index.drop_duplicates()
    nuecc_shw_pot1 = CAFpic.get_df_keepindeces(nuecc_shw_pot1,cut2_indeces)

    cut2_nue,_ = CAFpic.cut_ntrks(cut1_nue,0) #Keep only events with one reconstructed shower
    events_cut_reco[1,seed,recocut_iter] = cut2_nue.index.drop_duplicates().shape[0]
    cut2_indeces = cut2_nue.index.drop_duplicates()
    nue_shw_pot1 = CAFpic.get_df_keepindeces(nue_shw_pot1,cut2_indeces)
    recocut_iter+=1

    if seed < 2:
      ax,_ = CAFplotters.back_sig_hist([cut2_nue,cut2],labels,f'{CAFpic.recoprefix}ntrk',precut_signal,xlabel=r'$n_{trk}$',alpha=0.8,
        bins=[0,1,2,3,4,5])
      ax.set_xlim([0,5])
      plotters.save_plot(f'ntrk_cut{seed}')
      ax.set_ylim([ntrk_miny,ntrk_maxy])
      plotters.save_plot(f'ntrk_cut_zoomy{seed}')
      plt.close()

    #Shower energy
    if seed < 2:
      CAFplotters.back_sig_hist([nue_shw_pot1,nuecc_shw_pot1],labels,f'{CAFpic.shwprefix}bestplane_energy',precut_signal,
      xlabel=r'$E_{reco}$ [GeV]',alpha=0.8,bins=erecobins)
      plotters.save_plot(f'ereco_precut{seed}')
      plt.close()
    nuecc_shwE = CAFpic.cut_recoE(nuecc_shw_pot1)
    events_cut_reco[0,seed,recocut_iter] = len(nuecc_shwE)

    nue_shwE = CAFpic.cut_recoE(nue_shw_pot1)
    events_cut_reco[1,seed,recocut_iter] = len(nue_shwE)
    recocut_iter+=1

    if seed < 2:
      CAFplotters.back_sig_hist([nue_shw_pot1.loc[nue_shwE],nuecc_shw_pot1.loc[nuecc_shwE]],labels,f'{CAFpic.shwprefix}bestplane_energy',precut_signal,
      xlabel=r'$E_{reco}$ [GeV]',alpha=0.8,bins=erecobins)
      plotters.save_plot(f'ereco_cut{seed}')
      plt.close()

    #Electron razzle score
    bins = np.arange(0,1,0.1)
    if seed < 2:
      ax,_ = CAFplotters.back_sig_hist([nue_shw_pot1.loc[nue_shwE],nuecc_shw_pot1.loc[nuecc_shwE]],labels,f'{CAFpic.shwprefix}razzle.electronScore',precut_signal,
      xlabel=r'Electron Razzle Score',alpha=0.8,bins=bins)
      ax.set_xlim([0,1])
      plotters.save_plot(f'erazzle_precut{seed}')
      plt.close()
    cut3 = CAFpic.cut_razzlescore(nuecc_shw_pot1.loc[nuecc_shwE],cutoff=razzlecutoff)
    events_cut_reco[0,seed,recocut_iter] = cut3.index.drop_duplicates().shape[0]
    
    #print(nue_shwE,nue_reco_pot1[razzle_nuecc_index].index.drop_duplicates())
    cut3_nue = CAFpic.cut_razzlescore(nue_shw_pot1.loc[nue_shwE],cutoff=razzlecutoff)
    events_cut_reco[1,seed,recocut_iter] = cut3_nue.index.drop_duplicates().shape[0]
    recocut_iter+=1

    if seed < 2:
      ax,_ = CAFplotters.back_sig_hist([cut3_nue,cut3],labels,f'{CAFpic.shwprefix}razzle.electronScore',precut_signal,
      xlabel=r'Electron Razzle Score',alpha=0.8,bins=bins)
      ax.set_xlim([0,1])
      plotters.save_plot(f'erazzle_cut{seed}')
      plt.close()

    #Calc thetae and Etheta^2
    nuecc_1 = CAFpic.calc_thetat(cut3,return_key=f'{CAFpic.shwprefix}thetal',px_key=f'{CAFpic.shwprefix}dir.x',
      py_key=f'{CAFpic.shwprefix}dir.y',pz_key=f'{CAFpic.shwprefix}dir.z') #Use direction method
    nuecc_2 = CAFpic.calc_Etheta(nuecc_1,return_key=f'{CAFpic.shwprefix}Etheta2',E_key=f'{CAFpic.shwprefix}bestplane_energy',
      theta_l_key=f'{CAFpic.shwprefix}thetal')
    
    nue_1 = CAFpic.calc_thetat(cut3_nue,return_key=f'{CAFpic.shwprefix}thetal',px_key=f'{CAFpic.shwprefix}dir.x',
      py_key=f'{CAFpic.shwprefix}dir.y',pz_key=f'{CAFpic.shwprefix}dir.z') #Use direction method
    nue_2 = CAFpic.calc_Etheta(nue_1,return_key=f'{CAFpic.shwprefix}Etheta2',E_key=f'{CAFpic.shwprefix}bestplane_energy',
      theta_l_key=f'{CAFpic.shwprefix}thetal')

    #Etheta2 cuts - should be looking at just single shower events

    #print(nue_2.head(20),nuecc_2.head(20))

    if seed < 2:
      ax,_ = CAFplotters.back_sig_hist([nue_2,nuecc_2],labels,f'{CAFpic.shwprefix}Etheta2',precut_signal,xlabel=r'$E_e\theta_e^2$ [GeV rad$^2$]',alpha=0.8)
      ethetamin,ethetamax = ax.get_xlim()
      plotters.save_plot(f'reco_etheta_precut{seed}')
      plt.close()

      etheta_zoom_bins = np.arange(0,0.01,0.001)
      # ax,_ = CAFplotters.back_sig_hist([nue_2,nuecc_2],labels,f'{CAFpic.shwprefix}Etheta2',precut_signal,xlabel=r'$E_e\theta_e^2$ [GeV rad$^2$]',
      #   annotate=False,alpha=0.8,bins=etheta_zoom_bins)
      ax,_ = CAFplotters.back_sig_hist([nue_2,nuecc_2],labels,f'{CAFpic.shwprefix}Etheta2',precut_signal,xlabel=r'$E_e\theta_e^2$ [GeV rad$^2$]',
        annotate=False,alpha=0.8,bins=np.arange(0,0.01,0.001))
      ethetamin_zoom,ethetamax_zoom = ax.get_xlim()
      plotters.save_plot(f'reco_etheta_precut_zoom{seed}')
      plt.close()
      CAFplotters.back_sig_hist([nue_2,nuecc_2],labels,f'{CAFpic.shwprefix}bestplane_energy',precut_signal,xlabel=r'$E_e$ [GeV]',alpha=0.8)
      plotters.save_plot(f'reco_e_precut{seed}')
      ax,_ = CAFplotters.back_sig_hist([nue_2,nuecc_2],labels,f'{CAFpic.shwprefix}thetal',precut_signal,xlabel=r'$\theta_e$ [rad]',alpha=0.8)
      #thetamin,thetamax = ax.get_xlim()
      thetamin_y,thetamax_y = ax.get_ylim()
      plotters.save_plot(f'reco_theta_precut{seed}')
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_2,nuecc_2],labels,f'{CAFpic.shwprefix}thetal',precut_signal,xlabel=r'$\theta_e$ [rad]',alpha=0.8,
        bins=thetabins)
      thetamin_y,thetamax_y = ax.get_ylim()
      plotters.save_plot(f'reco_theta_precut_zoom{seed}')
      plt.close()

    #Make cuts and prepare seperate dfs for shower and track confusion matrices
    print('nuecc')
    nuecc_cut1,_ = CAFpic.make_reco_cuts(nuecc_2,Etheta=Etheta1)
    nuecc_cut1_indeces = nuecc_cut1.index.drop_duplicates()
    nuecc_nreco_pot1_cut1 = CAFpic.get_df_keepindeces(nuecc_nreco_pot1,nuecc_cut1_indeces)
    nuecc_mcprim_pot1_cut1 = CAFpic.get_df_keepindeces(nuecc_mcprim_pot1,nuecc_cut1_indeces)
    events_cut_reco[0,seed,recocut_iter] = CAFpic.number_events(nuecc_cut1)

    print('nue')
    nue_cut1,_ = CAFpic.make_reco_cuts(nue_2,Etheta=Etheta1)
    nue_cut1_indeces = nue_cut1.index.drop_duplicates()
    nue_nreco_pot1_cut1 = CAFpic.get_df_keepindeces(nue_nreco_pot1,nue_cut1_indeces)
    nue_mcprim_pot1_cut1 = CAFpic.get_df_keepindeces(nue_mcprim_pot1,nue_cut1_indeces)
    events_cut_reco[1,seed,recocut_iter] = CAFpic.number_events(nue_cut1)

    recocut_iter+=1

    #Get confusion matrices
    #print(nuecc_mcprim_pot1_cut1.keys())

    if seed == 999:
      shwconfusion_cut1[0,seed,:,:] = CAFpic.get_shw_confusion_matrix(nuecc_nreco_pot1_cut1,nuecc_mcprim_pot1_cut1,n=nshw)
      trkconfusion_cut1[0,seed,:,:] = CAFpic.get_trk_confusion_matrix(nuecc_nreco_pot1_cut1,nuecc_mcprim_pot1_cut1,n=ntrk)
      
      shwconfusion_cut1[1,seed,:,:] = CAFpic.get_shw_confusion_matrix(nue_nreco_pot1_cut1,nue_mcprim_pot1_cut1,n=nshw)
      trkconfusion_cut1[1,seed,:,:] = CAFpic.get_trk_confusion_matrix(nue_nreco_pot1_cut1,nue_mcprim_pot1_cut1,n=ntrk)

    if seed < 2:
      ax,_ = CAFplotters.back_sig_hist([nue_cut1,nuecc_cut1],labels,f'{CAFpic.shwprefix}Etheta2',precut_signal,xlabel=r'$E_e\theta_e^2$ [GeV rad$^2$]',alpha=0.8)
      #ax.set_xlim([ethetamin,ethetamax])
      plotters.save_plot(f'reco_etheta_cut1_{seed}')
      plt.close()

      ax,_ = CAFplotters.back_sig_hist([nue_cut1,nuecc_cut1],labels,f'{CAFpic.shwprefix}Etheta2',precut_signal,xlabel=r'$E_e\theta_e^2$ [GeV rad$^2$]',alpha=0.8,
        bins=etheta_zoom_bins)
      ax.set_xlim([ethetamin_zoom,ethetamax_zoom])
      plotters.save_plot(f'reco_etheta_cut1_zoom{seed}')
      plt.close()
      CAFplotters.back_sig_hist([nue_cut1,nuecc_cut1],labels,f'{CAFpic.shwprefix}bestplane_energy',precut_signal,xlabel=r'$E_e$ [GeV]',alpha=0.8)
      plotters.save_plot(f'reco_e_cut1_{seed}')
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_cut1,nuecc_cut1],labels,f'{CAFpic.shwprefix}thetal',precut_signal,xlabel=r'$\theta_e$ [rad]',alpha=0.8,
        bins=thetabins)
      #ax.set_xlim([thetamin,thetamax])
      plotters.save_plot(f'reco_theta_cut1_{seed}')
      ax.set_ylim([thetamin_y,thetamax_y])
      plotters.save_plot(f'reco_theta_cut1_zoom{seed}')
      plt.close()
    print('nuecc')

    nuecc_cut2,_ = CAFpic.make_reco_cuts(nuecc_2,Etheta=Etheta2)
    events_cut_reco[0,seed,recocut_iter] = CAFpic.number_events(nuecc_cut2)
    print('nue')
    
    nue_cut2,_ = CAFpic.make_reco_cuts(nue_2,Etheta=Etheta2)
    events_cut_reco[1,seed,recocut_iter] = CAFpic.number_events(nue_cut2)
    recocut_iter+=1

    if seed < 2:
      ax,_ = CAFplotters.back_sig_hist([nue_cut2,nuecc_cut2],labels,f'{CAFpic.shwprefix}Etheta2',precut_signal,xlabel=r'$E_e\theta_e^2$ [GeV rad$^2$]',alpha=0.8)
      #ax.set_xlim([ethetamin,ethetamax])
      plotters.save_plot(f'reco_etheta_cut2_{seed}')
      plt.close()

      ax,_ = CAFplotters.back_sig_hist([nue_cut2,nuecc_cut2],labels,f'{CAFpic.shwprefix}Etheta2',precut_signal,xlabel=r'$E_e\theta_e^2$ [GeV rad$^2$]',alpha=0.8,
        bins=etheta_zoom_bins)
      ax.set_xlim([ethetamin_zoom,ethetamax_zoom])
      plotters.save_plot(f'reco_etheta_cut2_zoom{seed}')
      plt.close()
      CAFplotters.back_sig_hist([nue_cut2,nuecc_cut2],labels,f'{CAFpic.shwprefix}bestplane_energy',precut_signal,xlabel=r'$E_e$ [GeV]',alpha=0.8)
      plotters.save_plot(f'reco_e_cut2_{seed}')
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_cut2,nuecc_cut2],labels,f'{CAFpic.shwprefix}thetal',precut_signal,xlabel=r'$\theta_e$ [rad]',alpha=0.8,
        bins=thetabins)
      #ax.set_xlim([thetamin,thetamax])
      plotters.save_plot(f'reco_theta_cut2_{seed}')
      ax.set_ylim([thetamin_y,thetamax_y])
      plotters.save_plot(f'reco_theta_cut2_zoom{seed}')
      plt.close()

    print('nuecc')

    nuecc_cut3,_ = CAFpic.make_reco_cuts(nuecc_2,Etheta=Etheta3)
    events_cut_reco[0,seed,recocut_iter] = CAFpic.number_events(nuecc_cut3)
    print('nue')
    
    nue_cut3,_ = CAFpic.make_reco_cuts(nue_2,Etheta=Etheta3)
    events_cut_reco[1,seed,recocut_iter] = CAFpic.number_events(nue_cut3)
    recocut_iter+=1

    if seed < 2:
      ax,_ = CAFplotters.back_sig_hist([nue_cut3,nuecc_cut3],labels,f'{CAFpic.shwprefix}Etheta2',precut_signal,xlabel=r'$E_e\theta_e^2$ [GeV rad$^2$]',alpha=0.8)
      #ax.set_xlim([ethetamin,ethetamax])
      plotters.save_plot(f'reco_etheta_cut3_{seed}')
      plt.close()

      ax,_ = CAFplotters.back_sig_hist([nue_cut3,nuecc_cut3],labels,f'{CAFpic.shwprefix}Etheta2',precut_signal,xlabel=r'$E_e\theta_e^2$ [GeV rad$^2$]',alpha=0.8,
        bins=etheta_zoom_bins)
      ax.set_xlim([ethetamin_zoom,ethetamax_zoom])
      plotters.save_plot(f'reco_etheta_cut3_zoom{seed}')
      plt.close()
      CAFplotters.back_sig_hist([nue_cut3,nuecc_cut3],labels,f'{CAFpic.shwprefix}bestplane_energy',precut_signal,xlabel=r'$E_e$ [GeV]',alpha=0.8)
      plotters.save_plot(f'reco_e_cut3_{seed}')
      plt.close()
      ax,_ = CAFplotters.back_sig_hist([nue_cut3,nuecc_cut3],labels,f'{CAFpic.shwprefix}thetal',precut_signal,xlabel=r'$\theta_e$ [rad]',alpha=0.8,
        bins=thetabins)
      #ax.set_xlim([thetamin,thetamax])
      plotters.save_plot(f'reco_theta_cut3_{seed}')
      ax.set_ylim([thetamin_y,thetamax_y])
      plotters.save_plot(f'reco_theta_cut3_zoom{seed}')
      plt.close()

    
    print(f'nuecc passed cut1 ind {nuecc_cut3.index.drop_duplicates()}')
    print(f'nue passed cut1 ind {nue_cut3.index.drop_duplicates()}')
    #POT
    events_cut_reco[:,seed,recocut_iter] = pot1

    t5 = time()
    print(f'Reco cuts seed {seed*SEED}: {t5-t4:.1f} (s)')

#Save cut information
for row in range(2): #Iterate by rows, one for each sample
  if row == 0:
    sample = 'nuecc'
  if row == 1:
    sample = 'nue'
  if truthstudy:
    events_cut[row,-2,:] = np.std(events_cut[row,:-2],axis=0) #standard deviation
    events_cut[row,-1,:] = np.mean(events_cut[row,:-2],axis=0) #mean
    cuts_df = pd.DataFrame(events_cut[row],columns=columns)
    cuts_df.to_csv(f'{DATA_DIR}truth_cuts_{sample}{suffix}.csv')

  if recostudy:
    events_cut_reco[row,-2,:] = np.std(events_cut_reco[row,:-2],axis=0) #standard deviation
    events_cut_reco[row,-1,:] = np.mean(events_cut_reco[row,:-2],axis=0) #mean
    cutsreco_df = pd.DataFrame(events_cut_reco[row],columns=columns_reco)
    cutsreco_df.to_csv(f'{DATA_DIR}reco_cuts_{sample}{suffix}.csv')

  #Save confusion matrix info
  if recostudy:
    for i in range(nshw):
      for j in range(nshw):
        shwconfusion[row,-2,i,j] = np.std(shwconfusion[row,:-2,i,j]) #Mean value of ith jth comp, exclude last two since these are for std,mean
        shwconfusion[row,-1,i,j] = np.mean(shwconfusion[row,:-2,i,j]) #Mean value of ith jth comp, exclude last two since these are for std,mean 
        shwconfusion_cut1[row,-2,i,j] = np.std(shwconfusion_cut1[row,:-2,i,j]) #Mean value of ith jth comp, exclude last two since these are for std,mean
        shwconfusion_cut1[row,-1,i,j] = np.mean(shwconfusion_cut1[row,:-2,i,j]) #Mean value of ith jth comp, exclude last two since these are for std,mean 

    #sns.heatmap(shwconfusion[n], annot=True)
    np.savetxt(f'{DATA_DIR}shwconfusion_std_{sample}{suffix}.csv',shwconfusion[row,-2],delimiter=',')
    np.savetxt(f'{DATA_DIR}shwconfusion_mean_{sample}{suffix}.csv',shwconfusion[row,-1],delimiter=',')
    np.savetxt(f'{DATA_DIR}shwconfusion_n0_{sample}{suffix}.csv',shwconfusion[row,0],delimiter=',')

    np.savetxt(f'{DATA_DIR}shwconfusion_std_cut1{sample}{suffix}.csv',shwconfusion_cut1[row,-2],delimiter=',')
    np.savetxt(f'{DATA_DIR}shwconfusion_mean_cut1{sample}{suffix}.csv',shwconfusion_cut1[row,-1],delimiter=',')
    np.savetxt(f'{DATA_DIR}shwconfusion_n0_cut1{sample}{suffix}.csv',shwconfusion_cut1[row,0],delimiter=',')

    for i in range(ntrk):
      for j in range(ntrk):
        trkconfusion[row,-2,i,j] = np.std(trkconfusion[row,:-2,i,j]) #Mean value of ith jth comp, exclude last two since these are for std,mean
        trkconfusion[row,-1,i,j] = np.mean(trkconfusion[row,:-2,i,j]) #Mean value of ith jth comp, exclude last two since these are for std,mean 
        trkconfusion_cut1[row,-2,i,j] = np.std(trkconfusion_cut1[row,:-2,i,j]) #Mean value of ith jth comp, exclude last two since these are for std,mean
        trkconfusion_cut1[row,-1,i,j] = np.mean(trkconfusion_cut1[row,:-2,i,j]) #Mean value of ith jth comp, exclude last two since these are for std,mean 

    np.savetxt(f'{DATA_DIR}trkconfusion_std_{sample}{suffix}.csv',trkconfusion[row,-2],delimiter=',')
    np.savetxt(f'{DATA_DIR}trkconfusion_mean_{sample}{suffix}.csv',trkconfusion[row,-1],delimiter=',')
    np.savetxt(f'{DATA_DIR}trkconfusion_n0_{sample}{suffix}.csv',trkconfusion[row,0],delimiter=',')

    np.savetxt(f'{DATA_DIR}trkconfusion_std_cut1{sample}{suffix}.csv',trkconfusion_cut1[row,-2],delimiter=',')
    np.savetxt(f'{DATA_DIR}trkconfusion_mean_cut1{sample}{suffix}.csv',trkconfusion_cut1[row,-1],delimiter=',')
    np.savetxt(f'{DATA_DIR}trkconfusion_n0_cut1{sample}{suffix}.csv',trkconfusion_cut1[row,0],delimiter=',')

    # print(trkconfusion_cut1[row,0])
    # print(trkconfusion[row,0])
    # print(shwconfusion_cut1[row,0])
    # print(shwconfusion[row,0])






