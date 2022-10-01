#!/sbnd/data/users/brindenc/.local/bin/python3.9
#
#Brinden Carlson
#9-29-22
#
#Make plots for nu+e

import sys
sys.path.append('/sbnd/app/users/brindenc/mypython') #My utils path
#from bc_utils.CAFana import pic as CAFpic
from bc_utils.CAFana import plotters as CAFplotters
from bc_utils.utils import pic,plotters
from time import time
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import genfromtxt
from datetime import date

#Params import
import nuecc

#Constants
DATA_DIR = nuecc.DATA_DIR
pot1 = nuecc.pot1
SEED = nuecc.SEED
n = nuecc.n

#pot1=1e19
#n=2

xls = 20 #axis size
tls = 20 #title size
lls = 16 #legend size
suffix = '_full'
#suffix = ''

#Get data
nue_reco_cuts = pd.read_csv(f'{DATA_DIR}reco_cuts_nue{suffix}.csv')
nuecc_reco_cuts = pd.read_csv(f'{DATA_DIR}reco_cuts_nuecc{suffix}.csv')
nue_truth_cuts = pd.read_csv(f'{DATA_DIR}truth_cuts_nue{suffix}.csv')
nuecc_truth_cuts = pd.read_csv(f'{DATA_DIR}truth_cuts_nuecc{suffix}.csv')

shwconfusion_nue = genfromtxt(f'{DATA_DIR}shwconfusion_n0_nue{suffix}.csv',delimiter=',')
shwconfusion_nuecc = genfromtxt(f'{DATA_DIR}shwconfusion_n0_nuecc{suffix}.csv',delimiter=',')
trkconfusion_nue = genfromtxt(f'{DATA_DIR}trkconfusion_n0_nue{suffix}.csv',delimiter=',')
trkconfusion_nuecc = genfromtxt(f'{DATA_DIR}trkconfusion_n0_nuecc{suffix}.csv',delimiter=',')

shwconfusion_nue_mean = genfromtxt(f'{DATA_DIR}shwconfusion_mean_nue{suffix}.csv',delimiter=',')
shwconfusion_nuecc_mean = genfromtxt(f'{DATA_DIR}shwconfusion_mean_nuecc{suffix}.csv',delimiter=',')
trkconfusion_nue_mean = genfromtxt(f'{DATA_DIR}trkconfusion_mean_nue{suffix}.csv',delimiter=',')
trkconfusion_nuecc_mean = genfromtxt(f'{DATA_DIR}trkconfusion_mean_nuecc{suffix}.csv',delimiter=',')

#Get keys from params
nreco_keys = nuecc.nreco_keys
shw_keys = nuecc.shw_keys
mcnu_keys = nuecc.mcnu_keys
mcprim_keys = nuecc.mcprim_keys

#Set params
recocolumns = [1,2,3,4,5,6,7,8]
truthcolumns = [1,2,3,4,5,6,7,8]
#recocolumns = np.arange(1,np.shape(nue_reco_cuts)[1],7)
#truthcolumns = np.arange(1,np.shape(nue_truth_cuts)[1],13)
#print(recocolumns,truthcolumns,np.shape(nue_truth_cuts))

legend_labels=[r'$\nu + e$',r'$\nu_e$CC']
reco_dfs = [nue_reco_cuts,nuecc_reco_cuts] #Signal df must be first
truth_dfs = [nue_truth_cuts,nuecc_truth_cuts] #Signal df must be first

#Reco
ax = CAFplotters.plot_background_cuts(reco_dfs,legend_labels,recocolumns,
  r'Reconstructed Event Selection for $\nu+e$ Events'+'\n'+f'POT = {pot1:.1e}'+r' $N$ = '+f'{n:.0f}')
plotters.save_plot(f'reco_background_cuts{suffix}_N{n}')
plt.close()
ax = CAFplotters.plot_background_cuts([reco_dfs[0]],[legend_labels[0]],recocolumns,
  r'Reconstructed Event Selection for $\nu+e$ Events'+'\n'+f'POT = {pot1:.1e}'+r' $N$ = '+f'{n:.0f}')
plotters.save_plot(f'nuereco_background_cuts{suffix}_N{n}')
plt.close()
ax = CAFplotters.plot_background_cuts([reco_dfs[1]],[legend_labels[1]],recocolumns,
  r'Reconstructed Event Selection for $\nu+e$ Events'+'\n'+f'POT = {pot1:.1e}'+r' $N$ = '+f'{n:.0f}')
plotters.save_plot(f'nueccreco_background_cuts{suffix}_N{n}')
plt.close()
CAFplotters.plot_efficiency_purity(reco_dfs,recocolumns,
  r'Reconstructed Event Selection for $\nu+e$ Events'+'\n'+f'POT = {pot1:.1e}'+r' $N$ = '+f'{n:.0f}',event=-1)
plotters.save_plot(f'reco_pe{suffix}_N{n}')
plt.close()

#Truth
CAFplotters.plot_background_cuts(truth_dfs,legend_labels,truthcolumns,
  r'Truth Event Selection for $\nu+e$ Events'+'\n'+f'POT = {pot1:.1e}'+r' $N$ = '+f'{n:.0f}')
plotters.save_plot(f'truth_background_cuts{suffix}_N{n}')
plt.close()
ax = CAFplotters.plot_background_cuts([truth_dfs[0]],[legend_labels[0]],truthcolumns,
  r'Truth Event Selection for $\nu+e$ Events'+'\n'+f'POT = {pot1:.1e}'+r' $N$ = '+f'{n:.0f}')
ax.set_xlim([0,None])
plotters.save_plot(f'nuetruth_background_cuts{suffix}_N{n}')
plt.close()
ax = CAFplotters.plot_background_cuts([truth_dfs[1]],[legend_labels[1]],truthcolumns,
  r'Truth Event Selection for $\nu+e$ Events'+'\n'+f'POT = {pot1:.1e}'+r' $N$ = '+f'{n:.0f}')
plotters.save_plot(f'nuecctruth_background_cuts{suffix}_N{n}')
plt.close()
CAFplotters.plot_efficiency_purity(truth_dfs,truthcolumns,
  r'Truth Event Selection for $\nu+e$ Events'+'\n'+f'POT = {pot1:.1e}'+r' $N$ = '+f'{n:.0f}',event=-1)
plotters.save_plot(f'truth_pe{suffix}_N{n}')
plt.close()




    






