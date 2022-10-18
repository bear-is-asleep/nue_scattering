import sys
sys.path.append('/sbnd/app/users/brindenc/mypython') #My utils path
from bc_utils.CAFana import pic as CAFpic
from bc_utils.utils import pic


DATA_DIR = '/sbnd/data/users/brindenc/analyze_sbnd/nue/v09_58_02/'
pot1 = 1e21
SEED = 1
n = 100 #Number of universes

#Write down CAF keys we want to use
nreco_keys = ['nshw',
  'ntrk',
  'nstub']
nreco_keys = [CAFpic.recoprefix + key for key in nreco_keys]
shw_keys = ['razzle.electronScore',
  'bestplane_energy',
  'dir.x',
  'dir.y',
  'dir.z',
  'razzle.pdg']
shw_keys = [CAFpic.shwprefix + key for key in shw_keys]
mcnu_keys = ['iscc',
  'position.x',
  'position.y',
  'position.z']
mcnu_keys = [CAFpic.mcnuprefix+key for key in mcnu_keys]
mcprim_keys = ['pdg',
  'gstatus',
  'genT',
  'genE',
  'genp.x',
  'genp.y',
  'genp.z']
mcprim_keys = [CAFpic.primprefix+key for key in mcprim_keys]
