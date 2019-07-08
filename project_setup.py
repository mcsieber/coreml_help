#
### Project Setup for Jupyter Notebook - customize this after first load
#
# Uncomment these after initial load (if you want them)
#
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline
# %alias netron open -a /Applications/Netron.app
#

from ms_core import *

#Project Setup - Names
project_name   = 'change_this'  # Used as prefix for several derived names below
model_name     = 'this_too'
model_arch     = 'and_maybe_this'   # rn50 = Resnet50 , rn34 = Resnet34, etc ...

# Commonly Used Values
img_size       = 300
batch_size     = 32

### Uncomment and change if you don't want the defaults
#
# local_root = Path('/Users/mcsieber/storage')  # On Paperspace would be just "/storage"
# data_root  = Path('/Volumes/ArielD/storage')  # On Paperspace would be just "/storage"
# notebooks  = local_root / 'notebooks'
# user_lib   = local_root / 'lib'


### Run script to continue the setup ..
#
%run -i {user_lib}/the_usual_suspects.py  # --show [ Proj Paths Env All None ] # defaults to 'Proj'