"""
Put run-time configurations here
"""
import sys
import re
import numpy as np
from   pathlib import Path
#
### Starting Landmarks - projects may change these.
# Doing it this way so that PyCharm doesn't whine about undefined symbols
# when checking to avoid exactly that problem, and, if these have
# been changed after initialization, we don't reset them.
#
if re.search('darwin',sys.platform): # => on my Mac
  if 'local_root' not in globals(): local_root = Path('/Users/mcsieber/storage')
  if 'data_root'  not in globals(): data_root  = Path('/Volumes/ArielD/storage')
  if 'user_lib'   not in globals(): user_lib   = local_root/'lib'
  if 'notebooks'  not in globals(): notebooks  = local_root/'notebooks'

elif re.search('linux',sys.platform): # => Assume, for now, Paperspace
  if 'local_root' not in globals(): local_root = Path('/storage')
  if 'data_root'  not in globals(): data_root  = Path('/storage')
  if 'user_lib'   not in globals(): user_lib   = local_root/'lib'
  if 'notebooks'  not in globals(): notebooks  = local_root/'notebooks'

else:     # => ? Lost !
  raise EnvironmentError("Don't recognize this place - are we lost?")

### Convenience Types
# If a type starts with a 'U', it is almost certainly one of these, and defined here
#
from typing import Union, List
from PIL    import Image
from numpy  import ndarray

Uarray = Union[ndarray, List]
Uimage = Union[ndarray, Image.Image]
Upath  = Union[Path, str]

### Set display width and precision for numpy and torch Tensors
# Set for a wide monitor and less noise when showing floats
#
_default_line_width = 133

np.set_printoptions(
  precision=4,
  threshold=400,
  floatmode='maxprec',
  linewidth=133
)
