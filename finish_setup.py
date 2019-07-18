"""
Standard Jupyter Notebook set up for host 'ariel' after project name is defined

'%run' from notebook immediately after basic project values (project_name, model_name) have been set.

See "notebook_setup.py"
"""
import re
import sys

_show_args = sys.argv[1:] # Capture whether to display anything once we are done

# Check for case in which this script is run before importing 'ms_core'

if 'user_lib' not in globals():
  from pathlib import Path
#  from ms_core import *

# Check for case in which this script is run before defining the project and model names

if 'project_name' not in globals() :  project_name = None

if project_name is None:
    print('"project_name" not defined')
    print('Please define "project_name" and "model_name" before invoking this setup code.')
    print('Defaulting to project_name="new" and model_name="new"')
    print()
    project_name = 'new'
    model_name   = 'new'
    model_arch   = 'Resnet50'
    img_size     = 300
    batch_size   = 32

# Naming ... Note, for the example string: "/ArielSV/Users/mcsieber/storage/lib/usual_suspects.py"
#   path    = "ArielSV/Users/mcsieber/storage/lib/usual_suspects.py"
#   drive   = "ArielSV", root = "/", anchor = "ArielSV/"
#   dir     = "ArielSV/Users/mcsieber/storage/lib" (is also a path, but not using it that way)
#   name    = "usual_suspects.py", stub = "usual_suspects", suffix = ".py"
#
# Define these here (or re-define) if not set up by ms_config
if 'data_root' not in globals():
  local_root  = Path('/Users/mcsieber/storage')  # On Paperspace would be just "/storage"
  data_root   = Path('/Volumes/ArielD/storage')  # On Paperspace would be just "/storage"
  notebooks   = local_root/'notebooks' # Synonym for projects directory

data_dir      = data_root/'data'
projects_dir  = local_root/'notebooks'
test_images   = data_dir/'test/images'

# Project Dir and Data
proj_data_dir = data_dir/project_name
proj_dir      = projects_dir/project_name
models_dir    = data_dir/project_name/'models'
xprojects_dir = Path('/Users/mcsieber/dev/xprojects')
# models        = models_path  # NO - conflicts w/ fastai - wasSynonym for models directory

# Project Models (not all may be needed)
fastai_stub   = model_name
fastai_name   = Path(f"{model_name}.pth")
fastai_path   = models_dir/fastai_name
fastai_txt_path = fastai_path.as_posix()

coreml_stub   = model_name
coreml_name   = Path(f"{model_name}.mlmodel")
coreml_path   = models_dir/coreml_name
coreml_output = models_dir/Path(f"{coreml_stub}-ml-output.txt")

onnx_stub     = model_name
onnx_name     = Path(f"{model_name}.onnx")
onnx_dir      = models_dir
onnx_path     = models_dir/onnx_name
onnx_txt_path = onnx_path.as_posix()

# Other
bs = batch_size  # Common Synonym
sz = img_size    # Common Synonym

# Show None, Some, or almost All of the created variables
# Default to showing just project variables

def _pt(name_str:str):
  """Prints a list of conveniently specified values
  all values in a single string, separated by space or comma"""
  from re import split
  names = split(r"[,\s]+",name_str)
  for name in names:
    if len(name)> 0:
      try: print(f"{name:15} = {eval(name)}")
      except Exception as e: print(e)


if re.search('darwin',sys.platform):
  _pt('defaults')
  print()


Current_dir = Path.cwd()
_sl = len(_show_args)

if _sl == 0 or\
   _sl == 1 and _show_args[0] == '--show': _show_args.append('proj')

if 'None' not in _show_args :

    sa = [ str.lower(a) for a in _show_args ]

    if 'all'   in sa : sa.extend(['proj','paths','env'])

    if 'proj'  in sa or 'project' in sa :
        _pt('project_name  model_name  proj_dir  proj_data_dir  models_dir')
        _pt('fastai.version.__version__  torch.version.__version__')
        print()

    if 'paths' in sa :
        _pt('coreml_name  coreml_path  coreml_output')
        print()
        _pt('onnx_name  onnx_dir  onnx_path')
        print()
        _pt('fastai_stub  fastai_name  fastai_path')

    if 'env'   in sa :
        print()
        current_dir = Path.cwd()
        _pt('Current_dir sys.platform sys.prefix sys.version ')
        print()
        _pt('sys.path')


del _sl, _show_args, _pt,  Current_dir # Clean up namespace
