"""
Standard Jupyter Notebook set up for host 'ariel' after project name is defined

'%run' from notebook immediately after basic project values (project_name, model_name) have been set.

See "project_setup.py"
"""
import sys
_show_args = sys.argv[1:] # Capture whether to display anything once we are done

# Check for case in which this script is run before importing 'ms_core'

if 'user_lib' not in globals():
  import numpy as np
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
    model_arch   = 'rn50'
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
# models        = models_path  # NO - conflicts w/ fastai - wasSynonym for models directory

# Project Models (not all may be needed)

coreml_stub   = model_name
coreml_name   = Path(f"{model_name}.mlmodel")
coreml_path   = models_dir/coreml_name
coreml_output = models_dir/Path(f"{coreml_stub}-ml-output.txt")

# Show None, Some, or almost All of the created variables
# Default to showing just project variables

def _pt(name_str:str):
  """Prints a list of values conveniently"""
  from re import split
  names = split(r"[,\s]+",name_str)
  for name in names :
    if name == '+': print()
    else : print(f"{name:15} = {eval(name)}")

Current_dir = Path.cwd()
_sl = len(_show_args)

if _sl == 0 or\
   _sl == 1 and _show_args[0] == '--show': _show_args.append('Proj')

if 'None' not in _show_args :

    if 'All'   in _show_args : _show_args.extend(['Proj','Paths','Env'])

    if 'Proj'  in _show_args or 'Project' in _show_args :
        print()
        _pt('project_name model_name proj_dir proj_data_dir models_dir + data_dir notebooks')

    if 'Paths' in _show_args :
        print()
        _pt('coreml_name, coreml_path, coreml_output')

    if 'Env'   in _show_args :
        print()
        current_dir = Path.cwd()
        _pt('Current_dir sys.platform sys.prefix sys.version')
        print()
        _pt('sys.path')


del _sl, _show_args, _pt,  Current_dir # Clean up namespace
