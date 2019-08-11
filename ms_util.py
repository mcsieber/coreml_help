"""
Common Functions, Utilities, and Classes to facilitate ML work
"""
# Projects-wide configurations

from ms_config import *

def str2list(s:str)->list:
  from re import split
  return split(r'[\s\,\:]+',s)

def is_global(sym:str)->bool:
  """Check if a symbol is globally available """
  return sym in globals() or sym in sys.modules

def show_if_available(sym:str):
  """ Show value if available"""
  try:  print(f"{sym:12.24} = ", eval(sym))
  except Exception as e: print(e)

def is_imgfile(f:Upath)->bool:
  """
  True if the file ends in 'jpg' or 'png'

  Args:
    f: str or Path
  """
  f = Path(f)
  return f.is_file() and (f.suffix == '.jpg' or f.suffix == '.png')


def show_names(d:dict, *args):
  """
  Show value of any dict entries matching the patterns supplied in *args.

  Args:
    d(dict):
    args: names or patterns to check

  Returns:
    Prints values if found, Exception info if search triggers an exception
  """

  import re

  for pattern in args:
    for name in d.keys():
      try:
        if re.search(pattern, name):
          print(f"{name:15.15s} = ", d[name])
      except Exception as e:  print(e)
    #print()
  return None




def _fetch_arg_names(count:int):
  """Return the argument text(s) for the caller of the caller of this function

  Get the string for the argument passed to 'pp' - inspect the caller's stack frame
  and extract the argument text from the code context

  DOES NOT handle instances with nested lists. e.g afunc(1,2,bfunc(3,4)) or afunc(1,2,blist[3,4]) ... etc
  """
  import inspect

  arg_separator = re.compile(r',')
  first_arg     = re.compile(r'\(\(([^)]*)') if count > 1 else re.compile(r'\(([^)]*)')
  last_arg      = re.compile(r'([^)]+)\)')
  arg_names     = ["Obj"]
  # Errors    = Union[AttributeError, ValueError, IndexError, TypeError, NameError, KeyError]

  st          = inspect.stack()

  try:      # use 'try' to make sure we dont leak or create unwanted long-lasting ref cycles
    arg_splits = re.split(arg_separator,st[2].code_context[0])
    if len(arg_splits) > 0:
      arg_names   = [first_arg.search(arg_splits[0]).group(1)]
      for a in arg_splits[1:-1] : arg_names.append(a.lstrip())
      arg_names.append(last_arg.search(arg_splits[-1]).group(1))
  except Exception as e:
    print(f"_fetch_arg_names : {e}")
    if arg_splits is not None: print(f"   arg_splits = {arg_splits}")
  finally:    # clean up
    del st

  return arg_names


def _pp_one(name:str, obj, depth:int, maxlen:int, sep:str, ignore):
  """Print name of object, then determine how to format and print the object"""
  import pprint
  import inspect

  _ppf   = pprint.PrettyPrinter(indent=4, depth=depth, width=maxlen).pformat
  _pp    = pprint.PrettyPrinter(indent=4, depth=depth, width=maxlen).pprint

  # Short representation: e.g. int, float ... short string
  if not hasattr(obj,'__len__') : print(name, _ppf(obj)) ; return
  if type(obj) is str and len(obj) <= maxlen : print(name, _ppf(obj)) ; return

  print(name)

  # Structured object
  if hasattr(obj, '__dict__') or hasattr(obj,'__package__'):   # => structured object (eg. class or module)
    if hasattr(obj, '_data'): _pp(obj._data)
    _pp({ k: str(v)[:maxlen] for k, v in inspect.getmembers(obj) if not k.startswith('_') })

  # String w/ separators
  elif type(obj) is str and sep is not None:
        for seg in obj.split(sep) : print('  ',seg)

  # Everything else
  else: _pp(obj)

  print()


def pp( obj, depth:int =3, maxlen:int =60, sep:str =None, ignore=None):
  """
  Selective and Slightly formatted printing.
  Intended to show something useful but also limit line length and control depth of print

  Args:
    obj(s): The object to format and print, tuple or single item
    depth:  For nested objects, descend this many levels. Passed unchanged to python's "pprint.PrettyPrinter"
    maxlen: Limits the length of lines when printing.
    sep:    Separator to use when parsing iterative items
    ignore:   *stop* printing the object if this attribute name is encountered
            (e.g. avoid printing 000's of "weight" values in a neural network object)

  Notes:
    Front-ends python's pprint.PrettyPrinter
    Uses 'inspect' to grab the string for the value passed to the 'pp' call

  """

  objs      = obj if type(obj) == tuple else (obj,)
  arg_names = _fetch_arg_names(len(objs))
  arg_list  = zip(arg_names,objs)

  for name,ob in arg_list :
    display_name = f"{name!s:14} = "
    try: _pp_one(display_name, ob, depth, maxlen, sep, ignore)
    except Exception as e: print(f"{display_name} e")


def ps(sym_string:str): pp(str2list(sym_string))


def show_env(verbose:bool=False):
  """
  Show the extant environment.

  Args:
    verbose: True => print out a *whole* lot more : config paths, config vars and os environment.

  Return:

    Prints platform, system prefix, python, fastai and torch versions, cuda availability and sys path.
    Call this at the start to confirm (or discover) the environment.
  """
  import sys
  from pathlib import Path

  print("\n")
  print('Current dir  = ', Path.cwd()   )
  print('sys.platform = ', sys.platform )
  print('sys.prefix   = ', sys.prefix   )
  print('sys.version  = ', sys.version  )

  show_if_available('fastai.version','fastai.version.__version__' )
  show_if_available('torch','torch.version.__version__')

  pp(sys.path)

  print("\nUse 'show_env(True)' to see a lot more detail\n")

  if verbose: # If a lot more detail is needed:
    import os, sysconfig
    sys_config_paths = sysconfig.get_paths()
    sys_config_vars  = sysconfig.get_config_vars()
    print()
    pp(sys_config_paths)
    print()
    pp(sys_config_vars)
    print()
    pp(os.environ)
      
  print()



def path_check(*args:Upath)->bool :
  """
  Check that  paths exist, print when they do, raise warning if any are missing

  Args:
    args: paths to check

  Returns:
    True if all paths exist

  Raises:
    Warning if some paths do not exist

  Sanity check to sort-of emulate similar interactive checking in Jupyter Notebook.
  not really used - consider removing

  """
  not_there = []
  for p in args:
    if Path(p).exists(): print(f"{p} = {Path(p)}")
    else : not_there.append(p)
  missing = len(not_there) > 0
  if missing : UserWarning(f"Some paths do not exist: {not_there}")
  return not missing



def main():
  show_env()
  print("\nmcs utilities loaded")

if __name__ == '__main__': main()






