"""
This module contains python classes and functions to facilitate examining and repairing CoreML models.
A companion module, `pred_help`, contains classes and functions to generate, display and compare
predictions from CoreML and other model types.

Here you will find:

  - Class `CoremlBrowser` - View and edit the CoreML *protobuf spec*,
    generate and capture the shapes, and compile the *spec* to produce a MLModel object.

  - Convenience functions to call CoremlBrowser methods, and gather random images.


.. tip::
  If you want *real* help with CoreML, I highly recommend **Matthijs Holleman's**
  *“Core ML Survival Guide.”*. Informative and well-written.
  Easy to read, as much as books on this subject can be.

These functions depend on package `coremltools`. If you are converting between ONNX and CoreML,
you will need `onnx_coreml`, `onnx`, and `onnxruntime` as well.

I wrote these as a learning exercise for my own use. Feedback welcome.
Most of this is based on the work of others,  but there can be no question that any
bugs, errors, misstatements,and, especially, inept code constructs, are entirely mine.

---------------------
"""

# pdoc - dictionary and helper function - used below to document named tuples
__pdoc__ = {}
def _doc(key:str, val:str): __pdoc__[key] = val

import numpy as np
from pathlib import Path
from collections import namedtuple
from coremltools.proto import Model_pb2
from coremltools.models.model import MLModel
import coremltools.models.model as cm

### Convenience Types
# If a type starts with a 'u', it is almost certainly one of these, and defined here
#
if 'Uarray' not in globals():
  from typing import Union, List
  from PIL    import Image
  from numpy  import ndarray

  Uarray = Union[ndarray, List]
  """ ndarray or a list"""
  Uimage = Union[ndarray, Image.Image]
  """ ndarray or an image"""
  Upath  = Union[Path,str]
  """ Path object or a string"""

### Data Formats #########################

LayerAudit = namedtuple('LayerAudit', 'changed_layer input_before input_after error')
# doc
_doc('LayerAudit','Namedtuple to track changes to a CoreML model')
_doc('LayerAudit.changed_layer', '*Name* of the changed layer')
_doc('LayerAudit.input_before', 'Value of the input list *before* any changes')
_doc('LayerAudit.input_after', 'Value of the input list *after* any changes')
_doc('LayerAudit.error', 'Errors, if any')

_sp  = ' '  # Spacer, e.g. f"{_sp:10}"

class CoremlBrowser:
  """
  Class to browse and repair CoreML models.

  To **use**, initialize a browser instance using the '.mlmodel' file
  (Also called the *spec* file or the *protobuf spec* file).

      from coreml_help import CoremlBrowser
      cmb = CoremlBrowser(" ... a .mlmodel file " )

  Then in the browser object following will be **initialized**:

      cmb.spec        # The *protobuf spec*
      cmb.nn          # The neural network object
      cmb.layers      # The neural network layers list
      cmb.layer_dict  # maps layer names to layer indexes
      cmb.layer_count # the count of nn layers
      cmb.shaper      # The shape inference object for this model

  To **show** layers 10 - 15 (including shapes)

      cmb.show_nn(10,5)

  To **delete** the layers named "conv_10" and "relu_14"

      cmb.delete_layers(['conv_10', 'relu_14'])

  The principal inspection and "model surgery" methods are

    - `CoremlBrowser.show_nn`         Show a summary of neural network layers by index or name
    - `CoremlBrowser.connect_layers`  Connect the output of one layer to the input of another
    - `CoremlBrowser.delete_layers`   Delete CoreML NN layers by *name*.

  Associated Convenience Functions:

    - `show_nn`          Show a summary of nn (Function equivalent of show_nn method)
    - `show_head`
    - `show_tail`        Convenience functions  of  method `show_nn`
    - `get_rand_images`  Return images (jpg and png) randomly sampled from child dirs.

  --------
  """

  def __init__(self, mlmodel: Union[Upath, MLModel]):
    """
    Args:
      mlmodel: Either the path to the `protobuf` spec, or an extant `MLModel` object
    """
    self.mlmodel = None
    """ A MLModel object. The result of **compiling** the '.mlmodel' file """
    self.mlmodel_path = None
    """ The full path of the '.mlmodel' file """

    if isinstance(mlmodel, MLModel):
      self.mlmodel = mlmodel
      self.mlmodel_path = None
    elif isinstance(mlmodel, Path) or isinstance(mlmodel, str):
      self.mlmodel_path = Path(mlmodel)
      self.mlmodel = cm.MLModel(self.mlmodel_path.as_posix())
    else:
      raise TypeError("'mlmodel is not a MLModel, a Path, or a file path string")

    self.spec = self.mlmodel.get_spec()
    """ (Protobuf) spec for the model. Also the result of *loading* the '.mlmodel' file """
    self.nn = self.get_nn()
    """ Neural network layers object"""
    self.layers = self.nn.layers
    """ Neural network layers"""
    self.layer_count = len(self.layers)
    """ NUmber  of layers"""
    self.layer_dict = {layer.name: i for i, layer in enumerate(self.layers)}
    """ Maps a layer name to its index"""
    self.name_len_centile = int(np.percentile(np.array([len(l.name) for l in self.layers]), 90))
    """ 90% of the layer names are equal to or shorter than this value"""
    self.shaper = None
    """ Shape inference object for this model"""
    self.layer_shapes = None
    """ Shape dictionary for this model"""
    self.init_shapes()


  def compile_coreml(self)->str:
    """
    Compile the protobuf spec using the OSX `coremlcompiler` application.
    Used to capture shape information when instantiating a CoremlBrowser.

    Uses:
      `CoremlBrowser.spec`, the (Protobuf) spec for the model.

    Returns:
      `stdout` (str): The *stdout* from the compiler,
      which (should) contain shape info, or an empty string.

    Note:
       XCode installs `coremlcompiler`.
       An example of the *OSX shell command* to run the compiler is

      `xcrun coremlcompiler compile rn50.mlmodel rn50_out_dir `

    """
    from sys import platform
    from subprocess import run, CompletedProcess, PIPE

    if platform != 'darwin': return ''
    compile_cmd = ['xcrun', 'coremlcompiler', 'compile', self.mlmodel_path, '.']
    compilation: CompletedProcess = run(compile_cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    return compilation.stdout if compilation.returncode == 0 else ''


  def extract_shapes(self, comp_output:str )->dict:
    """
    Extract the shape of the network layers from the mlmodel compilation output file.

    Args:
      comp_output: The text output captured from compiling the ".mlmodel" file.

    Returns:
      A dictionary of lists keyed by layer name. Each list is a triplet `[C,H,W]` representing the output shape of that layer.

        `{ layer0_name:[C,H,W], layer1_name:[C,H,W] ... }`

    Notes:
      Here is an sample line from the compiler output.

      `Neural ... 174:320, name=507, output shape : (C,H,W)=(4096,1,1)`

    """
    if comp_output is None or  len(comp_output) < 20 :
      print(f"Compilation output string is None or too small",f"compiliation output: {comp_output}" )
      return None

    # Used to extract name and shapes from lines in the out_file

    import re
    name  = re.compile(r"name =\s+([/\w]+),")
    shape = re.compile(r"=\s+\((-?\d+),\s+(-?\d+),\s+(-?\d+)\)")
    lines = re.split("\n",comp_output)

    # The comprehension below pulls name and shapes from each line in the array
    # and outputs them as a dictionary.

    layer_shapes = \
      {n.group(1): [s.group(1), s.group(2), s.group(3)]
       for n, s in [(name.search(ln), shape.search(ln)) for ln in lines] if n and s }

    # First line is a header
    # Second line (index 1) is input name and is a different format - just reported, not compiled

    line1 = lines[1]
    name1 = re.search(r'^\w+', line1).group(0)
    s     = shape.search(line1)
    layer_shapes[name1] = [s.group(1), s.group(2), s.group(3)]

    if len(layer_shapes) == 0:
      raise ValueError(f"Nothing found, check contents of compilation output")

    return layer_shapes


  def init_shapes(self, use_shaper=False)->bool:
    """
    Get shapes for the layers in the model. Compiles model to get shapes.

    Args:
      use_shaper (bool): How to generate and capture the shapes.

        - `False` Ignore the shaper object, try to compile model to get shapes.
        - `True` Use the shaper object if available.

    Return:
      - `True` for success; the instance variable `layer_shapes` contains a valid shape dictionary.
      - `False` for failure.

      The`NeuralNetworkShaper` object crashes python sometimes
      (prob. because the network is invalid in some way),
      so it is not preferred.
    """
    self.layer_shapes = None
    self.shaper       = None

    if use_shaper:
      try:
        self.shaper = cm.NeuralNetworkShaper(self.spec)
      except Exception as e :
        self.shaper = None
        shaper_exception = e
        print("'NeuralNetworkShaper' reports ", shaper_exception)

    if self.shaper is None:
      comp_out = self.compile_coreml()
      self.layer_shapes = self.extract_shapes(comp_out)

    if self.layer_shapes is not None:
      print(f"Using shape info from compilation output")

    if self.shaper is None and self.layer_shapes is None:
      print("  Can't infer shapes because 'NeuralNetworkShaper' is not available")
      print("  and could not compile the model to generate shapes")
      print()




  def _repr(self):
    """
    Show the path, layer count and description for this model.
    (goal is to something more useful than "object" when called)
    """
    all_text = ''

    for n in ('mlmodel_path','layer_count'):
      v = eval(f"self.{n}")
      nv_text = f"{n:17.17} = {v}"
      print(nv_text)
      all_text.join(nv_text)

    if self.layer_shapes is not None:
      nv_text = f"layer_shapes_count = {len(self.layer_shapes)}"
      print(nv_text)
      all_text.join(nv_text)

    # Show this last
    v = self.spec.description
    nv_text = f"\n{n:17.17} = {v}"
    print(nv_text)
    all_text.join(nv_text)

    return all_text


  def __repr__(self): return self._repr()


  def get_shape_for(self, name:str) -> Union[list,dict,None]:
    """
    Try to get the shape for layer `name`

    Args:
      name (str): The name of the layer

    Returns (Union[dict, str]):
      The shape dict object from the shape dictionary if it exists, or
      The shape dict returned by the shaper object if it exists, or
      The text of the exception generated by the shaper object, or
      None
    """
    res = None

    if self.layer_shapes is not None:
      res = self.layer_shapes.get(name)

    if res is None and self.shaper is not None:
      try: res = self.shaper.shape(name)
      except IndexError as e: pass

    return res

  def get_layer_name(self, name:str):
    """Locate and return a nn layer using its name"""
    return self.layers[self.layer_dict[name]]

  def get_layer_num(self, idx:int):
    """Locate and return a nn layer using its index value"""
    return self.layers[idx]

  def get_nn(self) -> Model_pb2.Model.neuralNetwork:
    """
    Get the layers object for a CoreML neural network.

    Uses:
      self.spec (Model): The `protobuf` spec. for this CoreML model.
                Returned by `coremltools.util.load_spec("file.mlmodel")`

    Return:
      The neural network layers of the model or an Attribute Error.
      The precise return type is determined by the value of `spec.WhichOneof("Type")`,
      which should be one of:

        - Model.neuralNetwork
        - Model.neuralNetworkClassifier
        - Model.neuralNetworkRegressor

    Raises:
      AttributeError: if spec is not one of the 3 neuralNetwork sub-classes

    """

    nn_dict = dict(
        neuralNetwork = self.spec.neuralNetwork,
        neuralNetworkRegressor = self.spec.neuralNetworkRegressor,
        neuralNetworkClassifier = self.spec.neuralNetworkClassifier
    )
    nn = nn_dict[self.spec.WhichOneof("Type")]
    if nn is None: raise AttributeError("MLModel is not a neural network sub-class")
    return nn

### ------------------------------------------------ ###

# Field formatting functions
 # Item and line formatting functions

  _ph = '~'  # Placeholder char(s) for strings below ...
  @staticmethod
  def _tbd(self,l): return f"{'-':>8} "
  def _repf(self, rf): return str.join('x', [str(f) for f in rf]) if len(rf) != 0 else self._ph
  #
  def _fmt_act(self,l):      return f"{l.activation.WhichOneof('NonlinearityType'):8} "
  def _fmt_pool(self, l):    return f"{'pool':8} {_sp:9}  sz:{self._repf(l.pooling.kernelSize)}  str:{self._repf(l.pooling.stride)}"
  def _fmt_add(self, l):     return f"{'add':8} "
  def _fmt_concat(self, l):  return f"{'concat':8} "
  def _fmt_reshape(self, l): return f"{'reshape':8} {_sp:9}  target:{l.reshape.targetShape}"

  def _fmt_bn(self, l):
    bn  = l.batchnorm
    bc  = f"{bn.channels}"
    return f"{'bnorm':8} {bc:9}  ep:{bn.epsilon:.3e}  wc:{len(bn.beta.floatValue) + len(bn.gamma.floatValue)}"

  def _fmt_innerp(self, l):
    c   = l.innerProduct
    ic  = f"{c.outputChannels}x{c.inputChannels}"
    return f"{'innerp':8} {ic:9}  wc:{len(c.weights.floatValue)}"

  def _fmt_conv(self, l):
    c     = l.convolution
    kc    = f"{c.outputChannels}x{c.kernelChannels}"
    conv1 = f"{'conv':8} {kc:9}  sz:{self._repf(c.kernelSize)}  str:{self._repf(c.stride)}"
    conv2 = f"  dil:{self._repf(c.dilationFactor)}  wc:{len(c.weights.floatValue)}"
    return conv1 + conv2

  # Maps layer types to formatting functions

  _fmt_funcs = dict(innerProduct=_fmt_innerp, reshape=_fmt_reshape,
                    convolution=_fmt_conv, batchnorm=_fmt_bn,
                    pooling=_fmt_pool, activation=_fmt_act,
                    add=_fmt_add, concat=_fmt_concat)


  def _fmt_shape(self, name: str) -> str:
    """
    Format the shape line
    """
    s = self.get_shape_for(name)

    if s is None : return f"        -   -   - "

    if type(s) is dict:
      if 'k' in s.keys():
        line = f"k h w n: {s['k']:2} {s['h']:2} {s['w']:2} {s['n']:2}"
      else:
        line = f" c h w:  {s['C']:2} {s['H']:2} {s['W']:2}   sb:{s['S']}{s['B']:}"
    if type(s) is list:
        line = f" c h w:  {s[0]:2} {s[1]:2} {s[2]:2}"
    else:
      line   = f" c h w:  {s}"

    return line


  def _fmt_for_one_line(self, layer, li: int) -> str:
    """
    Format one nn layer to print on one line.

    This routine attempts (poorly, so far) to adjust field positions based
    on the length of the layer name.  Layer name length seems to vary
    from 3 chars (Models converted from ONNX) to 24 chars (Apple-generated CoreML models)

    """
    # Field widths for one layer/line
    # layer = 3
    # layer_name (ln) = calculated (max 8)
    # shapes (assume 3x3-digit fields, on avg) = 9+2+2

    # Calculate and construct the parts for each line

    layer_typ   = layer.WhichOneof('layer')
    name_len    = self.name_len_centile
    _fmt_type   = self._fmt_funcs.get(layer_typ, self._tbd)

    w_inputs    = int(name_len * 2) + 4
    w_outputs   = name_len + 3

    layer_name  = format(f"{layer.name}", f"<{name_len}s")
    inputs      = format(f"[{str.join(', ', layer.input)}]", f"<{w_inputs}s")
    outputs     = format(f"[{str.join(', ', layer.output)}]", f"<{w_outputs}s")
    out_shape       = self._fmt_shape(layer.name)

    # Assemble the line to print

    return f"{li:3} {layer_name:8}  {inputs:10} {outputs:10} {out_shape:>13.16}  {_fmt_type(self,layer)}"


  def _fmt_for_two_lines(self,layer, li: int) -> str:
    """
    Format one nn layer to print on two lines.

    This routine attempts (poorly, so far) to adjust field positions based
    on the length of the layer name.  Layer name length seems to vary
    from 3 chars (Models converted from ONNX) up to 24 chars (Apple-generated CoreML models)

    """

    # Calculate and construct the parts for each line

    layer_typ = layer.WhichOneof('layer')
    name_len  = self.name_len_centile
    _fmt_type = self._fmt_funcs.get(layer_typ, self._tbd)

    w_inputs  = name_len + 2  # int(name_len * 2) + 4
    w_outputs = name_len + 2

    sp         = f"   "
    layer_name = format(f"{layer.name}", f"<{name_len}s")
    inputs     = format(f"[{str.join(', ', layer.input)}]", f"<{w_inputs}s")
    outputs    = format(f"[{str.join(', ', layer.output)}]", f"<{w_outputs}s")
    out_shape  = self._fmt_shape(layer.name)

    # Assemble the line(s) to print

    line1     = f"{li:3} {layer_name:24.24} {inputs :<30.48}  {_fmt_type(self,layer)}"
    line2     = f"{sp:3} {sp        :24.24} {outputs:<30.30}  {out_shape}"

    return line1 + "\n" + line2 + "\n"

  # So that these can be changed dynamically, for now
  sp = "   " # formatting spacer
  _one_line_heading = f"Lay Name{ sp:6}In{sp:9}Out{sp:9}Shapes{sp:10}Type,Chan(s){sp:9}Size,Stride,Dilation,#Wts"
  _two_line_heading = f"Lay Name{sp:21}In/Out{sp:26}Type,Chan(s)/Shape{sp:3}Size,Stride,Dilation,#Wts"

  def show_nn(self,  start:Union[int,str]=0, count=4,  break_len=8 ) -> None:
    """
    Beginning at `nn` layer `start`, print a summary of `count` network layers

    Args:

      start (Union[int,str]): The starting layer. Can be an `int` (=>Layer index) or a `str` (=>Layer Name).
        Negative values work backward from the end, similar to lists.

      count (int): How many layers to summarize and print

      break_len (int): Formatting criteria. If most ( ~ 90% ) of the layer names are
        less than or equal to   "break_len", one line is used, otherwise, two lines.

      Inconsistent or invalid values for start and count are repaired by reseting to appropriate defaults

    """
    nn_count = self.layer_count

    # If necessary convert layer name to layer index
    if type(start) is str: start = self.layer_dict[start]

    # Fix any contradictory start and count values
    # If start is negative, simulate list behavior and work backwards from the end
    if count is None or count <= 0 : count = 4
    if start < 0                   : count = 3; start = nn_count + start
    if start + count > nn_count    : start = nn_count - count

    # If >= 90% layer names are "short", print layer on one line, otherwise use two
    if self.name_len_centile <= break_len:
      format_layer = self._fmt_for_one_line
      heading      = self._one_line_heading
    else:
      format_layer = self._fmt_for_two_lines
      heading      = self._two_line_heading

    print(heading)

    # Format and print each layer, include shape values if available

    li = start

    for ly in self.layers[start:start+count]:
      print(format_layer(ly, li))
      li += 1

  """ 
  CoreML Model Surgery - connect and delete layers 
  """

  def connect_layers(self, from_:str, to_:str, replace=True)->namedtuple:
    """
    Connect the output of one CoreML model layer to the input of another.

    Layers are identified by name. An invalid layer name aborts any connection attempt.
    Note that when two layers are *connected*, only one layer is modified:
    the only field that changes is the **to** layer's *input* field.

    Args:
      from_ (str): The name of the layer supplying the outputs

      to_ (str): The name of the layer receiving the `from` outputs. This layer's `input` field is modified.

      replace (bool): **`True`** **replaces** (overwrites) the *to layer*'s `input` with the *from layer*'s `output`.
        **`False`** **appends** the *from layer*'s `output` to the the *to layer*'s `input`.

    Returns:
      A *named tuple* describing the change (see examples that follow)

    Examples:
      The statements
      ```
        cmb = CoremlBrowser( ... path to 'mlmodel' file ...)
        cmb.connect_layers(from_='conv336', to_='bnorm409')
      ```
      returns:
      ```
        ( changed_layer = 'bnorm409',
          input_before  = ['concat408_output', 'add400_output'],
          input_after   = ['conv336_output'] )
      ```
      The statement
      ```
         connect_layers(nn, from_='conv100', to_='concat408')
      ```
      returns:

        `(changed_layer =  'None', error = "Layer ['conv100'] not found")`
    """
    from copy import deepcopy

    ldict        = self.layer_dict
    layers       = self.layers
    layer_names  = ldict.keys()
    missing      = [ name for name  in [from_, to_] if name not in layer_names ]

    if len(missing) > 0:
      return LayerAudit(changed_layer="NONE", input_before=None, input_after=None, error=f"Layer(s) {[missing]} not found")

    from_layer   = layers[ldict[from_]]
    to_layer     = layers[ldict[to_]]
    input_before = deepcopy(to_layer.input)

    if replace :  # remove the current inputs
      for i in range(len(to_layer.input)):
        to_layer.input.pop()

    for i in range(len(from_layer.output)):
      to_layer.input.append(from_layer.output[i])

    return LayerAudit(changed_layer=to_layer.name, input_before=input_before, input_after=deepcopy(to_layer.input), error=None)


  def delete_layers(self, names_to_delete:[str])->[dict]:
    """
    Delete NN layers by **name**.  Invalid layer names are silently ignored.

    Args:
      names_to_delete ([str]): list of layer names

    Return:
      An array of dicts, one for each deletion

    Example:
    ```
      delete_layers(nn,['conv335','bn400','avt500']) # ( assume 'avt500' does not exist)
    ```
    returns:
    ```
      [
        {'deleted_layer': 'conv335', 'input': ['bn334'], 'output': ['conv335']},
        {'deleted_layer': 'bn400', 'input': ['conv399'], 'output': ['bn400']},
      ]
    ```
    """
    from copy import deepcopy
    deleted = []

    for target_name in names_to_delete:
      # to be safe, we have to re-enumerate after every deletion
      for i, layer in enumerate(self.layers):
        if layer.name == target_name :
          deleted.append(
            dict( deleted_layer=target_name, input=deepcopy(layer.input), output=deepcopy(layer.output))
          )
          del self.layers[i]
          break

    # Update the layer count and layer dict kept by the Coreml browser instance
    self.layer_count = len(self.layers)
    self.layer_dict  = {layer.name:i for i,layer in enumerate(self.layers)}

    return deleted

  def compile_spec(self)->MLModel:
    """
    Convenience method to re-compile and save model after editing the spec.
    Returns the compiled spec - the MLModel object. Equivalent to:

      `CoremlBrowser.mlmodel` = `coremltools.models.MLModel`(`CoremlBrowser.spec`)
    """
    self.mlmodel = cm.MLModel(self.spec)
    return self.mlmodel


# Convenience Routines

def show_nn(cmb:CoremlBrowser, start:Union[int, str]=0, count=4, break_len=8):
  """ Convenience for `CoremlBrowser.show_nn()`"""
  cmb.show_nn(start, count=count, break_len=break_len)


def show_head(cmb:CoremlBrowser):
  """ Convenience for `show_nn(nn,0,3)`"""
  show_nn(cmb, 0, 3)

def show_tail(cmb:CoremlBrowser):
  """ Convenience for `show_nn(nn,-3)`"""
  show_nn( cmb, -3)

def is_imgfile(f:Upath)->bool:
  """True if the file ends in 'jpg' or 'png' """
  f = Path(f)
  return f.is_file() and f.suffix in ['.jpg','.png','jpeg']


def _rand_imgs_fm_dir(dir_path: Upath, n_images=40, limit=400) -> list:
  """
  Return a list of image file names chosen randomly from `dir_path`.

  Args:
    dir_path (Upath): Path or str for the directory
    n_images   (int): Requested number of image file names
    limit      (int): Limit the number of files used for the random sample.
                      Avoids un-intentional sampling of very large directorys.

  Returns:
    A list of randomly chosen '.jpg' or '.png' file names.
    Number of files returned could be less than the requested amount

  Note:
    Only known to work on Unixen systems.
  """
  import random

  dir_path = Path(dir_path)
  nlink_count = dir_path.stat().st_nlink  # The file count in directory (so far)
  max_files = min(limit, nlink_count)  # max num files to search in any direct

  # Collect  image files from directory and return a random sample

  imgs_in_dir = [f for i, f in zip(range(max_files), dir_path.iterdir()) if is_imgfile(f)]
  return random.sample(imgs_in_dir, min(len(imgs_in_dir), n_images))

def get_rand_images(dir_path:Upath, n_images=100, search_limit=400)->list:
  """
  Return images (jpg and png) randomly sampled from child directories.

  Args:
    dir_path (Upath): The parent directory of the children to search
    n_images (int): Total number of images to return (actual number may be less)
    search_limit (int): Limit on the number of files to sample.
                         (To avoid performance issues with very large file counts)
  Returns:
    List of image files. Count may be less than requested.

  """
  dir_path = Path(dir_path)

  # Generate list of directories to search for images
  dirs = [d for d in dir_path.iterdir() if d.is_dir()]
  imgs_per_dir = max(1, int(n_images / len(dirs)))
  img_files = []

  for d in dirs:  # Accumulate random images from each child directory in turn
    r = _rand_imgs_fm_dir(d, n_images=imgs_per_dir, limit=search_limit)
    img_files.extend(r)

  return img_files


def main():
  print("\ncoreml help functions loaded")

if __name__ == '__main__': main()
