"""
Python helper functions to facilitate working with CoreML and ONNX and converting from one to the other.

These functions depend on package `coremltools`. If you are converting between ONNX and CoreML,
you will need `onnx_coreml`, `onnx`, and `onnxruntime` as well.

.. tip::
  If you want *real* help with CoreML, I highly recommend **Matthijs Holleman's**
  *“Core ML Survival Guide.”*. Informative and well-written.
  Easy to read, as much as books on this subject can be.

Also,

.. tip::    Use **Netron**

In `coreml_help.py` you will find:

The class "CoremlBrowser" inspection and "model surgery" methods
```
      show_nn         Show a summary of neural network layers by index or name
      connect_layers  Connect the output of one layer to the input of another
      delete_layers   Delete CoreML NN layers by *name*.
      get_nn          Get the layers object for a CoreML neural network
```
Convenience Functions:
```
      show_nn          Show a summary of nn (Function equivalent of `show_nn` method)
      show_head
      show_tail        Convenience functions  of  method `show_nn`
      get_rand_images  Return images (jpg and png) randomly sampled from child dirs.
```
Model Execution and Calculation Functions:
```
     norm_for_imagenet  Normalize using ImageNet values for mean and standard dev.
     pred_for_coreml    Run and show Predictions for a native CoreML model
     pred_for_onnx      Run and show Predictions for a native ONNX model
     pred_for_o2c       Run and show Predictions for a CoreML model converted from ONNX
     softmax
```
Use:

  To use, initialize a browser instance using the '.mlmodel' file

        from coreml_help import CoremlBrowser
        cmb = CoremlBrowser(" ... a '.mlmodel' file " )

  Then the following are initialized:

        cmb.spec        # The protobuf spec
        cmb.nn          # The neural network object
        cmb.layers      # The nn layers array
        cmb.layer_dict  # maps layer names to layer indexes
        cmb.layer_count # the count of nn layers
        cmb.shaper      # The shape inference object for this model

  To show layers 10 - 15 (including shapes)

        cmb.show_nn(10,5)

  To delete the layers named "conv_10" and "relu_14"

        cmb.delete_layers(['conv_10', 'relu_14'])


I wrote these as a learning exercise for my own use. Feedback welcome.
Most of this is based on the work of others,  but there can be no question that any
bugs, errors, misstatements,and, especially, inept code constructs, are entirely mine.

---------------------
"""

# Configuration, common imports and functions

import numpy as np
from pathlib import Path
from collections import namedtuple
from coremltools.proto import Model_pb2
import coremltools.models.model as cm
import coremltools.models.utils as cu

### Convenience Types
# If a type starts with a 'u', it is almost certainly one of these, and defined here
#
if 'uarray' not in globals():
  from typing import Union, List
  from PIL    import Image
  from numpy  import ndarray

  uarray = Union[ndarray, List]
  uimage = Union[ndarray, Image.Image]
  upath  = Union[Path,str]

### Data , data sources and functions #########################

ImageRepo = namedtuple('ImageRepo' ,"mean std labels_url")

imagenet = ImageRepo( mean   = [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225],
                      labels_url ='https://s3.amazonaws.com/onnx-model-zoo/synset.txt' )

cifar    = ImageRepo( mean = [0.491, 0.482, 0.447], std=[0.247, 0.243, 0.261], labels_url=None)

mnist    = ImageRepo( mean = [0.15]*3, std  = [0.15]*3, labels_url=None)


# CoreML Model inspection

class CoremlBrowser(object):
  """
  Encapsulates routines to browse and edit CoreML Models
  """
  def __init__(self, mlmodel_file:upath):
    self.mlmodel_path = Path(mlmodel_file)
    """ Path to the mlmodel file"""
    self.spec   = cu.load_spec(self.mlmodel_path)
    """ (Protobuf) spec for the model"""
    self.shaper = cm.NeuralNetworkShaper(self.spec)
    """ Shape inference object for this model"""
    self.nn     = self.get_nn()
    """ Neural network layers object"""
    self.layers = self.nn.layers
    """ Neural network layers"""
    self.layer_count = len(self.layers)
    self.layer_dict = {layer.name:i for i,layer in enumerate(self.layers)}
    """ Maps a layer name to its index"""
    self.name_len_centile = int(np.percentile(np.array([len(l.name) for l in self.layers]), 90))

  def _repr(self):
    """
    Show something more useful than "object" when called
    """
    all_text = ''
    for n in ('mlmodel_path','layer_count','spec.description'):
      v = eval(f"self.{n}")
      nv_text = f"{n:17} = {v}"
      print(nv_text)
      all_text.join(nv_text)
    return all_text


  def __repr__(self): return self._repr()


  def get_nn(self) -> Model_pb2.Model.neuralNetwork:
    """
    Get the layers object for a CoreML neural network.

    Args:
      spec (Model): The `protobuf` spec. for this CoreML model.
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
  def _tbd(self,l): return "              - "
  def _repf(self, rf): return str.join('x', [str(f) for f in rf]) if len(rf) != 0 else self._ph
  @staticmethod
  def _fmt_add(self,l): return f"add     "
  def _fmt_act(self,l): return f"{l.activation.WhichOneof('NonlinearityType'):8}"
  def _fmt_pool(self, l): return f"pool   " + f"          sz:{self._repf(l.pooling.kernelSize)}  str:{self._repf(l.pooling.stride)}"
  def _fmt_concat(self, l): return f"concat  "
  def _fmt_reshape(self, l): return f"reshape          target:{l.reshape.targetShape}"

  def _fmt_bn(self, l):
    bn  = l.batchnorm
    bc  = f"{bn.channels}"
    return f"bnorm  " + f"{bc:9} ep:{bn.epsilon:.3e}  wc:{len(bn.beta.floatValue) + len(bn.gamma.floatValue)}"

  def _fmt_innerp(self, l):
    c   = l.innerProduct
    ic  = f"{c.outputChannels}x{c.inputChannels}"
    return f"innerp " + f"{ic:9} wc:{len(c.weights.floatValue)}"

  def _fmt_conv(self, l):
    c     = l.convolution
    kc    = f"{c.outputChannels}x{c.kernelChannels}"
    conv1 = f"conv   " + f"{kc:9} sz:{self._repf(c.kernelSize)}  str:{self._repf(c.stride)}"
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
    try:
      s = self.shaper.shape(name)
    except IndexError as e:
      line = f"      - {e} - "
    else:
      line = f"CHW: {s['C']} {s['H']} {s['W']}   SB:{s['S']}{s['B']}"
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
    shout       = self._fmt_shape(layer.name)

    # Assemble the line to print

    return f"{li:3} {layer_name:5}  {inputs} {outputs} {shout:>13}  {_fmt_type(self,layer)}"

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
    shout      = self._fmt_shape(layer.name)

    # Assemble the line(s) to print

    line1     = f"{li:3} {layer_name:32} {inputs :<24}  {_fmt_type(self,layer)}"
    line2     = f"{sp:3} {sp        :32} {outputs:<24}  {shout}"

    return line1 + "\n" + line2 + "\n"


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

    sp = "   " # formatting spacer
    one_line_heading = f"Lay Name{ sp:4}In{sp:8}Out{sp:6}Shapes{sp:7}Type,Chan(s){sp:7}Size,Stride,Dilation,#Wts"
    two_line_heading = f"Lay Name{sp:32}In/Out{sp:24}Type,Chan(s){sp:7}Size,Stride,Dilation,#Wts"

    # If >= 90% layer names are "short", print layer on one line, otherwise use two
    if self.name_len_centile <= break_len:
      format_layer = self._fmt_for_one_line
      heading      = one_line_heading
    else:
      format_layer = self._fmt_for_two_lines
      heading      = two_line_heading

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
    the only field that changes is the **to** layer's *input* field. Note also
    that the  keyword arguments `from_` and `to_` are suffixed by underscores.

    Args:
      from_ (str): The name of the layer supplying the outputs

      to_ (str): The name of the layer receiving the `from` outputs.
                   This layer's `input` field is modified.

      replace (bool): *True* (default) Replaces (overwrites)
                  the 'to' layer's input with the 'from' layer's output.
                  *False* appends the 'from' layer's output to the the 'to' layer's input.

    Return:
      A named tuple describing the change (see examples that follow)

    Examples:

          cmb = CoremlBrowser( ... path to 'mlmodel' file ...)

          cmb.connect_layers(from_='conv336', to_='bnorm409')

      returns:

          ( changed_layer = 'bnorm409',
            input_before  = ['concat408_output', 'add400_output'],
            input_after   = ['conv336_output'] )


          connect_layers(nn, from_='conv100', to_='concat408')

      returns:

          (changed_layer =  'None', error = "Layer ['conv100'] not found")

    """
    from copy import deepcopy

    ldict        = self.layer_dict
    layers       = self.layers
    layer_change = namedtuple('layer_audit','changed_layer input_before input_after error')
    layer_names  = ldict.keys()
    missing      = [ name for name  in [from_, to_] if name not in layer_names ]

    if len(missing) > 0: return layer_change(changed_layer = "NONE",
                                             error         = f"Layer(s) {[missing]} not found",
                                             input_before  = None,
                                             input_after   = None )
    from_layer   = layers[ldict[from_]]
    to_layer     = layers[ldict[to_]]
    input_before = deepcopy(to_layer.input)

    if replace :  # remove the current inputs
      for i in range(len(to_layer.input)):
        to_layer.input.pop()

    for i in range(len(from_layer.output)):
      to_layer.input.append(from_layer.output[i])

    return layer_change(changed_layer = to_layer.name,
                        input_before  = input_before,
                        input_after   = deepcopy(to_layer.input),
                        error = None )


  def delete_layers(self, names_to_delete:[str])->[dict]:
    """
    Delete NN layers by **name**.  Invalid layer names are silently ignored.

    Args:
      names_to_delete ([str]): list of layer names

    Return:
      An array of dicts, one for each deletion

    Example:

          delete_layers(nn,['conv335','bn400','avt500']) # ( assume 'avt500' does not exist)

        returns:
          [
            {'deleted_layer': 'conv335',  'input': ['bn334'], 'output': ['conv335']},
            {'deleted_layer': 'bn400', 'input': ['conv399'], 'output': ['bn400']},
          ]

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


def get_shapes( mlmodelc: upath ) -> dict:
  """
  Get the shape of the network layers from the 'model.espresso.shape' file generated by compiling *.mlmodel*
    (Not really used since incorporation of 'NeuralNetworkShaper' object

  Args:
    mlmodelc(str): The *directory* containing the complied model. If the model name is *"xyz.mlmodel"*,
    this directory will be named *"xyz.mlmodelc"*, in the same directory as the *mlmodel* file (usually).

  Return:
    Dictionary of tuples keyed by layer name. Each tuple is a triplet (C,H,W)
    representing the output shape of that layer.

        `{ layer0_name:(C,H,W), layer1_name:(C,H,W), ... }`
  """
  import json

  mlmodelc = Path(mlmodelc)
  if  mlmodelc is not None  \
      and mlmodelc.exists() \
      and mlmodelc.is_dir() :
    with open(mlmodelc/'model.espresso.shape') as f: shape_dict = json.load(f)
    shapes = { k:(ls['k'],ls['w'],ls['h']) for k,ls in shape_dict['layer_shapes'].items() }
  else:
    raise NotADirectoryError(f"mlmodelc = '{mlmodelc}'")
  return shapes


""" 
Layer Calculations
"""

def softmax( x:uarray )->ndarray:
  """
  Scale values to be between 0.0 - 1.0 so they can be used as probabilities.
  Formula is:
  
      exp(x)/sum(exp(x))

  Args:
    x (Union[List,ndarray]): Values on which to calculate the softmax.
                             Should be ndarray or convertible to an ndarray

  Returns:
    softmax as ndarray

  """

  np_exp = np.exp(np.array(x))
  return np_exp / np.sum(np_exp, axis=0)


def norm_for_imagenet( img:uimage )->ndarray:
  """
  Normalize an image using ImageNet values for mean and standard deviation.

  For each pixel in each channel, scale to the interval [0.0, 1.0] and then
  normalize using the mean and standard deviation from ImageNet.
  The input values are assumed to range from 0-255,
  input type is assumed to be an ndarray,
  or an image format that can be converted to an ndarray.
  Here is the formula:

      normalized_value = (value/255.0 - mean)/stddev
      
      mean = [0.485, 0.456, 0.406]
      std  = [0.229, 0.224, 0.225]

  Args:
    img (Union[ndarray, Image.Image]):
      Image data with values between 0-255. 
      If not an ndarray, must be convertible to one.
      Shape must be either (3,_,_) or (_,_,3)

  Return:
    Normalized image data as an ndarray[float32]

  Raises:
    ValueError: If image shape is not (3,_,_) or (_,_,3), or number of dimensions is not 3

  """
  img = np.array(img)
  if img.ndim != 3 : raise ValueError(f"Image has {img.ndim} dimensions, expected 3")

  # Mean and Stddev for image net
  mean  = imagenet.mean
  std   = imagenet.std

  shape = img.shape
  nimg  = np.zeros(shape).astype('float32')

  # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
  if shape[0] == 3:
    for i in range(3): nimg[i, :, :] = (img[i, :, :] / 255.0 - mean[i]) / std[i]
  elif shape[2] == 3:
    for i in range(3): nimg[:, :, i] = (img[:, :, i] / 255.0 - mean[i]) / std[i]
  else:
    raise ValueError(f"Image shape is {shape}, expected (3,_,_) or (_,_,3)")

  return nimg


""" 
Model Execution and Prediction
"""

def pred_for_o2c(model, img, n_top=3, in_name='data', out_name='resnetv24_dense0_fwd', labels=None):
  """
  Run the CoreML Classifier model that was converted from ONNX and return the top results.

  This function converts the output from the final layer to a list of probabilities,
  then extracts the top items and associated labels.

  This step is needed because the ONNX Resnet50 model does not contain a final softmax layer, and the
  conversion to CoreML does not add one. (The native CoreML Resnet50 does have a softmax layer)

  Args:
    model (object): The CoreML model to use for inference
    img (Image.Image): The image to process. Expected to be an image with values 0-255
    in_name (str): Starting Layer Input name
    out_name (str): Final Layer Output name
    n_top (int): Number of top values to return (default 3)
    labels ([str]): Class Labels for output

  Return:
    dict with four items:

      - topI [ Indexes to top probabilities ], from np.argsort
      - topP [ Top probabilities ]
      - topL [ Top Labels ]
      - topRes [ Top Labels and probabilities as formatted strings ], or [] if labels=None
  """

  topI, topP, topL, topRes = [0], [0.00], ["No Results"], ["No Results"]
  try:
    y = model.predict({in_name:img}, usesCPUOnly=True)

  except Exception as e :
    topRes[0] = f"No Results; Exception: {e}"
    print('Exception:',e)

  else:
    r    = y[out_name]
    res  = np.squeeze(np.array(r))
    prob = softmax(res)
    topI = np.argsort(prob)[:-(n_top+1):-1]
    topP = [ prob[i]*100 for i in topI ],
    topL = [ labels[i]   for i in topI ],
    topRes = [f"{labels[i][:30]:32} {100 * prob[i]:.4g}" for i in topI]
    #
  return dict(topI=topI , topP=topP, topL=topL, topRes=topRes)


def pred_for_onnx(sess, img:uimage, n_top=3, labels=None):
  """
  Run the ONNX Classifier model and return the top results.

  This function

    - normalizes the image data,
    - if needed, massages the data to a shape of (3,_,_)
    - runs the model using `onnxruntime`
    - converts the output from the final layer to a list of probabilities,
    - extracts the top items and associated labels.

  Args:
    sess (object) : the ONNX run-time session(model) to use for prediction
    img (Union[ndarray,Image.Image]):  image or image data to use for test
    n_top (int): Number of top values to return (default 4)
    labels ([str]): Class labels for output

  Return:
    Dict with four items:
    
      - topI [ Indexes to top probabilities ], from argsort
      - topP [ Top probabilities ]
      - topL [ Top Labels ]
      - topRes [ Top Labels and probabilities as formatted strings ], or [] if labels=None
  """
  # Use the image to generate acceptable input for the model
  # - move axes if needed, normalize, add a dimension to make it (1,3,224,224)
  nimg  = np.array(img)
  nimg2 = np.moveaxis(nimg,[0,1,2],[1,2,0]) if nimg.shape[2] == 3 else nimg
  topI, topP, topL, topRes = [0], [0.00], ["No Results"], ["No Results"]

  try: pimg  = norm_for_imagenet(nimg2)
  except Exception as e :
    print('Exception:',e)
    return dict(topI=topI, topP=topP, topL=topL, topRes=topRes)

  x     = np.array([pimg])
  # Get input and output names for the model
  input0 = sess.get_inputs()[0]
  output = sess.get_outputs()[0]
  input0_name = input0.name
  output_name = output.name

  # Run the model
  try:
    r = sess.run([output_name], {input0_name: x})

  except Exception as e :
    topRes[0] = f"No Results; Exception: {e}"
    print('Exception:',e)

  else:  # Get predictions from the results
    res  = np.squeeze(np.array(r))  # eliminate dimensions w/ len=1 , e.g. from (1,1,1000) --> (1000,)
    prob = softmax(res)
    topI = np.argsort(prob)[:-(n_top+1):-1]
    topP = [ prob[i]*100 for i in topI ],
    topL = [ labels[i]   for i in topI ],
    topRes = [f"{labels[i][:30]:32} {100 * prob[i]:.4g}" for i in topI]

  return dict(topI=topI, topP=topP, topL=topL, topRes=topRes)


def pred_for_coreml(model, img, n_top=3, in_name='image', out_name='classLabelProbs'):
  """
  Run a native CoreML Classifier and return the top results.

  Args:
    model (object) : the coreml model to use for the prediction
    in_name (str): Starting Input Layer name
    out_name (str): Final Output Layer name
    img (Image.Image): fitted image to use for test
    n_top (int): Number of top values to return (default 3)

  Return:
    dict with four items:

      - topI [ Indexes to top probabilities ], from argsort
      - topP [ Top probabilities ]
      - topL [ Top Labels ] or [] if labels=None
      - topRes [ Top Labels and probabilities as formatted strings ] or [] if labels=None

  """
  topI, topP, topL, topRes = [0], [0.00], ["No Results"], ["No Results"]

  try:
    y = model.predict({in_name:img}, usesCPUOnly=True)

  except Exception as e :
    topRes[0] = f"No Results; Exception: {e}"
    print('Exception:',e)

  else:
    pdict  = y[out_name]
    prob   = np.array([v for v in pdict.values()])
    labels = np.array([k for k in pdict.keys()])
    topI   = np.argsort(prob)[:-(n_top+1):-1]
    topP   = [prob[i] * 100 for i in topI],
    topL   = [labels[i] for i in topI],
    topRes = [f"{labels[i][:30]:32} {100 * prob[i]:.4g}" for i in topI]

  return dict( topI=topI, topP=topP, topL=topL, topRes=topRes)


def _is_imgfile(f:upath)->bool:
  """True if the file ends in 'jpg' or 'png' """
  f = Path(f)
  return f.is_file() and (f.suffix == '.jpg' or f.suffix == '.png')


def _rand_imgs_fm_dir(dir_path: upath, n_images=40, limit=400) -> list:
  """
  Return a list of image file names chosen randomly from `dir_path`.

  Args:
    dir_path(upath): Path or str for the directory
    n_images (int):  Requested number of image file names
    limit (int):     Limit the number of files used for the random sample.
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

  imgs_in_dir = [f for i, f in zip(range(max_files), dir_path.iterdir()) if _is_imgfile(f)]
  return random.sample(imgs_in_dir, min(len(imgs_in_dir), n_images))


def get_rand_images(dir_path: upath, n_images=100, search_limit=400) -> list:
  """
  Return images (jpg and png) randomly sampled from child directories.

  Args:
    dir_path (upath): The parent directory of the children to search
    n_images (int)  : Total number of images to return (actual number may be less)
    search_limit (int) : Limit on the number of files to sample.
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


def _get_labels(repo: Union[ImageRepo, str]) -> list:
  """
   Get the labels for the specified data source - still a work in progress

   If the data is available locally, use that, otherwise download and save locally

   Args:
     repo (ImageRepo,str): namedTuple for the data source (See example) or a file path as str

   Return:
     List containing the labels

   Example:
     The 'imagenet' ImageRepo looks like this:

          imagenet =
              ImageRepo(mean   = [0.485, 0.456, 0.406],
                        std    = [0.229, 0.224, 0.225],
                        url    = None,
                        labels_url ='https://s3.amazonaws.com/onnx-model-zoo/synset.txt',
                        local  = data_root/'imagenet',
                        images = None,
                        labels = data_root/'imagenet/imagenet_labels.txt',
                       )

   """
  labels_file = repo if type(repo) is str else repo.labels
  if labels_file is not None:
    with open(labels_file, 'r') as list_:
      labels = [line.rstrip() for line in list_]
  else:
    raise NotImplementedError('Labels file not found locally.  Please download and try again')
  return labels


def main():
  print("\ncoreml help functions loaded")


if __name__ == '__main__': main()

