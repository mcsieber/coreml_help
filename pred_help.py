"""
Python helper classes and functions to facilitate generation and display of predictions from CoreML, ONNX, and Torch models.

What's here:

**Class `Classifier`**

To invoke models, and collect and manage the resulting predictions.
For examples, see [coreml_help_examples](https://github.com/mcsieber/coreml_help/blob/master/coreml_help_examples.ipynb)

**Class `Results`**

To display *results*, *agreement*, and *certainty*.
For examples, see [pred_help_examples](https://github.com/mcsieber/coreml_help/blob/master/pred_help_examples.ipynb)

**Model Execution and Calculation Functions**

- `norm_for_imagenet` Normalize using ImageNet values for mean and std dev.
- `pred_for_coreml`   Classify an image using a native CoreML model.
- `pred_for_onnx`     Classify an image using a native ONNX model.
- `pred_for_o2c`      Classify an image using a CoreML model converted from ONNX.
- `softmax`

The general purpose of the *pred* functions is

- On input, take a standard image - e.g. RGB, pixels values from 0-255 - and transform it to be acceptable as input
to the specific model. This might require normalizing the data, or rescaling to the interval 0.0 - 1.0, etc.

- On output, take the output from the model and transform it to an `ImagePrediction`

"""

# pdoc dictionary and helper function - used to document named tuples

__pdoc__ = {}
def _doc(key:str, val:str): __pdoc__[key] = val

# ----------------------------------------------------

from ms_util import *
from typing import Callable
from collections import namedtuple
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import cv2

""" 
Formats and Data
"""

ImagePrediction = namedtuple('ImagePrediction', 'topI topP topL')
# doc
_doc('ImagePrediction','*Namedtuple*: standard format returned from *pred* functions')
_doc('ImagePrediction.topI', 'Indexes to top classes')
_doc('ImagePrediction.topP', 'Top probabilities')
_doc('ImagePrediction.topL', 'Top class Labels')

ImageRepo  = namedtuple('ImageRepo' , 'mean std labels_url')
# doc
_doc('ImageRepo','*Namedtuple* that lists normalization stats and URLs for a repository')
_doc('ImageRepo.mean', '*mean* values for normalization')
_doc('ImageRepo.std', '*std* values for normalization')
_doc('ImageRepo.labels_url', 'URL for class labels')

PredParams = namedtuple('PredParams','func runtime imgsize labels')
# doc
_doc('PredParams', '*Namedtuple*: specifies the prediction function, the runtime session for a model, the expected image size and the class labels')
_doc('PredParams.func', '*pred* function to use')
_doc('PredParams.runtime', 'model object to invoke to generate predictions')
_doc('PredParams.imgsize', 'tuple for the expected image size')
_doc('PredParams.labels', 'List containing the class labels, or None')

### Data, Data Sources

imagenet = ImageRepo( mean   = [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225],
                     labels_url ='https://s3.amazonaws.com/onnx-model-zoo/synset.txt' )
""" Imagenet `ImageRepo` """

cifar    = ImageRepo( mean = [0.491, 0.482, 0.447], std=[0.247, 0.243, 0.261], labels_url=None)
""" Cifar `ImageRepo` """

mnist    = ImageRepo( mean = [0.15]*3, std  = [0.15]*3, labels_url=None)
""" Mnist `ImageRepo` """


def if_None(x:any, default:any )->any:
  """Return `default` if `x` is None."""
  return x if x is not None else default

#  TODO: Move this to ms_util.py


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"],
                     threshold=None, **textkw):
  """
  A function to annotate a heatmap.

  Args:
    im: The AxesImage to be labeled.
    data: Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt: The format of the annotations inside the heatmap.
        This should either use the string format method, e.g. "$ {x:.2f}",
        or be a `matplotlib.ticker.Formatter`.  Optional.
    textcolors: A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold: Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs: All other arguments are forwarded to each call to `text` used to create
        the text labels.
  """
  if not isinstance(data, (list, np.ndarray)): data = im.get_array()

  # Normalize the threshold to the images color range.
  threshold = im.norm(data.max()) / 2. if threshold is None else im.norm(threshold)

  # Set default alignment to center, but allow it to be overwritten by textkw.
  kw = dict(horizontalalignment="center",
            verticalalignment="center")
  kw.update(textkw)

  # Get the formatter in case a string is supplied

  if isinstance(valfmt, str):
    valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

  # Loop over the data and create a `Text` for each "pixel".
  # Change the text's color depending on the data.
  texts = []
  for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
      text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
      texts.append(text)

  return texts


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw: dict = {}, cbarlabel="", **kwargs):
  """
  Create a heatmap from a numpy array and two lists of labels.

  Args
  ----
  data
      A 2D numpy array of shape (N, M).
  row_labels
      A list or array of length N with the labels for the rows.
  col_labels
      A list or array of length M with the labels for the columns.
  ax
      A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
      not provided, use current axes or create a new one.  Optional.
  cbar_kw
      A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
  cbarlabel
      The label for the colorbar.  Optional.
  **kwargs
      All other arguments are forwarded to `imshow`.
  """

  if not ax: ax = plt.gca()

  # Plot the heatmap
  im = ax.imshow(data, **kwargs)

  # Create colorbar
  # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
  # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
  cbar = None

  # We want to show all ticks...
  ax.set_xticks(np.arange(data.shape[1]))
  ax.set_yticks(np.arange(data.shape[0]))

  # ... and label them with the respective list entries.
  ax.set_xticklabels(col_labels)
  ax.set_yticklabels(row_labels)

  # Let the horizontal axes labeling appear on top.
  ax.tick_params(top=True, bottom=False,
                 labeltop=True, labelbottom=False)

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
           rotation_mode="anchor")

  # Turn spines off and create white grid.
  for edge, spine in ax.spines.items():
    spine.set_visible(False)

  ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
  ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
  ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
  ax.tick_params(which="minor", bottom=False, left=False)

  return im, cbar


""" 
=== Layer Calculations ===============================
"""

def softmax(x: Uarray) -> ndarray:
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


def norm_for_imagenet(img: Uimage) -> ndarray:
  """
  Normalize an image using ImageNet values for mean and standard deviation.

  Args:
    img (ndarray,Image.Image): Image data with values between 0-255.
      If not an ndarray, must be convertible to one.
      Shape must be either (3,_,_) or (_,_,3)

  Returns: (ndarray): Normalized image data as an ndarray[float32].

  Raises: (ValueError): If image shape is not (3,_,_) or (_,_,3), or number of dimensions is not 3.

  Notes:

    For each pixel in each channel, scale to the interval [0.0, 1.0] and then
    normalize using the mean and standard deviation from ImageNet.
    The input values are assumed to range from 0-255,
    input type is assumed to be an ndarray,
    or an image format that can be converted to an ndarray.
    Here is the formula:

        normalized_value = (value/255.0 - mean)/stddev

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
  """
  img = np.array(img)
  if img.ndim != 3: raise ValueError(f"Image has {img.ndim} dimensions, expected 3")

  # Mean and Stddev for image net
  mean = imagenet.mean
  std = imagenet.std

  shape = img.shape
  nimg = np.zeros(shape).astype('float32')

  # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
  if shape[0] == 3:
    for i in range(3): nimg[i, :, :] = (img[i, :, :] / 255.0 - mean[i]) / std[i]
  elif shape[2] == 3:
    for i in range(3): nimg[:, :, i] = (img[:, :, i] / 255.0 - mean[i]) / std[i]
  else:
    raise ValueError(f"Image shape is {shape}, expected (3,_,_) or (_,_,3)")

  return nimg

""" 
Model Execution and ImagePrediction
"""

def _image_pred(topI=Uarray, topP=Uarray, topL=Uarray)->ImagePrediction:
  """ Construct and return an `ImagePrediction` tuple"""
  return ImagePrediction(topI=topI, topP=np.array(topP), topL=topL)

_no_results = ([0], [0.00], ["No Results"])
""" Default ImagePrediction values"""


"""
=== Prediction Functions ===============================
"""

def pred_for_coreml(model:Callable, img:Uimage, labels=None, n_top:int=3 )->ImagePrediction:
  """
  Run a native CoreML Classifier and return the top results as a standardized *ImagePrediction*.
  If you want to run a CoreML model **converted** from ONNX, use `pred_for_o2c`

  Args:
    model (object): The coreml model to use for the prediction
    img (Image.Image): Fitted image to use for test
    n_top (int): Number of top values to return (default 3)
    labels (list): Not needed for CoreML, ignored. Kept as an argument for consistency with other "pred" functions.

  Returns:
    ImagePrediction

  Notes:
    The the description for the native CoreML Resnet50 model states that it takes images in BGR format.
    However, converting input images from RGB to BGR results in much poorer predictions than leaving them in RGB.
    So I'm assuming that the model does some pre-processing to check the image and do the conversion on its own.
    Or maybe the description is incorrect.
  """

  topI, topP, topL = _no_results
  in_name, out_name = None, None

  try:

    description = model.get_spec().description
    in_name   = description.input[0].name
    out_name  = description.output[0].name

    y       = model.predict({in_name:img}, useCPUOnly=True)

    pdict   = y[out_name]
    prob    = [v for v in pdict.values()]
    labels  = [k for k in pdict.keys()]
    topI    = np.argsort(prob)[:-(n_top+1):-1]
    topP    = np.array([prob[i]  for i in topI])
    topL    = [labels[i] for i in topI]

  except Exception as e :
    print()
    print(f"Exception from pred_for_coreml(input={in_name}, output={out_name})")
    print(e)

  return _image_pred(topI=topI, topP=topP, topL=topL)


def pred_for_o2c(model, img:Uimage,  labels=None, n_top:int=3 )->ImagePrediction:
  """
  Run a CoreML Classifier model that was converted from ONNX;
  return the top results as a standardized *ImagePrediction*.

  This function converts the output from the final layer to a list of probabilities,
  then extracts the top items and associated labels. This step is needed because
  the ONNX Resnet50 model does not contain a final softmax layer, and the
  conversion to CoreML does not add one. (The native CoreML Resnet50 does have a softmax layer)

  Args:
    model (object): The CoreML model to use for inference
    img (Image.Image): The image to process. Expected to be an image with values 0-255
    n_top (int): Number of top values to return (default 3)
    labels ([str]): Class Labels for output, if needed

  Return:
    ImagePrediction
  """

  topI, topP, topL = _no_results
  in_name,  out_name = None, None

  try:

    description = model.get_spec().description
    in_name   = description.input[0].name
    out_name  = description.output[0].name

    y         = model.predict({in_name:img}, useCPUOnly=True)
    y_out     = y[out_name]
    out_type  = type(y_out)

    if out_type is ndarray: # Case 1: conversion from onnx->coreml
      pvals = np.squeeze(y_out)
    elif out_type is dict:  # Case 2: conversion from torch->onnx->coreml
      pvals   = np.array([v for v in y_out.values()])
      labels  = np.array([k for k in y_out.keys()])
    else:                   # Case ?: Don't know ... probably an error
      raise TypeError(f"Type {out_type} of model output is unexpected or incorrect")

    prob    = softmax(pvals)
    topI    = np.argsort(prob)[:-(n_top+1):-1]
    topP    = [ prob[i]   for i in topI ]
    topL    = [ 'None' if labels is None else labels[i] for i in topI ]

  except Exception as e:
    print(f"Exception from pred_for_o2c(input={in_name}, output={out_name})")
    print(e)

  pred = _image_pred(topI=topI, topP=topP, topL=topL)
  return pred


def pred_for_onnx(sess:object, img:Uimage, labels=None, n_top=3 )->ImagePrediction:
  """
  Run the ONNX Classifier model and return the top results as a standardized *ImagePrediction*.

  This function

    - normalizes the image data,
    - if needed, massages the data to a shape of (3,_,_)
    - runs the model using `onnxruntime`
    - converts the output from the final layer to a list of probabilities,
    - extracts the top items and associated labels.

  Args:
    sess (object): The ONNX run-time session(model) to use for prediction
    img (Union[ndarray,Image.Image]):  Image or image data to use for test
    n_top (int): Number of top values to return
    labels ([str]): Class labels for output

  Return:
    ImagePrediction
  """
  # Use the image to generate acceptable input for the model
  # - move axes if needed, normalize, add a dimension to make it (1,3,224,224)

  topI, topP, topL = _no_results

  # Get input and output names for the model
  input0 = sess.get_inputs()[0]
  output = sess.get_outputs()[0]
  input0_name = input0.name
  output_name = output.name

  try:
    np_img    = np.array(img)
    rs_img    = np.moveaxis(np_img,[0,1,2],[1,2,0]) if np_img.shape[2] == 3 else np_img
    norm_img  = norm_for_imagenet(rs_img)
    x         = np.array([norm_img])

  # Run the model
    r = sess.run([output_name], {input0_name: x})

    # Get predictions from the results
    res  = np.squeeze(np.array(r))  # eliminate dimensions w/ len=1 , e.g. from (1,1,1000) --> (1000,)
    prob = softmax(res)
    topI = np.argsort(prob)[:-(n_top+1):-1]
    topP = [ prob[i]    for i in topI ]
    topL = [ labels[i]  for i in topI ]

  except Exception as e:
    print()
    print(f"Exception from pred_for_onnx(input={input0_name}, output={output_name})")
    print(e)

  return _image_pred(topI=topI, topP=topP, topL=topL)


def pred_for_torch(model:Callable, img:Uimage, labels=None, n_top:int=3, )->ImagePrediction:
  """
  Run the Torch Classifier model return the top results.

  This function converts the output from the final layer to a list of probabilities,
  then extracts the top items and associated labels. This step is needed because the
  Torch Resnet50 model does not contain a final softmax layer

  Args:
    model (object): The CoreML model to use for inference
    img (Uimage): The image to classify. Either an Image or image data in a ndarray with values 0-255.
    labels ([str]): Class Labels for output
    n_top (int): Number of top values to return (default 3)


  Return:
    ImagePrediction

  """
  import torch
  from torch.autograd import Variable
  from torchvision.transforms.functional import to_tensor
  from torchvision.transforms.functional import normalize
  from torch.nn.functional import softmax as t_softmax

  topI, topP, topL = _no_results

  try:
    norm_img      = normalize(to_tensor(img), mean=imagenet.mean, std=imagenet.std)
    reshaped_img  = norm_img.reshape(tuple([1]) + tuple(norm_img.shape))
    img_tensor    = torch.as_tensor(reshaped_img, dtype=torch.float)
    x = Variable(img_tensor)

    y = model(x)

    tout      = t_softmax(y, dim=1)
    top       = tout.topk(n_top)
    topI = top.indices[0].tolist()
    topP = top.values[0].tolist()
    topL = [ labels[i]   for i in topI ]

  except Exception as e :
    print()
    print(f"Exception from pred_for_torch(input={''}, output={''})")
    print(e)

  return _image_pred(topI=topI, topP=topP, topL=topL)




def _fmt_imagenet_label( label: str) -> str:
  """Reverse the order of id and name, so that name comes first"""
  import re
  if re.search("n\d+ ", label):
    t1, t2 = re.split(' ', label, maxsplit=1)
    t1 = f"({t1})"
  else:
    t1, t2 = '', label
  return f"{t2:16.24s} {t1}"


def _fmt_results(pred: ImagePrediction, n2show=2) -> str:
  """
  Return a formatted string for all results in the ImagePrediction tuple

  Args:
    pred(ImagePrediction): "ImagePrediction" named tuple

  Returns:
    a formatted string for result at index 'idx'

  Example:
    '46.05%  Eagle '
  """
  results = ''
  for i in range(n2show):
    l = _fmt_imagenet_label(pred.topL[i])
    p = pred.topP[i]
    results += f"  {p:06.02%} {l}\n"
  return results


def show_pred(img_path:Upath, pred:ImagePrediction, model_id="Model",
              pred2show=3, figsize=(2.0, 3.5), img_size=(200, 200),
              fontsize=12, fontfamily='monospace'):
  """
  Display the image and predictions side by side.

  Args:
    img_path (Union[str,Path]): The path to the image
    pred (ImagePrediction): The prediction tuple returned from the *pred* function
    model_id (str): The model short name
    pred2show (ing): How many of the top probabilities to display
    figsize (tuple): Size of the subplot
    img_size (tuple): Size of the image
    fontsize (int): Font size
    fontfamily (str): Font family
  """

  def add_text(x,y,txt):
    ax.text(x, y, txt, verticalalignment='top', fontsize=fontsize, fontfamily=fontfamily)

  img_path  = Path(img_path)
  indent    = 20
  y_start   = 4
  y_per_line= int(1.9 * fontsize) + 2

  # Show the image  without frame or ticks
  _, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=dict(frame_on=False, xticks=[], yticks=[]))
  ax.imshow( ImageOps.fit(Image.open(img_path), size=img_size, method=Image.NEAREST) )

  # Show the image file name
  x = img_size[0] + indent
  y = y_start
  add_text(x, y, img_path.name)

  # Show the model abbr.
  x += indent
  y += y_per_line
  add_text(x, y, model_id)

  # Show the prediction probabilities
  y += y_per_line
  add_text(x, y, _fmt_results(pred, n2show=pred2show))

  plt.show()

  #

# def preds_result(img_path: Upath, pred_dict: dict) -> dict:
#   """ Get all predictions for one image and return them in result item dict"""
#
#   # Fix RGBA images (or others) on the fly, if possible ...
#   img = Image.open(img_path)
#   if img.mode != 'RGB':
#     print(f'Converting {img.name} to RGB from {img.mode} ')
#     img = img.convert('RGB')
#
#   # Some of the models are picky about the image size ...
#   img_sized = { 224 : ImageOps.fit(img, (224, 224), method=_resize_method, centering=(0.5, 0.4) ),
#                 300 : ImageOps.fit(img, (300, 300), method=_resize_method, centering=(0.5, 0.4) ) }
#
#   # Create result item for this image using the prediction dictionary
#   res = { 'name': img.name, 'path':img_path}
#   for model, pred in pred_dict:
#     res[model] = pred.func(pred.runtime, img_sized[pred.imgsize], pred.labels)
#
#   return res
  
class Classifier:
  """
  This class keeps the models to be run and captures their predictions.
  """
  def __init__(self, params:dict, top_count=3, num_images=8, resize_method=Image.NEAREST):
    """
    Args:
      params (dict): A dictionary containing a `PredParams` for each model.
        Specifies the *pred* function,  arguments to use to invoke each model.
      top_count (int): How many top prediction values (class indexes, probabilities) to keep
      num_images (int): Placeholder for number of images to process.
      resize_method (enum): How to resize the image. Defaults to Image.NEAREST
    """
    self.pred_params  = params
    self.model_list = [m for m in self.pred_params.keys() ]
    self.model_dict = {m:i for i,m in enumerate(self.model_list)}
    self.num_models = len(self.model_list)
    self.num_imgs   = num_images
    self.resize_method = resize_method
    self.top_count  = top_count
    self.top_probs  = None
    self.top_classes= None
    self.results    = None
    self.stat_int   = None
        
        
  def i2m(self,i:int)->str:
    """
    Return the model short name for the index `i`
    Args:
      i (int): Index into the `model_list`
    """
    return self.model_list[i]


  def m2i(self,m:str)->int:
    """
    Return the index into the model_list for the model short name
    Args:
      m (str): Short name (or id, or abbreviation) for the model.
    """
    return self.model_dict[m]
  
        
  def classify(self, imgs:list, top_count=None)->list:
    """
    Generate predictions for each of the images, by model.
    Populates the Classifier `results` list, the `top_probs` ndarray, and the `top_classes` ndarray.
    
    Args:
      imgs (list): List of image file paths.
        Updates the value of `num_images` in the Classifier
      top_count (int): Reset how many predictions to keep for each img and model.
        If None, use the value already set in the Classifier.
        If set, updates the value saved in the Classifier
        
    Returns:
        The `results` list. There is one entry in the list for each image. Each entry
        is a dict with predictions for the image, by model.
    """
    # Generate values, allocate arrays
    self.num_imgs    = len(imgs)
    self.stat_int    = max(16,int(self.num_imgs/4))
    self.top_probs   = np.empty((self.num_models, self.num_imgs, self.top_count), dtype=float)
    self.top_classes = np.empty((self.num_models, self.num_imgs, self.top_count), dtype=int)
    self.results     = [self.pred_params] * self.num_imgs
    if top_count is not None: self.top_count = top_count
    
    # Get predictions for each image, save results
    # Save copies of class indexes and probabilities for later calculation
    
    for i, img_path in enumerate(imgs):
        self.results[i] = self.preds_for(img_path)
        
        for im, model in enumerate(self.model_list): 
            model_result = self.results[i][model]
            self.top_classes[im][i] = np.array(model_result.topI)
            self.top_probs[im][i]   = np.array(model_result.topP)
            
        if (i%self.stat_int) == 0 : 
            print(f"{i} of {self.num_imgs} processed, most recent is {img_path.name}")

    print(f"Total of {self.num_imgs} images processed")
    return self.results

  def preds_for(self, img_path: Upath) -> dict:
    """
     Get all predictions for one image and return them in result item dict.
     Will attempt to convert non-RGB images to RGB.

    Args:
      img_path (Upath): Path to the image

    Uses:
      The `pred_params` values from the Classifier
      
    Returns:
      A `returns` list item. A dict with image name, path, and
      the predicted `top_count` classes, probabilities and labels
      for each of the models.

    """
    # Open image, fix RGBA images (or others) on the fly, if possible ...
    img_path = Path(img_path)
    img = Image.open(img_path)
    if img.mode != 'RGB':
      print(f'Converting {img_path.name} to RGB from {img.mode} ')
    img = img.convert('RGB')
    
    # Some of the models are picky about the image size ...
    mid_top  = (0.5, 0.4)
    resize  = self.resize_method
    img_sized= {
        224 : ImageOps.fit(img,(224,224), centering=mid_top, method=resize, ),
        300 : ImageOps.fit(img,(300,300), centering=mid_top, method=resize, ),
    }
    # Create result item for this image using the prediction dictionary
    topn = self.top_count
    result_item = { 'name': img_path.name, 'path':img_path }
    for mname, pred_params in self.pred_params.items():
        pred_for = pred_params.func
        model  = pred_params.runtime
        img    = img_sized[pred_params.imgsize]
        labels = pred_params.labels
        result_item[mname] = pred_for(model, img, labels, topn )

    return result_item


class Results:
  """
  Methods and parameters to
    - display the results of classifying a list of images
    - compare results
    - calculate and display agreement between models
  """
  def __init__(self, classifier:Classifier, pred2show=2, figsize=(3.0,3.5),
                     cols=1, imgsize=(224,224), fontsize=12, fontfamily='monospace'):
    """

    Args:
      classifier (Classifier): The Classifier object containing the results.
      pred2show (int): How many predictions to display
      fontsize (int): The fontsize
      fontfamily (str): The font family

    Returns:
      Results Object

    """
    super()
    self.classifier = classifier
    self.resize_method = classifier.resize_method
    self.model_list = classifier.model_list
    self.model_dict = classifier.model_dict
    self.results    = classifier.results
    self.results_len = len(self.results)
    self.num_imgs    = classifier.num_imgs
    self.top_classes = classifier.top_classes
    self.top_probs  = classifier.top_probs
    self.fontsize   = fontsize
    self.fontfamily = fontfamily
    self.figsize    = figsize
    self.imgsize    = imgsize
    self.cols       = cols
    self.pred2show  = pred2show
    self.m2i = classifier.m2i
    self.i2m = classifier.i2m
    self.x   = 0
    self.y   = 0
    #
    self.ax           = None
    self.model_id     = None
    self.agree        = None
    self.agree_counts = None
    self.agree_diff   = None
    #
    self._init_agreement()



  def _init_agreement(self):

    cf  = self.classifier
    tc  = cf.top_classes
    tp  = cf.top_probs
    CML = cf.m2i('cml')

    # Allocate the 2 and 3 dim arrays we will need
    nm  = cf.num_models
    ni  = cf.num_imgs
    self.agree        = np.empty((nm, nm, ni), dtype=bool)
    self.agree_counts = np.empty((nm, nm), dtype=int)
    self.agree_diff   = np.empty((nm, nm, ni), dtype=float)

    # Populate the agreement tensors (CML will need to be revised ... see below)
    for im, m in enumerate(cf.model_list):
      for ik, k in enumerate(cf.model_list):
        self.agree[im, ik]        = tc[im, :, 0] == tc[ik, :, 0]
        self.agree_counts[im, ik] = self.agree[im, ik].sum()

    # Get accurate CML agreement counts
    for ir, r in enumerate(self.results):
      cml_comp = self._cml_compare(r)
      self.agree[CML, :, ir] = cml_comp
      self.agree[:, CML, ir] = cml_comp

    # Replace CML in the `agree` and `agree_counts` with accurate results
    for im, m in enumerate(cf.model_list):
      cml_sum = self.agree[CML, im, :].sum()
      self.agree_counts[CML, im] = cml_sum
      self.agree_counts[im, CML] = cml_sum

    # Populate the agreement difference matrix
    # If the two models agree on the top class, use the difference in probabilities,
    # if not, use 1.0 (they disagree 100% = no agreement )
    for im, m in enumerate(cf.model_list):
      for ik, k in enumerate(cf.model_list):
        for ir, r in enumerate(self.results):
          self.agree_diff[im, ik, ir]   = abs(tp[im, ir , 0] - tp[ik, ir , 0 ]) if self.agree[im,ik,ir] else 1.0


  def _cml_compare(self, res_item: dict) -> ndarray:
    """
    A results comparison function just for cml ...

    For this result item (i.e. image), compare the top CML label to the top label for the other models
    Return boolean array indicating which models agree with CML and which do not
    """
    # Allocate an empty array, get the CML label, clean it up
    cml_agree = np.empty(len(self.model_list), dtype=bool)
    cml_label = res_item['cml'].topL[0]
    #print("")
    #print(f"cml label   = {cml_label}")
    cml_label.strip(' ,-:;')
    #print(f"cml label s = {cml_label}")
    # Compare the CML label to each of the top labels for the other models
    for im, m in enumerate(self.model_list):
      topL0 = res_item[m].topL[0]
      mitem = re.search(cml_label, topL0)
      cml_agree[im] = (mitem is not None)
      #print(f"m     = {m}")
      #print(f"topL0 = {topL0}")
      #print(f"mitem = {mitem}")
      #print(f"im    = {im}")
      #print(f"cml_agree[im] = { cml_agree[im] }")
      #print("")
    #
    #print(f"cml_agree = {cml_agree}")
    return cml_agree

  def agree_matrix(self):
    """Show a heat-mapped agreement matrix"""
    fig, ax = plt.subplots(figsize=(8, 8))
    am, _   = heatmap(self.agree_counts, self.model_list, self.model_list,
                      ax=ax, cmap="PiYG", cbarlabel="Agreement")
    annotate_heatmap(am, valfmt="{x:d}", textcolors=["white", "black"], size=12)
    am = ax.imshow(self.agree_counts)
    return am

  def best_worst( self, model1:str, model2:str )->(int,int):
    """
    **Agreement** - Returns indexes to the results with the best(= min diff) and worst(= max diff)
    agreement between two models

    Args:
      model1 (str): model id specified in model_params( e.g. "onnx")
      model2 (str): model id specified in in model_params

    """
    M1, M2 = self.m2i(model1), self.m2i(model2)
    # Copy the result array bkz we are going to zap it
    mmd = self.agree_diff[M1,M2].copy()
    best = mmd.argmin()
    # Zero all the "1.0" that represent the diff for non-matching classes
    mmd[mmd == 1.0] = 0.0
    # Now we can get an argmax diff for those classes that do match
    worst  = mmd.argmax()
    return best, worst


  def most_least(self)->list:
    """**Certainty** - Return the most and least certain results for all models"""
    tp = self.classifier.top_probs
    most_least_list  = [ [tp[im,:,0].argmax(), tp[im,:,0].argmin()] for im,m in enumerate(self.classifier.model_list) ]
    return most_least_list


  def _add_pred(self, pred:ImagePrediction, model_id:str=None, n2show=2,
                x:int=None, y:int=None):
    """
    Add a Prediction to an existing axes.

    Args:
      pred (ImagePrediction): Image Prediction named tuple
      model_id (str): Model short name
      n2show (int): How many predictions to display
      x (int): starting x position for the text
      y (int): starting y position for the text

    Returns:
      Current value of x,y coordinates (x,y is also saved in Results object)
    """

    ax = self.ax
    x  = if_None(x,self.x)
    y  = if_None(y,self.y)
    model  = if_None(model_id, self.model_id)
    fontsize = self.fontsize

    name_indent      = 10
    y_per_line  = int(1.9 * fontsize)+2
    y_between   = fontsize // 3
    results_indent  = name_indent + int(4* fontsize)

    # Show the model short name
    y +=  y_between
    ax.text(x + name_indent, y, model, verticalalignment='top',
            fontsize=self.fontsize, fontfamily=self.fontfamily)

    # Show the prediction results
    ax.text(x + results_indent, y, _fmt_results(pred, n2show=n2show),
             verticalalignment='top', fontsize=self.fontsize, fontfamily=self.fontfamily)
    self.x = x
    self.y = y + n2show * y_per_line
    return x, y

  def show_agreement(self,model1:str):
    """Show agreement counts between `model1` and the others"""
    cf    = self.classifier
    M1    = cf.m2i(model1)
    nimgs = cf.num_imgs
    for im, m in enumerate(cf.model_list):
      agreed = self.agree_counts[M1, im]
      print(f"{model1:7} and {m:7} agree on {agreed:4} of {nimgs:4} or {agreed / nimgs:2.2%}")


  def show_one(self, result:dict, models:list=None,
               pred2show:int = None, img_size=None, figsize=None, fontsize=None, fontfamily=None):
    """
    Show selected or all predictions for one image ( = one result list item )

    Args:
      result (dict): The predictions for each model.
      models (list): For display, overrides the list of model names kept in Classifier.
      pred2show (int): How many of the top results to show for each prediction.

    Returns: Object from plt.subplot call or `None` if there are no predictions.

    """
    cf          = self.classifier
    models      = if_None(models,   cf.model_list)
    figsize     = if_None(figsize,  self.figsize)
    img_size    = if_None(img_size, self.imgsize)
    fontsize    = if_None(fontsize, self.fontsize)
    fontfamily  = if_None(fontfamily, self.fontfamily)
    pred2show   = if_None(pred2show, self.pred2show)

    img_path    = Path(result['path'])

    y_start = 0
    y_per_line = int(1.9 * fontsize)+2
    indent = 20

    # Show the image without frame or ticks
    fimg = ImageOps.fit(Image.open(img_path), size=img_size, method=cf.resize_method, centering=(0.5, 0.4))
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=dict(frame_on=False, xticks=[], yticks=[]))
    self.ax = ax
    ax.imshow(fimg)

    # Show the image file name
    x = img_size[0] + indent - 4
    y = y_start
    ax.text(x, y, img_path.name, fontsize=fontsize, fontfamily=fontfamily, verticalalignment='top')

    # Show the model(s) and their prediction probabilities
    self.x = x + indent
    self.y = y + y_per_line
    for m in models:
      self._add_pred(result[m], model_id=m, n2show=pred2show)

    plt.show()
    return ax

  def show(self, items:Union[int,list,tuple], models=None ):
    """
    Show items from the result list
    Args:
      items (list): List of indexes into the results list.
        Or an int to show one item only.
      models (list): Constrains which models to show results for.

    """
    results = self.classifier.results
    rlen    = self.results_len

    if type(items) is int:
      if items <= rlen :
        self.show_one(results[items], models=models)
      return

    if type(items) is not list and type(items) is not tuple:
      raise TypeError(f"type(items)={type(items)}; 'items' must be an int or a list of ints")

    for n in items :
      if n <= rlen :
        self.show_one(results[n], models=models)

  # def show_result(self, result:dict, pred2show:int=3, figsize=(3.0,3.5),
  #                 img_size=(224,224), fontsize=12, fontfamily='monospaced') :
  #   """
  #   Show selected or all predictions for one image ( = one result item )
  #     Args:
  #       result(dict): The path to the image.
  #       img(Image): Image to use (optional, if not passed, image is read from 'img_path')
  #       pred2show(int): How many of the top results to show for each prediction.
  #         Set to True to display immediately after show_pred call
  #         Set to False to allow additional "_add_pred" calls before displaying.
  #         Use `plt.show()` to display when complete
  #
  #     Returns:
  #        axs: object from plt.subplot call
  #        x:   x position
  #        y:   y position
  #        None if there are no predictions
  #   """
  #
  #   models = [m for m in result.keys() if m not in ['name','path']]
  #
  #   img_path = Path(result['path'])
  #
  #   y_start       = 4
  #   y_per_line    = int(1.9 * self.fontsize)
  #   indent        = 20
  #
  #   # Show the image  without frame or ticks
  #   fimg     = ImageOps.fit(Image.open(img_path), size=img_size, method=Image.NEAREST, centering=(0.5, 0.4))
  #   fig, ax  = plt.subplots(1, 1, figsize=figsize, subplot_kw=dict(frame_on=False, xticks=[], yticks=[]))
  #   self.ax  = ax
  #   ax.imshow(fimg)
  #
  #   # Show the image file name
  #   x = img_size[0] + indent - 4
  #   y = y_start
  #   ax.text(x, y, img_path.name, fontsize=fontsize, fontfamily=fontsize)
  #
  #   # Show the model(s) and their prediction probabilities
  #   self.x = x + indent
  #   self.y = y + y_per_line + 2
  #   for m in models[1:len(models)]:
  #      self._add_pred( result[m], model_id=m, n2show=pred2show)
  #
  #   plt.show()
  #   return ax

  def show_random(self,count=5,models=None) :
    import random
    display_list = random.sample(range(self.results_len), count)
    display_list.sort()
    print(f"\nShowing results {display_list} \n  and top {self.pred2show} probabilities for each model")
    self.show(display_list, models=models)



