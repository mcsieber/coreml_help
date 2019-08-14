"""
Functions to help with generation and display of predictions from
CoreML, ONNX, and Torch models.

TODO: Fix - too much repetitive code in "pred" functions.
TODO: Move "show_result" and related into this module?
      Create Class "DisplayResults" to encapsulate a lot of the share functions and data?
"""
from collections import namedtuple
#from enum import Enum,unique
from ms_util import *
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

""" 
Formats and Data
"""
# @unique
# class PredAxis(Enum):
#   IMG   = 0
#   MODEL = 1
#   PROB  = 2
#   IDX   = 2
#   RANK  = 3
#
# class PredPos(Enum):
#   IDX   = 0
#   PROB  = 1
#   LABEL = 2

ImagePrediction = namedtuple('ImagePrediction', 'topI topP topL')
""" Standardizes the format of predictions returned by various models. Used when comparing results."""

ImageRepo  = namedtuple('ImageRepo' , 'mean std labels_url')
""" Formats  normalization stats and URLs for various repositories"""

PredParams = namedtuple('PredParams','func runtime imgsize labels')
""" Specifies prediction params for a model"""

### Data, Data Sources

imagenet = ImageRepo( mean   = [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225],
                     labels_url ='https://s3.amazonaws.com/onnx-model-zoo/synset.txt' )

cifar    = ImageRepo( mean = [0.491, 0.482, 0.447], std=[0.247, 0.243, 0.261], labels_url=None)

mnist    = ImageRepo( mean = [0.15]*3, std  = [0.15]*3, labels_url=None)

#_resize_method = Image.NEAREST

def if_None(x,default): return x if x is not None else default


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw: dict = {}, cbarlabel="", **kwargs):
  """
  Create a heatmap from a numpy array and two lists of labels.

  Parameters
  ----------
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


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
  """
  A function to annotate a heatmap.

  Args:
    im:
        The AxesImage to be labeled.
    data:
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt:
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors:
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold:
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs:
        All other arguments are forwarded to each call to `text` used to create
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


""" 
Layer Calculations
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

def image_pred(topI=Uarray, topP=Uarray, topL=Uarray)->ImagePrediction:
  return ImagePrediction(topI=topI, topP=np.array(topP), topL=topL)

_no_results = ([0], [0.00], ["No Results"])
""" Default ImagePrediction values"""


def pred_for_torch(model, img:Uimage, labels=None, n_top:int=3, )->ImagePrediction:
  """
  Run the Torch Classifier model return the top results.

  This function converts the output from the final layer to a list of probabilities,
  then extracts the top items and associated labels.

  This step is needed because the Torch Resnet50 model does not contain a final softmax layer

  Args:
    model (object): The CoreML model to use for inference
    img (Image.Image): The image to process. Expected to be an image with values 0-255
    labels ([str]): Class Labels for output
    n_top (int): Number of top values to return (default 3)


  Return:
    'ImagePrediction' named tuple with three items:
      - topI [ Indexes to top probabilities ], from argsort
      - topP [ Top probabilities ]
      - topL [ Top Labels ]
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

  return image_pred(topI=topI, topP=topP, topL=topL)


def pred_for_onnx(sess, img:Uimage,labels=None, n_top:int=3 )->ImagePrediction:
  """
  Run the ONNX Classifier model and return the top results as a standardized *ImagePrediction*.

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
    'ImagePrediction' named tuple with three items:
      - topI [ Indexes to top probabilities ], from argsort
      - topP [ Top probabilities ]
      - topL [ Top Labels ]
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

  return image_pred(topI=topI, topP=topP, topL=topL)


def pred_for_o2c(model, img:Uimage,  labels=None, n_top:int=3 )->ImagePrediction:
  """
  Run a CoreML Classifier model that was converted from ONNX; return the top results as a standardized *ImagePrediction*.

  This function converts the output from the final layer to a list of probabilities,
  then extracts the top items and associated labels.

  This step is needed because the ONNX Resnet50 model does not contain a final softmax layer, and the
  conversion to CoreML does not add one. (The native CoreML Resnet50 does have a softmax layer)

  Args:
    model (object): The CoreML model to use for inference
    img (Image.Image): The image to process. Expected to be an image with values 0-255
    n_top (int): Number of top values to return (default 3)
    labels ([str]): Class Labels for output, if needed

  Return:
    'ImagePrediction' named tuple with three items:
      - topI [ Indexes to top probabilities ], from argsort
      - topP [ Top probabilities ]
      - topL [ Top Labels ]
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

  pred = image_pred(topI=topI, topP=topP, topL=topL)
  return pred


def pred_for_coreml(model, img:Uimage, labels=None, n_top:int=3 )->ImagePrediction:
  """
  Run a native CoreML Classifier and return the top results as a standardized *ImagePrediction*.

  Args:
    model (object) : the coreml model to use for the prediction
    img (Image.Image): fitted image to use for test
    n_top (int): Number of top values to return (default 3)

  Return:
    'ImagePrediction' named tuple with three items:
      - topI [ Indexes to top probabilities ], from argsort
      - topP [ Top probabilities ]
      - topL [ Top Labels ]

  If you want to run a CoreML model **converted** from ONNX, use `pred_for_o2c`
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

  return image_pred(topI=topI, topP=topP, topL=topL)



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


def show_pred(img_path: Upath, pred: ImagePrediction, model_id="Model",
              pred2show=3, figsize=(2.0, 3.5), img_size=(200, 200),
              fontsize=12, fontfamily='monospace'):

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

  def __init__(self, params:dict, top_count=3, num_images=8, resize_method=Image.NEAREST):
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
        
        
  def i2m(self,i:int)->str: return self.model_list[i]    
  def m2i(self,m:str)->int: return self.model_dict[m]
  
        
  def classify(self, imgs:list, top_count=None)->list:
    """
    Generate predictions for each of the images, by model
    
    Args:
      imgs (list):
        List of image file paths.
        Updates the value of `num_images` in the Classifier
      top_count (int):
        Reset how many predictions to keep for each img and model.
        If None, use the value already set in the Classifier.
        If set, updates the value saved in the Classifier
        
    Returns:
      In the Classifier it allocated and populates the results list,
        the top_probs ndarray, and the top_classes ndarray.
        Returns the results list. 

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
     Get all predictions for one image and return them in result item dict
    Args:
      img_path (Upath): Path to the image
      
    Uses: 
      `pred_dict` from Classifier
      
    Returns:
      A return item. A dictionary with name, path, and 
        `top_count` class, probability and label predictions for each of the models.
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
  Results
  """
  def __init__(self, predictions:Classifier, pred2show:int=2, figsize=(3.0,3.5),
                     cols=1, imgsize=(224,224), fontsize=12, fontfamily='monospace'):
    """
    Args:
      self ():
      predictions(Classifier):
      pred2show(int):
      fontsize(int): The fontsize
      fontfamily(str): The font family

    Returns:
      DisplayResults Object

    """
    super()
    self.resize_method = predictions.resize_method
    self.model_list = predictions.model_list
    self.results    = predictions.results
    self.results_len = len(self.results)
    self.fontsize   = fontsize
    self.fontfamily = fontfamily
    self.figsize    = figsize
    self.imgsize    = imgsize
    self.cols       = cols
    self.pred2show  = pred2show
    self.model_id = None
    self.ax = None
    self.x = 0
    self.y = 0

  """
  Functions to display prediction results along side the image
  """


  def add_pred(self, pred:ImagePrediction, model_id:str=None, n2show=2,
                     x:int=None, y:int=None):
    """
    Add a Prediction to an existing axs. Call after "show_pred" to show additional model results for an image.
    Args:
      pred (ImagePrediction):
      model_id (str):
      n2show (int):
      x (int):
      y (int):

    Returns:
      current value of x,y coordinates (x,y is also saved in DisplayResults object)
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


  def show_one(self, result:dict, models:list=None,
               pred2show:int = None, img_size=None, figsize=None, fontsize=None, fontfamily=None):
    """
    Show selected or all predictions for one image ( = one result item )
      Args:
        result(dict): The path to the image.
        models(str): Display the (user-assigned) model identifier
        pred2show(int): How many of the top results to show for each prediction.
          Set to True to display immediately after show_pred call
          Set to False to allow additional "add_pred" calls before displaying.
          Use `plt.show()` to display when complete

      Returns:
         axs: object from plt.subplot call
         x:   x position
         y:   y position
         None if there are no predictions
    """
    models      = if_None(models,   self.model_list)
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
    fimg = ImageOps.fit(Image.open(img_path), size=img_size, method=self.resize_method, centering=(0.5, 0.4))
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
      self.add_pred(result[m], model_id=m, n2show=pred2show)

    plt.show()
    return ax

  def show(self, items:Union[int,list], models=None ):
    """
    Show items from the result list
    Args:
      items ():
      models ():

    Returns:

    """
    if type(items) is int:
      if items <= self.results_len :
        self.show_one(self.results[items], models=models)
      return
    if type(items) is not list:
      raise TypeError(f"type(items)={type(items)}; 'items' must be an int or a list of ints")
    for n in items :
      if n <= self.results_len :
        self.show_one(self.results[n], models=models)
    return


  # def show_result(self, result:dict, pred2show:int=3, figsize=(3.0,3.5),
  #                 img_size=(224,224), fontsize=12, fontfamily='monospaced') :
  #   """
  #   Show selected or all predictions for one image ( = one result item )
  #     Args:
  #       result(dict): The path to the image.
  #       img(Image): Image to use (optional, if not passed, image is read from 'img_path')
  #       pred2show(int): How many of the top results to show for each prediction.
  #         Set to True to display immediately after show_pred call
  #         Set to False to allow additional "add_pred" calls before displaying.
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
  #      self.add_pred( result[m], model_id=m, n2show=pred2show)
  #
  #   plt.show()
  #   return ax

  def show_random(self,count=5,models=None) :
    import random
    display_list = random.sample(range(self.results_len), count)
    display_list.sort()
    print(f"\nShowing results {display_list} \n  and top {self.pred2show} probabilities for each model")
    self.show(display_list, models=None)



