"""
Functions to help with generation and display of predictions from
CoreML, ONNX, and Torch models.

TODO: Fix - too much repetitive code in "pred" functions.
TODO: Move "show_result" and related into this module?
      Create Class "DisplayResults" to encapsulate a lot of the share functions and data?
"""

from collections import namedtuple
from enum import Enum,unique
from ms_util import *

""" 
Formats and Data
"""

@unique

class PredAxis(Enum):
  IMG   = 0
  MODEL = 1
  PROB  = 2
  IDX   = 2
  RANK  = 3

class PredPos(Enum):
  IDX   = 0
  PROB  = 1
  LABEL = 2

ImagePrediction = namedtuple('ImagePrediction', 'topI topP topL')
""" Standardizes the format of predictions returned by various models. Used when comparing results."""

ImageRepo  = namedtuple('ImageRepo' , 'mean std labels_url')
""" Formats  normalization stats and URLs for various repositories"""

### Data, Data Sources

imagenet = ImageRepo( mean   = [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225],
                     labels_url ='https://s3.amazonaws.com/onnx-model-zoo/synset.txt' )

cifar    = ImageRepo( mean = [0.491, 0.482, 0.447], std=[0.247, 0.243, 0.261], labels_url=None)

mnist    = ImageRepo( mean = [0.15]*3, std  = [0.15]*3, labels_url=None)

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

def image_pred(topI=Uarray, topP=Uarray, topL=Uarray)->ndarray:
  return ImagePrediction(topI=topI, topP=np.array(topP), topL=topL)

_no_results = ([0], [0.00], ["No Results"])
""" Default ImagePrediction values"""


def pred_for_torch(model, img:Uimage, n_top:int=3, labels=None)->ndarray:
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


def pred_for_onnx(sess, img:Uimage, n_top:int=3, labels=None)->ndarray:
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


def pred_for_o2c(model, img:Uimage, n_top:int=3, labels=None )->ndarray:
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


def pred_for_coreml(model, img:Uimage, n_top:int=3 )->ndarray:
  """
  Run a native CoreML Classifier and return the top results as a standardized *ImagePrediction*.

  Args:
    model (object) : the coreml model to use for the prediction
    img (Image.Image): fitted image to use for test
    n_top (int): Number of top values to return (default 3)
    in_name (str): Starting Input Layer name
    out_name (str): Final Output Layer name

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


"""
Functions to display prediction results along side the image
"""

def _fmt_imagenet_label(label: str) -> str:
  """Reverse the order of id and name, so that name comes first"""
  import re
  if re.search("n\d+ ", label):
    t1, t2 = re.split(' ', label, maxsplit=1)
    label = f"{t2} ({t1})"
  return label


def _fmt_results(pred:ImagePrediction, n_2show:int=1)->str:
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

  for i in range(n_2show):
    l = _fmt_imagenet_label(pred.topL[i])
    p = pred.topP[i]
    results += f"  {p:003.02%} {l}\n"
  return results


def add_pred(axs, x:int, y:int,  pred:ImagePrediction, model_id='Model', n_2show=1,
             fontsize=12, fontfamily='monospace'):
  """
  Add a Prediction to an existing axs. Call after "show_pred" to show additional model results for an image.
  Args:
    axs ():
    x ():
    y ():
    pred ():
    model_id ():
    fontsize ():

  Returns:
    axs: The axs passed in.

  Note: The call to show_pred should set 'immediate=False". Use plt.show() to display.
  """

  indent      = 20
  y_per_line  = int(1.9*fontsize)
  #
  y_between_models = fontsize//2
  results_indent   = 5*fontsize

  # The model id
  y = y+y_between_models
  axs.text(x+indent, y, model_id, fontsize=fontsize, fontfamily=fontfamily)

  # The prediction results
  y = y + n_2show*y_per_line
  axs.text(x + results_indent, y, _fmt_results(pred,n_2show=n_2show), fontsize=fontsize, fontfamily=fontfamily)

  return x, y


def show_pred(img_path:Upath, pred:ImagePrediction, model_id=None, n_2show=2, immediate=True,
              img:Image=None, img_size=(224,224), fontsize=12, fontfamily='monospace', fig_size=(2.5,4) ):
  """
  Show the image and predictions.

    Args:
      img_path(Upath): The path to the image
      pred(ImagePrediction): The prediction named tuple containing top argmax indexes and probs and labels.
      model_id(str): Display the (user-assigned) model identifier
      img(Image): Image to use (optional, if not passed, image is read from 'img_path')
      n_2show(int): How many of the top results to show for each prediction.
      fontsize(int): The fontsize
      fontfamily(str): The font family
      immediate(bool):
        Set to True to display immediately after show_pred call
        Set to False to allow additional "add_pred" calls before displaying.  Use plt.show() to display when complete

    Returns:
       axs: object from plt.subplot call
       x:   x position
       y:   y position
       None if there are no predictions
  """

  from matplotlib import pyplot as plt
  from PIL import Image, ImageOps

  y_start    = 4
  y_per_line = int(1.9*fontsize)
  indent     = 20
  img_path   = Path(img_path)

  if pred.topP[0] == 0.00 :
    print(f"No predictions for {img_path.name}; pred={pred}")
    return None

  # The image - show without frame or ticks
  if img is None : img = Image.open(img_path)
  fimg     = ImageOps.fit(img, size=img_size, method=Image.NEAREST, centering=(0.5, 0.4))
  fig, axs = plt.subplots(1, 1, figsize=fig_size, subplot_kw=dict(frame_on=False, xticks=[], yticks=[]))
  axs.imshow(fimg)

  # The image file name
  x = img_size[0]+indent-4
  y = y_start
  axs.text(x, y, img_path.name, fontsize=fontsize, fontfamily=fontfamily)

  # The model and results -
  x = x + indent
  y = y + y_per_line + 2
  x, y = add_pred(axs, x, y, pred, model_id=model_id, n_2show=n_2show, fontsize=fontsize, fontfamily=fontfamily)

  if immediate : plt.show() #  => no more calls to "add_pred" for this image.
  return axs, x, y


import matplotlib
import matplotlib.pyplot as plt


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw:dict={}, cbarlabel="", **kwargs):
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
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

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

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
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
    threshold = im.norm(data.max())/2. if threshold is None else im.norm(threshold)

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

