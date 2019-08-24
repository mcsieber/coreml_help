# Module coreml_help and pred_help

Python helper functions to facilitate working with CoreML and ONNX and converting from one to the other.

### Documentation
- **[coreml_help](https://mcsieber.github.io/coreml_help.html)**
- **[pred_help](https://mcsieber.github.io/pred_help.html)**

These functions depend on package `coremltools`. If you are converting between ONNX and CoreML,
you will need `onnx_coreml`, `onnx`, and `onnxruntime` as well.

  If you want *real* help with CoreML, I highly recommend [**Matthijs Holleman's**](https://github.com/hollance)
  [*“Core ML Survival Guide.”*](https://leanpub.com/coreml-survival-guide) Informative and well-written.
  Easy to read, as much as books on this subject can be.

## [coreml_help](https://mcsieber.github.io/coreml_help.html)

###  CoremlBrowser

Class and methods for inspection and "model surgery"
```
    show_nn         Show a summary of neural network layers by index or name
    connect_layers  Connect the output of one layer to the input of another
    delete_layers   Delete CoreML NN layers by *name*.
    get_nn          Get the layers object for a CoreML neural network
```
Once initialized, captures and keeps track of :

    cmb.spec        # The protobuf spec
    cmb.nn          # The neural network object
    cmb.layers      # The nn layers array
    cmb.layer_shapes  # The shape dictionary for this model
    cmb.layer_dict  # maps layer names to layer indexes
    cmb.layer_count # the count of nn layers
    cmb.shaper      # The shape inference object for this model

### Convenience Functions
```
    show_nn          Show a summary of nn (Function equivalent of `show_nn` method)
    show_head
    show_tail        Convenience functions  of  method `show_nn`
    get_rand_images  Return images (jpg and png) randomly sampled from child dirs.
```

## [pred_help](https://mcsieber.github.io/pred_help.html)

Python helper classes and functions to facilitate generation and display of predictions from CoreML, ONNX, and Torch models.

What's here:

Class **`Classifier`**  to invoke models, and collect and manage the resulting predictions.

Class **`Results`**  to browse and display results saved by Classifier

Model Execution and Calculation Functions:

- `norm_for_imagenet` Normalize using ImageNet values for mean and std dev.
- `pred_for_coreml`   Classify an image using a native CoreML model.
- `pred_for_onnx`     Classify an image using a native ONNX model.
- `pred_for_o2c`      Classify an image using a CoreML model converted from ONNX.
- `softmax`

The general purpose of the *pred* functions is

- On input, take a standard image - e.g. RGB, pixels values from 0-255 - and transform it to be acceptable as input
to the specific model. This might require normalizing the data, or rescaling to the interval 0.0 - 1.0, etc.

- On output, take the output from the model and transform it to an `ImagePrediction`

---------------------
