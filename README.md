# Module coreml_help and pred_help

Python helper functions to facilitate working with CoreML and ONNX and converting from one to the other.

[Documentation](https://mcsieber.github.io/index.html)

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

Python helper classes and functions to facilitate generation and display
of predictions from CoreML, ONNX, and Torch models.

class **Classifier**  to invoke models, and collect and manage the resulting predictions.
class **Results**  to browse and display results saved by Classifier

Model Execution and Calculation Functions:
```
   norm_for_imagenet  Normalize using ImageNet values for mean and standard dev.
   pred_for_coreml    Run and show Predictions for a native CoreML model
   pred_for_onnx      Run and show Predictions for a native ONNX model
   pred_for_o2c       Run and show Predictions for a CoreML model converted from ONNX
   softmax
```
---------------------
