Module coreml_help
==================
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

The class "CoremlBrowser" methods for inspection and "model surgery"
```
    show_nn         Show a summary of neural network layers by index or name
    connect_layers  Connect the output of one layer to the input of another
    delete_layers   Delete CoreML NN layers by *name*.
    get_nn          Get the layers object for a CoreML neural network
```
Convenience Functions
```
    show_nn          Show a summary of nn (Function equivalent of `show_nn` method)
    show_head
    show_tail        Convenience functions  of  method `show_nn`
    get_rand_images  Return images (jpg and png) randomly sampled from child dirs.
```
Model Execution and Calculation Functions
```
     norm_for_imagenet  Normalize using ImageNet values for mean and standard dev.
     pred_for_coreml    Run and show Predictions for a native CoreML model
     pred_for_onnx      Run and show Predictions for a native ONNX model
     pred_for_o2c       Run and show Predictions for a CoreML model converted from ONNX
     softmax
```
Use

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

I wrote these as a learning exercise. Feedback welcome.
Most of this is based on the work of others,  but there can be no question that any
bugs, errors, misstatements,and, especially, inept code constructs, are entirely mine.

---------------------