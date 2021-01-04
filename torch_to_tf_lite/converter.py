import pathlib
import os
import sys

import torch

from . import utils
#temp
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def torch_to_tf_lite(torch_model, tflite_file, input_shape, output_shape,
                    optimizations=None, convert_type='DYNAMIC', onnx_file=None,
                    keras_file=None, prune_weights=None, change_ordering=True):
  '''
  args:
    - torch_model: torch.nn.Module
    - tflite_file: path
    - input_shape: tuple e.g. (3,224,224) (without the batch size so not
                                          (1,3,224,224))
    - output_shape: tuple
    - optimizations: tf.lite.Optimize e.g. tf.lite.Optimize.DEFAULT
    - onnx_file: path (optional)
    - keras_file: path (optional)
    - prune_weights: float

  Usage:
  -------
  - torch_file is required and is the location of the PyTorch file to be loaded.
  - tflite_file is required and is the location for the tflite model to be
  exported.

  - if onnx_file is provided then an onnx file is exported and checked
  - if keras_file is provided then a keras file is exported and checked
  - if tf_file is provided then a tensorflow graph file is exported and checked
  - if tflite_settings is provided these settings are used for the tflite
    model export otherwise no optimisation settings are used
  - if prune_weights is provided then the percentage provided is pruned.
  '''
  #put torch model in eval mode
  torch_model.eval()

  if not isinstance(torch_model, torch.nn.Module):
    raise TypeError('torch_model is required to be a torch.nn.Module')
  if not isinstance(tflite_file, pathlib.Path):
    raise TypeError('tflite_file is require to be a pathlib.Path')

  if prune_weights:
    print('Pruning weights in PyTorch before conversion.')
    torch_model = utils.prune_torch_weights(torch_model, prune_weights)

  if not onnx_file: onnx_file = pathlib.Path('temp.onnx')
  #convert torch model to an onnx model
  utils.convert_torch_to_onnx(torch_model, onnx_file, input_shape)
  keras_model = utils.convert_onnx_to_keras(onnx_file, keras_file, torch_model,
                                            input_shape, change_ordering)
  #if file name is temp it can now be deleted.
  if onnx_file.stem == 'temp':
    onnx_file.unlink()

  # add weight clustering here

  #convert keras to tf lite
  tflite_model = utils.convert_keras_to_tflite(keras_model,
                                                        optimizations,
                                                        convert_type)
  # save tf lite model
  with open(tflite_file.as_posix(), 'wb') as f:
    f.write(tflite_model)

  return None

if __name__ == '__main__':
  import torch
  import torchvision
  from pathlib import Path

  model = torchvision.models.resnet18(pretrained=True)
  tf_lite = Path('tf_lite_model.tflite')
  input_shape = (3, 224, 224)
  output_shape = (1, 1000)

  torch_to_tflite(model, tf_lite, input_shape, output_shape)

'''
import torch_to_tf_lite;import torch, torchvision;from pathlib import Path
model = torchvision.models.resnet18(pretrained=True);tf_lite = Path('tf_lite_model.tflite');input_shape = (3, 224, 224);output_shape = (1, 1000)
torch_to_tf_lite.torch_to_tflite(model, tf_lite, input_shape, output_shape)
'''
