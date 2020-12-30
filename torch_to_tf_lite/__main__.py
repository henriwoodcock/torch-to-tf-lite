from pathlib import Path
import os
import logging
import sys

import click

import torch_to_tf_lite
#temp
os.environ['KMP_DUPLICATE_LIB_OK']='True'

@click.command()
@click.option('--torch_file', '-T', type=click.Path(exists=True), required=True)
@click.option('--onnx_file', '-O', type=click.Path())
@click.option('--keras_file', '-K', type=click.Path())
@click.option('--tf_file', '-TF', type=click.Path())
@click.option('--tflite_file', '-mu', type=click.Path(), required=True)
@click.option('--tflite_settings', '-mu_config', multiple=True)
@click.option('--prune_weights', '-P', type = float)
def main(torch_file, onnx_file, keras_file, tf_file, tflite_file,
        tflite_settings, prune_weights):
  '''
  args:
    - torch_file: path. required
    - onnx_file: path
    - keras_file: path
    - tf_file: path
    - tflite_file: path
    - tflite_settings: tuple
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
  loc = Path()
  if torch_file:
    torch_loc = loc / torch_loc
    torch_model = torch_to_tf_lite.load_torch(torch_loc)
  else:
    sys.exit(1)


  modelLoc = Path('models')
  dataLoc = Path('data')
  #load model from dl or locally
  model, accuracy = torch_to_tf_lite.load_model(load, modelLoc, dataLoc)
  print('model accuracy = ', accuracy)
  logging.info('accuracy on baseline model')
  logging.info(accuracy.numpy())

  #put in evaluate mode
  model.eval()
  # create an onnx model
  if convert_torch: torch_to_tf_lite.convert_torch_to_onnx(model, modelLoc)
  # convert onnx to tf frozen graph
  if convert_onnx:
    torch_to_tf_lite.convert_onnx_to_tf(modelLoc / 'resnet.onnx',
                                    modelLoc / 'resnet.pb')

  if convert_tflite:
    torch_to_tf_lite.convert_frozen_graph_to_tflite(modelLoc)

  if convert_keras:
    torch_to_tf_lite.convert_onnx_to_keras(modelLoc / 'resnet.onnx',
                                           modelLoc / 'keras')
    torch_to_tf_lite.check_torch_vs_keras(modelLoc / 'resnet.pth',
                                          modelLoc / 'keras')

  if prune_weights:
    acc = torch_to_tf_lite.prune_torch_weights(model, modelLoc, dataLoc, 0.25)
    print('accuracy after pruning = ', acc.numpy())
    logging.info('accuracy after pruning:')
    logging.info(acc.numpy())

  return None

if __name__ == '__main__':
  import sys
  import os
  sys.argv = ['', '-L','-T', '-O', '-K']
  main()
