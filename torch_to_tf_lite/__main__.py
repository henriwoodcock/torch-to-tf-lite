from pathlib import Path
import os
import logging

import click

import torch_to_tf_lite
#temp
os.environ['KMP_DUPLICATE_LIB_OK']='True'

@click.command()
@click.option('--load', '-L', is_flag=True)
@click.option('--convert_torch', '-T', is_flag=True)
@click.option('--convert_onnx', '-O', is_flag=True)
@click.option('--convert_tflite', '-mu', is_flag = True)
@click.option('--convert_keras', '-K', is_flag=True)
@click.option('--prune_weights', '-PW', is_flag = True)
def main(load, convert_torch, convert_onnx, convert_tflite, convert_keras,
        prune_weights):
  modelLoc = Path('models')
  dataLoc = Path('data')
  #load model from dl or locally
  model, acc = torch_to_tf_lite.load_model(load, modelLoc, dataLoc)
  print('model accuracy = ', acc)
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
