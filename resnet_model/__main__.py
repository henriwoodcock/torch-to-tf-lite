from pathlib import Path
import os
import logging

import click

import resnet_model
#temp
os.environ['KMP_DUPLICATE_LIB_OK']='True'

@click.command()
@click.option('--load', '-L', is_flag=True)
@click.option('--convert_torch', '-T', is_flag=True)
@click.option('--convert_onnx', '-O', is_flag=True)
@click.option('--convert_tflite', '-mu', is_flag = True)
@click.option('--prune_weights', '-PW', is_flag = True)
def main(load, convert_torch, convert_onnx, convert_tflite, prune_weights):
  modelLoc = Path('models')
  dataLoc = Path('data')
  #load model from dl or locally
  model, acc = resnet_model.load_model(load, modelLoc, dataLoc)
  print('model accuracy = ', acc)
  logging.info('accuracy on baseline model')
  logging.info(accuracy.numpy())

  #put in evaluate mode
  model.eval()
  # create an onnx model
  if convert_torch: resnet_model.convert_torch_to_onnx(model, modelLoc)
  # convert onnx to tf frozen graph
  if convert_onnx:
    resnet_model.convert_onnx_to_tf(modelLoc / 'resnet.onnx',
                                    modelLoc / 'resnet.pb')

  if convert_tflite:
    resnet_model.convert_frozen_graph_to_tflite(modelLoc)

  if prune_weights:
    acc = resnet_model.prune_torch_weights(model, modelLoc, dataLoc, 0.25)
    print('accuracy after pruning = ', acc.numpy())
    logging.info('accuracy after pruning:')
    logging.info(acc.numpy())

  return None

if __name__ == '__main__':
  import sys
  import os
  sys.argv = ['', '-PW']
  main()
