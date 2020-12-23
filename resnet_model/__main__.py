from pathlib import Path
import os

import click

import resnet_model
#temp
os.environ['KMP_DUPLICATE_LIB_OK']='True'

@click.command()
@click.option('--load', '-L', is_flag=True)
@click.option('--convert_torch', '-T', is_flag=True)
@click.option('--convert_onnx', '-O', is_flag=True)
@click.option('--convert_tflite', '-mu', is_flag = True)
@click.option('--convert_tflite2', '-mu2', is_flag = True)
def main(load, convert_torch, convert_onnx, convert_tflite, convert_tflite2):
  modelLoc = Path('models')
  #load model from dl or locally
  model = resnet_model.load_model(load, modelLoc)
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

  return None

if __name__ == '__main__':
  main()
