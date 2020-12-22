from pathlib import Path
import os

import click

import resnet_model
#temp
os.environ['KMP_DUPLICATE_LIB_OK']='True'

@click.command()
@click.option('--load', '-L', is_flag=True)
@click.option('--convert_torch', '-T', is_flag=True)
def main(load, convert_torch):
  modelLoc = Path('models')
  #load model from dl or locally
  model = resnet_model.load_model(load, modelLoc)
  #put in evaluate mode
  model.eval()

  if convert_torch: resnet_model.convert_torch_to_onnx(model, modelLoc)

  return None

if __name__ == '__main__':
  main()
