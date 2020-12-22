from pathlib import Path

import click

import resnet_model

@click.command()
@click.option('--load', '-L', is_flag=True)
def main(load):
  modelLoc = Path('models')
  #load model from dl or locally
  model = resnet_model.load_model(load, modelLoc)
  #put in evaluate mode
  model.eval()

  if convert_torch: resnet_model.convert_torch_to_onnx(model, modelLoc)

  return None

if __name__ == '__main__':
  main()
