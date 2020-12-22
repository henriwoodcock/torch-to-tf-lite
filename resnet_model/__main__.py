import click

import resnet_model

@click.command()
@click.option('--load', '-L', is_flag=True)
def main(load):

  if load: model = resnet_model.load_model()

  return None

if __name__ == '__main__':
  main()
