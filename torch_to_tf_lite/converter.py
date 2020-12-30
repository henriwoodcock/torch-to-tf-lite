from pathlib import Path
import os
import sys
#temp
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def torch_to_tflite(torch_model, tflite_file, input_shape, output_shape,
                    tflite_settings=None, onnx_file=None, keras_file=None,
                    tf_file=None, prune_weights=None):
  '''
  args:
    - torch_model: torch.nn.Module
    - tflite_file: path
    - input_shape: tuple e.g. (3,224,224) (without the batch size so not
                                          (1,3,224,224))
    - output_shape: tuple
    - tflite_settings: tuple (optional)
    - onnx_file: path (optional)
    - keras_file: path (optional)
    - tf_file: path (optional)
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

  if not isinstance(torch_model, torch.nn.Module):
    raise TypeError('torch_model is required to be a torch.nn.Module')
  if not isinstance(tflite_file, pathlib.Path):
    raise TypeError('tflite_file is require to be a pathlib.Path')

  if not onnx_file: onnx_file = pathlib.Path('temp.onnx')
  torch_to_tflite.convert_torch_to_onnx(torch_model, onnx_file, input_shape,
                                        output_shape)


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
