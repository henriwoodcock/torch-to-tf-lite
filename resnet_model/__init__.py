import torch
import torchvision
import onnx
import onnxruntime
from onnx_tf.backend import prepare
import numpy as np
import tensorflow as tf
from . import fine_tune

def load_model(load, model_path, dataLoc):
  '''load the pytorch model ready to be converted.

  args:
    - load: bool
    - model_path: pathlib.Path
    - dataLoc: pathlib.Path

  returns:
    - model: torch.nn.Module
  '''
  if load:
    model = torchvision.models.resnet18(pretrained=True)
    model, _ = fine_tune.train.feature_extractor(model, dataLoc)
    torch.save(model.state_dict(), model_path / 'resnet.pth')

  else:
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 10, bias = True)
    model.load_state_dict(torch.load((model_path / 'resnet.pth').as_posix()))

  return model

def create_rand_tens():
  '''create a random tensor using the torch.randn function

  args:

  returns:
    - torch.tensor
  '''
  return torch.randn(1, 3, 224, 224, requires_grad=True)

def to_numpy(tensor):
  '''convert a torch tensor to a numpy array

  args:
    - torch.tensor

  returns:
    - numpy.array
  '''
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def convert_torch_to_onnx(model, onnx_path):
  '''convert a torch model in an onnx model. This function will run
  onnx.checker.check_model and assert the output of both models from the same
  input are close through assertion

  args:
    - model: torch.Module (torch model)
    - onnx_path: pathlib.Path
  '''
  #generate random tensor to use
  rand_tens = torch.autograd.Variable(create_rand_tens())
  #get torch model output
  torch_out = model(rand_tens)
  torch.onnx.export(
    model=model,
    args=rand_tens,
    f=onnx_path / 'resnet.onnx',
  )

  onnx_model = onnx.load((onnx_path / 'resnet.onnx').as_posix())
  onnx.checker.check_model(onnx_model)

  ort_session = onnxruntime.InferenceSession((onnx_path / 'resnet.onnx').as_posix())
  ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(rand_tens)}
  ort_outs = ort_session.run(None, ort_inputs)
  # compare ONNX Runtime and PyTorch results
  np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03,
                              atol=1e-05)
  print("Exported model has been tested with ONNXRuntime, and the result looks"\
        " good!")

  return None

def convert_frozen_graph_to_tflite(modelLoc):
  '''convert a tensorflow frozen graph (.pb folder) into a tf life model.

  This saves the most basic version of a tflite model

  args:
    - modelLoc: pathlib.Path
  '''
  converter = tf.lite.TFLiteConverter.from_saved_model((modelLoc / 'resnet.pb').as_posix())
  #converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tf_lite_model = converter.convert()
  open(modelLoc / 'resnet.tflite', 'wb').write(tf_lite_model)

  return None

def convert_onnx_to_tf(onnx_path, tf_path):
  '''convert an onnx model to a tf model.

  args:
    - onnx_path: pathlib.Path
    - tf_path: pathlib.Path
  '''
  onnx_model = onnx.load(onnx_path.as_posix())  # load onnx model
  tf_rep = prepare(onnx_model)  # creating TensorflowRep object
  tf_rep.export_graph(tf_path.as_posix())

  return None
