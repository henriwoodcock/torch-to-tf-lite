import torch
import torchvision
import onnx
import onnxruntime
from onnx_tf.backend import prepare
import numpy as np
import tensorflow as tf

def load_model(load, model_path):

  if load:
    model = torchvision.models.resnet18(pretrained=True)
    torch.save(model.state_dict(), model_path / 'resnet.pth')

  else:
    model = torchvision.models.resnet18(pretrained=False)
    model.load_state_dict(torch.load((model_path / 'resnet.pth').as_posix()))

  return model

def create_rand_tens():
  return torch.randn(1, 3, 224, 224, requires_grad=True)

def to_numpy(tensor):
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def convert_torch_to_onnx(model, onnx_path):
  '''convert a torch model in an onnx model

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

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def convert_frozen_graph_to_tflite(modelLoc):
  converter = tf.lite.TFLiteConverter.from_saved_model((modelLoc / 'resnet.pb').as_posix())
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tf_lite_model = converter.convert()
  open(modelLoc / 'resnet.tflite', 'wb').write(tf_lite_model)

  return None

def convert_onnx_to_tf(onnx_path, tf_path):
  onnx_model = onnx.load(onnx_path.as_posix())  # load onnx model
  tf_rep = prepare(onnx_model)  # creating TensorflowRep object
  tf_rep.export_graph(tf_path.as_posix())

  return None
