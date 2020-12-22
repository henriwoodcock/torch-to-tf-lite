import torch
import torchvision
import onnx
import onnxruntime
import numpy as np

def load_model(load, model_path):

  if load:
    model = torchvision.models.resnet18(pretrained=True)
    torch.save(model, model_path / 'resnet.pth')

  else:
    model = torch.load(model_path / 'resnet.pth')

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
  rand_tens = create_rand_tens()
  #get torch model output
  torch_out = model(rand_tens)
  torch.onnx.export(
    model=model,
    args=rand_tens,
    f=onnx_path / 'resnet.onnx',
    verbose=False,
    export_params=True,
    do_constant_folding=False,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                  'output' : {0 : 'batch_size'}}
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









