import torch
import torchvision

def load_model():
  return torchvision.models.resnet18(pretrained=True)

def convert_torch_to_onnx():
  return None
