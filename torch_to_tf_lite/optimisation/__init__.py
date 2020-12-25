import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
import numpy as np
import scipy.stats
import tensorflow as tf

from pathlib import Path
import collections
import os

def load_data(data_dir, input_size, batch_size):
  # Data augmentation and normalization for training
  # Just normalization for validation
  data_transforms = {
    'train': transforms.Compose([
      transforms.RandomResizedCrop(input_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
      transforms.Resize(input_size),
      transforms.CenterCrop(input_size),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
  }
  print("Initializing Datasets and Dataloaders...")

  # Create training and validation datasets
  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                    data_transforms[x]) for x in ['train', 'val']}
  # Create training and validation dataloaders
  dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                          batch_size=batch_size, shuffle=True, num_workers=4
                          ) for x in ['train', 'val']}

  # Detect if we have a GPU available
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  return dataloaders_dict, device

def load_tf_data(data_dir, input_size, batch_size):
  val = load_data(data_dir, input_size, batch_size)[0]['val']
  batches = []
  labels_batch = []

  for inputs, labels in val:
    batches.append(inputs.numpy())
    labels_batch.append(labels.numpy())

  return batches, labels_batch

def evaluate_tensorflow_model(model_loc, batches, labels_batch):
  model = tf.saved_model.load(model_loc.as_posix())
  infer = model.signatures['serving_default']
  test = batches[0]
  test = test[0]
  test.shape = (1,3,224,224)
  test = tf.constant(test)
  infer(test)

  return None

def test_torch_accuracy(model, data_path):
  dataloaders_dict, device = load_data(data_path, 224, 32)
  running_corrects = 0.0

  for inputs, labels in dataloaders_dict['val']:
    with torch.no_grad():
      inputs = inputs.to(device)
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      running_corrects += torch.sum(preds == labels.data)

  accuracy = running_corrects.double() / len(dataloaders_dict['val'].dataset)

  return accuracy

def prune_weights(model, model_path, data_path, k=0.25):
  '''k = prune_percentage'''
  # Get all the weights
  weights = model.state_dict()
  # Get keys to access model weights
  layers = list(model.state_dict())
  ranks = {}
  pruned_weights = []
  # For each layer except the output one
  for l in layers[:-1]:
    # Get weights for each layer and conver to numpy
    data = weights[l]
    w = np.array(data)
    # Rank the weights element wise and reshape rank elements as the model weights
    ranks[l]=(scipy.stats.rankdata(np.abs(w), method='dense') - 1).astype(int).reshape(w.shape)
    # Get the threshold value based on the value of k(prune percentage)
    lower_bound_rank = np.ceil(np.max(ranks[l]) * k).astype(int)
    # Assign rank elements to 0 that are less than or equal to the threshold and 1 to those that are above.
    ranks[l][ranks[l] <= lower_bound_rank] = 0
    ranks[l][ranks[l] > lower_bound_rank] = 1
    #ignore batchnorm layers for now
    if ('bn' in l) or ('running' in l) or ('num_batches' in l):
      w = w * 1
    # Multiply weights array with ranks to zero out the lower ranked weights
    else:
      w = w * ranks[l]
    # Assign the updated weights as tensor to data and append to the pruned_weights list
    if isinstance(w, np.int64):
      w = np.array(w)
    data[...] = torch.from_numpy(w)
    pruned_weights.append(data)
  # Append the last layer weights as it is
  pruned_weights.append(weights[layers[-1]])
  # Update the model weights with all the updated weights
  new_state_dict = collections.OrderedDict()
  for l, pw in zip(layers, pruned_weights):
    new_state_dict[l] = pw
  #model.state_dict = new_state_dict
  k_num = int(k*100)
  model_name = f'resnet_{k_num}_perc_weights.pth'
  model.load_state_dict(new_state_dict)
  torch.save(model.state_dict(), model_path / model_name)
  # append the test accuracy to accuracies_wp
  #accuracies_wp.append(test_accuracy(model, testloader, criterion))
  return model


if __name__ == '__main__':
  from pathlib import Path
  val = load_tf_data(Path('data'), 224, 32)
  ins, labels = val
  evaluate_tensorflow_model(Path('models') / 'resnet.pb', ins, labels)
