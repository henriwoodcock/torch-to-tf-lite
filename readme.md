# Torch to Tensorflow Lite

PyTorch to Tensorflow Lite model converter.

## Contents
- [Installation](#installation)
- [Usage](#usage)
    - [Using Image Data](#using-image-data)
- [API Reference](#api-reference)
- [Examples](#examples)

## Installation

To install first clone the GitHub repository.

```
git clone https://github.com/henriwoodcock/torch-to-tf-lite.git
```

Change directory to the package

```
cd torch-to-tf-lite
```

Install Python (requires 3.8.6). If using `pyenv virtualenv` then use the
following:

```
pyenv virtualenv 3.8.6 torch_to_tf_lite
pyenv local torch_to_tf_lite
```

Install the required Python packages:

```
pip install -r requirements.txt
```

Finally install the package:

```
pip install .
```

## Usage

The converter will convert your PyTorch model to a Tensorflow Lite Model.

With your PyTorch Model

```python
import torch

class SimpleModel(nn.Module):
  """
  Module for Conv2d testing
  """
  def __init__(self, inp=10, out=1):
    super(SimpleModel, self).__init__()
    self.Layer1 = torch.nn.Linear(inp, out, bias=True)

  def forward(self, x):
    x = self.Layer1(x)
    return x

model = SimpleModel()
```

Next create the required inputs:

```python
from pathlib import Path

output_location = Path('my_smol_model.tflite')

input_shape = (10,)
output_shape = (1,)
```

Now pass all the inputs through the converter:

```python
from torch_to_tf_lite import torch_to_tf_lite

torch_to_tf_lite(torch_model=model, tflite_file=output_location,
                input_shape=input_shape, output_shape=output_shape)
```

### Using Image Data

When using Image data it is import to note the tensor shape. For example for
PyTorch models we tend to see tensors of shape `(3 x 224 x 224)` but for
Tensorflow models we tend to see `(224 x 224 x 3)`. To ensure this is converted
to the default for each software pass `change_ordering=True`

```python
from torch_to_tf_lite import torch_to_tf_lite

torch_to_tf_lite(torch_model=model, tflite_file=output_location,
                input_shape=input_shape, output_shape=output_shape,
                change_ordering=False)
```

## API Reference

The section explains the inputs for the `torch_to_tf_lite` function.

```python
from torch_to_tf_lite import torch_to_tf_lite

torch_to_tf_lite(torch_model, tflite_file, input_shape, output_shape,
                optimizations=None, convert_type='DYNAMIC',
                representative_data=None, onnx_file=None, keras_file=None,
                prune_weights=None, change_ordering=False)
```

Inputs:
- `torch_model`: (required). The PyTorch `torch.nn.Module` to be converted.
- `tflite_file`: (required). A `pathlib.Path` object pointing to the location to
export the model.
- `input_shape`: (required). A `tuple` represented the input tensor shape.
(No batch size).
- `output_shape`: (required). A `tuple` represented the output tensor shape.
(No batch size).
- `optimizations`: (optional). `tf.lite.Optimize` object. e.g.
`tf.lite.Optimize.DEFAULT`
- `convert_type`: (optional). A `str` to choose conversion type, the options are:
    - `"DYNAMIC"` (default)
    - `"INTEGER"`
    - `"FLOAT16"`
    More information available [here](https://www.tensorflow.org/lite/performance/model_optimization)
- `representative_data`: (optional). A generator to draw samples from a dataset.
Used in Tensorflow lite to help with optimization. More information
[here](https://www.tensorflow.org/lite/performance/model_optimization).
- `onnx_file`: (optional). Pass a `pathlib.Path` object to save the interim onnx
model.
- `keras_file`: (optional). Pass a `pathlib.Path` object to save the interim Keras
model.
- `prune_weights`: (optional). Pass a `float` between 0 and 1 to prune weights.
- `change_ordering`: (`default=False`). Change to `True` to change shape of
images from Torch format to Tensorflow format. (Currently only works for
`3x224x224` images).

## Examples

There are several examples available in the examples folder and more to be
added. Go to them [here](examples).

#### Quick fix to an issue

```
brew install libomp
```
