import torch_to_tf_lite #package

from pathlib import Path #built in

import torch #3rd party
import tensorflow as tf
import numpy as np

class SimpleFFN(torch.nn.Module):
  def __init__(self):
    """A simple feed forward network with 2 fully connected hidden layers
    """
    super().__init__()
    self.l1 = torch.nn.Linear(10, 5)
    self.l2 = torch.nn.Linear(5, 1)

  def forward(self, x):
    o = self.l1(x)
    o = torch.nn.functional.relu(o, inplace=False)
    o = self.l2(o)

    return o

def tflite_inference(tflite_path, data, data_type):
  '''perform inference on a tensorflow lite model

  args:
    - tflite_path = pathlib.Path
    - data = np.array
    - data_type = np.dtype

  returns:
    - output_data: np.array
  '''
  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path=tflite_path.as_posix())
  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Adjust graph input to handle batch tensor
  interpreter.resize_tensor_input(input_details[0]['index'], data.shape)

  # Adjust output #1 in graph to handle batch tensor
  interpreter.resize_tensor_input(output_details[0]['index'], (data.shape[0], 1))

  interpreter.allocate_tensors()

  input_data = np.array(data, dtype=data_type)
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  output_data = interpreter.get_tensor(output_details[0]['index'])

  return output_data


if __name__ == '__main__':
  #initiate model from above
  model = SimpleFFN()
  # save model
  torch.save(model.state_dict(), Path('outputs/torch_model.pth'))
  #put into eval mode
  model.eval()
  # create a path object for the tflite output
  tf_lite = Path('outputs/basic_tflite_model.tflite')
  #input shape and out shape
  input_shape, output_shape = (10,), (10,)
  # run the converter
  torch_to_tf_lite.torch_to_tf_lite(torch_model=model, tflite_file=tf_lite,
                                   input_shape=input_shape,
                                   output_shape=output_shape,
                                   change_ordering=False)

  #optimizations
  optimizations = tf.lite.Optimize.DEFAULT
  tf_lite2 = Path('outputs/dynamic_range_tflite_model.tflite')
  #with the default tf lite optimization (dynamic range)
  torch_to_tf_lite.torch_to_tf_lite(torch_model=model,
                                    tflite_file=tf_lite2,
                                    input_shape=input_shape,
                                    output_shape=output_shape,
                                    optimizations=optimizations,
                                    change_ordering=False)

  #full quantization uses a representative dataset
  #for this example just random data is used
  dataset = np.random.rand(100,10)
  def representative_dataset():
    for i in range(100):
      data = dataset[i]
      yield [data.astype(np.float32)]

  tf_lite3 = Path('outputs/float_fall_back_tflite_model.tflite')
  # add the generator to the function
  torch_to_tf_lite.torch_to_tf_lite(torch_model=model,
                                    tflite_file=tf_lite3,
                                    input_shape=input_shape,
                                    output_shape=output_shape,
                                    optimizations=optimizations,
                                    representative_data=representative_dataset,
                                    change_ordering=False)

  # full integer
  tf_lite4 = Path('outputs/full_integer_tflite_model.tflite')
  # change the convert type
  torch_to_tf_lite.torch_to_tf_lite(torch_model=model,
                                    tflite_file=tf_lite4,
                                    input_shape=input_shape,
                                    output_shape=output_shape,
                                    optimizations=optimizations,
                                    convert_type='INTEGER',
                                    representative_data=representative_dataset,
                                    change_ordering=False)

  #float 16 optimiszation
  tf_lite5 = Path('outputs/float16_tflite_model.tflite')
  # change the convert type
  torch_to_tf_lite.torch_to_tf_lite(torch_model=model,
                                    tflite_file=tf_lite5,
                                    input_shape=input_shape,
                                    output_shape=output_shape,
                                    optimizations=optimizations,
                                    convert_type='FLOAT16',
                                    representative_data=representative_dataset,
                                    change_ordering=False)

  # now to compare everything first generate a dataset

  test_data = np.random.rand(100, 10)
  torch_tensor = torch.Tensor(test_data)

  with torch.no_grad():
    torch_output = model(torch_tensor)

  # get output from original tflite
  tflite_out1 = tflite_inference(tf_lite, test_data, np.float32)
  tflite_out2 = tflite_inference(tf_lite2, test_data, np.float32)
  tflite_out3 = tflite_inference(tf_lite3, test_data, np.float32)
  tflite_out4 = tflite_inference(tf_lite4, test_data, np.int8)
  tflite_out5 = tflite_inference(tf_lite5, test_data, np.float32)

  print('Average Errors between tflite and torch model')
  print('-'*10)
  error = (torch_output - tflite_out1).mean()
  print(f'Average Error for Basic TFLite = {error}')
  error = (torch_output - tflite_out2).mean()
  print(f'Average Error for Dynamic Range = {error}')
  error = (torch_output - tflite_out3).mean()
  print(f'Average Error for float fall back = {error}')
  error = (torch_output - tflite_out4).mean()
  print(f'Average Error for full integer = {error}')
  error = (torch_output - tflite_out5).mean()
  print(f'Average Error for float16 = {error}')

  print('Model Sizes and Difference from Torch model size')
  print('-'*10)
  torch_model = Path('outputs/torch_model.pth').stat().st_size
  print('Torch Model Size', torch_model)
  tflite_model1 = tf_lite.stat().st_size
  error = (torch_model - tflite_model1)
  print(f'Basic TFLite model size = {tflite_model1}')
  print(f'Difference from Torch model size = {error}')
  tflite_model2 = tf_lite2.stat().st_size
  error = (torch_model - tflite_model2)
  print(f'Dynamic Range model size = {tflite_model2}')
  print(f'Difference from Torch model size = {error}')
  tflite_model3 = tf_lite3.stat().st_size
  error = (torch_model - tflite_model3)
  print(f'float fall back model size = {tflite_model3}')
  print(f'Difference from Torch model size = {error}')
  tflite_model4 = tf_lite4.stat().st_size
  error = (torch_model - tflite_model4)
  print(f'full integer model size = {tflite_model4}')
  print(f'Difference from Torch model size = {error}')
  tflite_model5 = tf_lite5.stat().st_size
  error = (torch_model - tflite_model5)
  print(f'float16 model size = {tflite_model5}')
  print(f'Difference from Torch model size = {error}')
