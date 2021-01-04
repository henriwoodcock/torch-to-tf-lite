import torch_to_tf_lite #package

from pathlib import Path #built in

import torch #3rd party
import tensorflow as tf
import numpy as np

class basic_example_model(torch.nn.Module):
    def __init__(self):
        """A simple linear model with model(x) = ax + b
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.a*x + self.b

if __name__ == '__main__':
  #initiate model from above
  model = basic_example_model()
  # save model
  torch.save(model.state_dict(), Path('outputs/basic_model.pth'))
  #put the model into eval model
  model.eval()
  # create a path object for the tflite output
  tf_lite = Path('outputs/basic_model.tflite')
  # create a path object for the keras output
  keras = Path('outputs/basic_model_keras') #this is optional
  #input shape and out shape
  input_shape, output_shape = (1,), (1,)
  # run the converter
  torch_to_tf_lite.torch_to_tf_lite(torch_model=model, tflite_file=tf_lite,
                                   input_shape=input_shape,
                                   output_shape=output_shape, keras_file=keras,
                                   change_ordering=False)

  ## testing the conversion ##

  # import the keras model created
  keras_model = tf.keras.models.load_model(keras)
  #generate a random numpy tensor
  torch_tensor = torch_to_tf_lite.utils.create_rand_tens(input_shape)
  numpy_tensor = torch_to_tf_lite.utils.to_numpy(torch_tensor)
  #torch output
  with torch.no_grad():
    torch_output = model(torch_tensor)
  torch_output = torch_output.data.numpy()

  keras_output = keras_model.predict(numpy_tensor)
  error = torch_output - keras_output
  print(f'Error from PyTorch to Keras Conversion', error.max())

  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path=tf_lite.as_posix())
  interpreter.allocate_tensors()
  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  input_data = np.array(numpy_tensor, dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  output_data = interpreter.get_tensor(output_details[0]['index'])
  error = torch_output - output_data
  print(f'Error from PyTorch to TfLite Conversion', error.max())

  print('PyTorch model size =', Path('outputs/basic_model.pth').stat().st_size)
  print('TfLite model size =', tf_lite.stat().st_size)
  print('Difference =',
        Path('outputs/basic_model.pth').stat().st_size - tf_lite.stat().st_size,
        'bytes')
