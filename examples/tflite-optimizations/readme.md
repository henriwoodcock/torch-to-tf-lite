# TFLite Optimizations

This example goes through different optimizations which are available with the
Tensorflow Lite converter. There are more available in the Torch to TF Lite
converter however this is to show the ones available in tf lite.

The script goes through the optimizations explained in
[https://www.tensorflow.org/lite/performance/model_optimization](https://www.tensorflow.org/lite/performance/model_optimization)

## Contents
- [Setup](#setup)
- [Usage](#usage)
    - [Ouputs](#outputs)
- [Explanation of Code](#explanation-of-code)

## Setup

Make sure all the requirements and the package is installed.

## Usage
To run the example enter the following command into your terminal.

```
python tflite_optimizations.py
```

### Outputs

Running the script will output several models in the `outputs` folder as well
as a few prints comparing torch and tensorflow lite against random tensors.

For example

```python
----------
Average Error for Basic TFLite = -2.98023217215615e-10
Average Error for Dynamic Range = -2.98023217215615e-10
Average Error for float fall back = 0.0014558422844856977
Average Error for full integer = -34.48484802246094
Average Error for float16 = -3.7908553167653736e-06
Model Sizes and Difference from Torch model size
----------
Torch Model Size 1943
Basic TFLite model size = 1356
Difference from Torch model size = 587
Dynamic Range model size = 1456
Difference from Torch model size = 487
float fall back model size = 1936
Difference from Torch model size = 7
full integer model size = 1632
Difference from Torch model size = 311
float16 model size = 1840
Difference from Torch model size = 103
```

## Explanation of Code

The first couple of lines initiates a PyTorch model, this could be any model
you have created. In this example a simple two-layered feed forward network
is created with `y = layer2(relu(layer1(x))`.
The model is then put into evaluation mode.

```python
  #initiate model from above
  model = SimpleFFN()
  # save model
  torch.save(model.state_dict(), Path('outputs/torch_model.pth'))
  #put into eval mode
  model.eval()
```

This first model is skipped as it is the same as can be found in the
basic-example example. The second model uses dynamic range quanitzation, this
is done by using the tensorflow default optimize setting.

```python
  optimizations = tf.lite.Optimize.DEFAULT
  tf_lite2 = Path('outputs/dynamic_range_tflite_model.tflite')
  #with the default tf lite optimization (dynamic range)
  torch_to_tf_lite.torch_to_tf_lite(torch_model=model,
                                    tflite_file=tf_lite2,
                                    input_shape=input_shape,
                                    output_shape=output_shape,
                                    optimizations=optimizations,
                                    change_ordering=False)
```

For the next few quantizations a representitive dataset is required. This is
done by creating a generator for tensorflow lite. Because this is an example
a random array is generated. To ensure each model uses the same array the
generator draws samples from a pre-drawn array

```python
#full quantization uses a representative dataset
#for this example just random data is used
dataset = np.random.rand(100,10)
def representative_dataset():
  for i in range(100):
    data = dataset[i]
    yield [data.astype(np.float32)]
```

For float-fall back quantization, the representative dataset is passed to the
function

```python
tf_lite3 = Path('outputs/float_fall_back_tflite_model.tflite')
# add the generator to the function
torch_to_tf_lite.torch_to_tf_lite(torch_model=model,
                                  tflite_file=tf_lite3,
                                  input_shape=input_shape,
                                  output_shape=output_shape,
                                  optimizations=optimizations,
                                  representative_data=representative_dataset,                                    change_ordering=False)
```

The next two quantization methods are done by passing a `convert_type` to the
function.

For `float16` conversion pass `convert_type = FLOAT16` and for integer pass
`convert_type = INTEGER`. For example with integer:

```python
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
```

More information about these types of quantization are available on the
tensorflow website.

https://www.tensorflow.org/lite/performance/model_optimization

The rest of the script generates a random tensor (with 100 samples) and
compares the average difference from the Torch model and the tensorflow lite
models. The file sizes are also compared.

To perform inference on the tflite models a function is created for ease. This
can be seen below.

```python
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
```

This function imports the tensorflow lite model and adjusts the input and output
 tensor sizes to account for the batch size passed through. This allows for
 inference of multiple samples at a time.
