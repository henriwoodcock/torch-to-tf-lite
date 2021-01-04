# Basic Example

This basic example uses a simple Pytorch model and converts it to a Tensorflow
Lite model.

## Setup
Make sure all the requirements and the package is installed.

## Usage
To run the example enter the following command into your terminal.

```
python basic_example.py
```

## Explanation of Code

The first couple of lines initiates a PyTorch model, this could be any model
you have created. In this example a simple linear regression model is created
with `y = ax + b`. The model is then put into evaluation mode.

```python
model = basic_example_model()
# save model
torch.save(model.state_dict(), Path('outputs/basic_model.pth'))
#put the model into eval model
model.eval()
```

The next four lines of code define the location for the outputs to be stored.
The tflite file is required but the keras file is optional if the user wants to
save a keras version of the model as well.

```python
# create a path object for the tflite output
tf_lite = Path('outputs/basic_model.tflite')
# create a path object for the keras output
keras = Path('outputs/basic_model_keras') #this is optional
```

The next couple of lines define the input and output shape of the model, this
is mainly used for verification of the conversion by comparing inputs and
outputs to make sure everything is as expected.

```python
#input shape and out shape
input_shape, output_shape = (1,), (1,)
```

After this all the inputs can be used in the torch_to_tf_lite converter to
create a `.tflite` file. Setting `change_ordering=False` means the same tensor
shape is used in TfLite and PyTorch. This option is incase of image data as
PyTorch and Tensorflow use different dimension orderings for image tensors.

```python
# run the converter
torch_to_tf_lite.torch_to_tf_lite(torch_model=model, tflite_file=tf_lite,
                                  input_shape=input_shape,
                                  output_shape=output_shape, keras_file=keras,
                                  change_ordering=False)
```

The rest of the code is used to compare the 3 models output to ensure the same
result was achieved in all 3.

Finally, the model size of the original Pytorch model and the tflite model is
compared. As can be seen the model is smaller for TfLite without any further
compression.
