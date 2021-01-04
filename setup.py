import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch_to_tf_lite", # Replace with your own username
    version="0.0.1",
    author="Henri Woodcock",
    author_email="henriwoodcock@gmail.com",
    description="A package to convert PyTorch models to Tensorflow Lite.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/henriwoodcock/torch-to-tf-lite",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.6',
)
