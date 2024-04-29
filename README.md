[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/mUgQRbhV)
# Installation instructions

This should give an explanation on how to install the library in editable mode inside a Python venv, virtualenv or conda environment. 
Dependencies besides PyTorch should be installed when running the `pyproject.toml`. Document all the steps with the commands using screenshots. 
You only need to explain how to run this on your system (pool computer, private laptop etc.) with your specific OS (e.g. Windows, Ubuntu, MacOS).

## 1. Create venv, virtualenv or conda env
![alt text](https://github.com/Gambotch1/markdown-here/raw/master/images/1 "Create venv")

## 2. Activate venv or virtualenv
![alt text](https://github.com/Gambotch1/markdown-here/raw/master/images/2 "Activate venv")

## 3. Install project in editable mode
![alt text](https://github.com/Gambotch1/markdown-here/raw/master/images/3 "Install project in editable mode")

## 4. Install missing hardware-specific dependencies (PyTorch)
![alt text](https://github.com/Gambotch1/markdown-here/raw/master/images/4 "Install missing hardware-specific dependencies")

## 5. Git commiting with the command line interface

# Preperation - Downloading MNIST Dataset

When you are in the environment navigate to the project root folder. Inside the project root folder run:

`python src/mynet/utils.py data`

This utility function downloads the MNIST data and extracts the images into `data/MNIST/`.

# Running test locally

If you configured your environment correctly, you are able to check the test cases without pushing them to github on your local machine. In the project root folder run the following command:

`pytest -v`

If you are using an IDE and run a test case separately, make sure you set the working directory to the project root. Otherwise, the test will not work since it searches files in different directories.

# Visualization Results

Include the loss+accuracy plot here

Include the inference images on the test data here