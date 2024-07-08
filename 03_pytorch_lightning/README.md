# Project Overview

This project is a practical introduction to deep learning using PyTorch and PyTorch Lightning. It is designed for graduate students who are new to these topics. The project involves training a model to make predictions based on input sequences, a common task in many deep learning applications.


## Key Files
- `script_constant_velocity.py` and script_MLP.py: These are the scripts that contain the code for the experiments. They are used to train the model and evaluate its performance. The scripts use PyTorch and PyTorch Lightning to define the model, process the data, and train the model.

- `experiment_setup.py`: This script sets up the experiment by defining training and testing loops, loading the data, and running the training process. 


## Key Concepts

- **Deep Learning**: This is a subfield of machine learning that focuses on algorithms inspired by the structure and function of the brain called artificial neural networks.

- **PyTorch**: This is an open-source machine learning library based on the Torch library. It's used for applications such as computer vision and natural language processing. It is primarily developed by Facebook's AI Research lab.

- **PyTorch Lightning**: This is a lightweight PyTorch wrapper for high-performance AI research. It organizes your existing PyTorch code and provides advanced features with minimal overhead.

- **Sequence Prediction**: This is a common task in many deep learning applications. It involves using a model to predict the next value(s) in a sequence based on the previous values.


## Getting Started

To get started with this project, you will need to have Python installed on your machine, preferably Python 3.6 or later. You will also need the following Python packages: PyTorch and PyTorch Lightning.

You can install PyTorch and PyTorch Lightning using pip:
```bash
pip install -r requirements.txt
```
Once you have Python and the necessary packages installed, you can run the scripts in this project.
Notice: There are two scripts in this project: `script_constant_velocity.py` and `script_MLP.py`. You can run either of 
them to train the model and evaluate its performance. To study what is going on, put a debug breakpoint in the respective
lines 7 and step through the code. You can "step over" or "step into" the code to see what functions are doing one level below.


## Learning Resources

If you're new to these topics, here are a few resources to get you started:

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/en/latest/)

If you encounter a function which is new to you, you should look it up in the PyTorch documentation. Since you might be new
to everything, this tips means that you might have to google every line. Let Github Copilot or ChatGPT help you by letting
it explain the code to you.
Remember, the best way to learn is by doing. Don't be afraid to make changes to the scripts and see what happens. Happy coding!