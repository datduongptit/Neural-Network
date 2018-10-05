
![Alt Text](https://github.com/jzisheng/Neural-Network/blob/master/images/training.gif)

# Numpy Neural Network

This repository contains two neural networks that demonstrate feedforward networks and error backpropagation. This project was created such that I could gain a deeper understanding of the fundamentals of neural networks. The single hidden layer neural network solving XOR visualizes the inputs and network weights in Matplotlib. The output of this graph is shown in the GIF above.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Setup and Installing

The network only needs Numpy installed. You either need to have numpy installed on your machine, or set up a new virtual environment with numpy. What follows are instructions setting up a virtual environment, and installing numpy. 

First make sure conda is installed

```
$ conda -V
conda 3.7.0
```

And make sure conda is up to date

```
$ conda update conda
```

Then create the environment for this neural network. You can name it what you wish, I chose ```np```. Then activate the environment

```
$ conda create -n np python=3.5 anaconda
```

## Running the tests
To train the neural network on `or`, run

```
python3 ornn.py
```

And to see the neural network solve for XOR, run
```
python3 xornn.py
```

## Built With

* [Numpy](http://www.numpy.org/) - Fundamental package for scientific computation

## Authors
* **Zisheng Jason Chang** 

