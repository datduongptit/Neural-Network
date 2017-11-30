# Neural Network Basics

This repository contains two neural networks that demonstrate feedforward networks and backpropagation.

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

