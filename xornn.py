

import numpy as np
import matplotlib.pyplot as plt

class Neural_Network(object):
    '''
    Neural Network Class:
    Implementation of a Neural Network with two hidden layers with 5 nodes each
    '''
    # Set this value to true for print statements
    debug = True
    def __init__(self,MAXITER=1000):
        # Sets up neural network layers
        np.random.seed(9)
        self.inputLayerSize = 2      # No. of inputs the neural network takes in
        self.hiddenLayerSize1 = 100    # No. of node in hidden layers
        self.hiddenLayerSize2 = 100
        self.outputLayerSize = 1     # No. of outputs of NN
        self.learningRate = 0.01      # Learning rate
        
        self.MAXITER = MAXITER

        # Initialize weights to random values
        self.W1 = np.random.randn(self.inputLayerSize+1,self.hiddenLayerSize1)
        self.W2 = np.random.randn(self.hiddenLayerSize1,self.hiddenLayerSize2)
        self.W3 = np.random.randn(self.hiddenLayerSize2,self.outputLayerSize)

    # Feedforward Function
    def forward(self, X):
        # Propogate inputs through network
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.sigmoid(self.z2)
        
        self.z3 = np.dot(self.a2,self.W2)
        self.a3 = self.sigmoid(self.z3)
        
        self.z4 = np.dot(self.a3,self.W3)
        self.yHat = self.sigmoid(self.z4)
        return self.yHat
    
    def predict(self,X):
        # Propogate inputs through network
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.sigmoid(self.z2)
        
        self.z3 = np.dot(self.a2,self.W2)
        self.a3 = self.sigmoid(self.z3)
        
        self.z4 = np.dot(self.a3,self.W3)
        self.yHat = self.sigmoid(self.z4)
        yThresh = self.yHat
        yThresh[yThresh >= 0.5] = 1
        yThresh[yThresh < 0.5] = 0
        return self.yHat

    def predict_proba(self, X):
        # Propogate inputs through network
        return self.predict(self,X)

    # Activation Function
    def sigmoid(self,z):
        # Apply sigmoid activation function
        return 1/(1+np.exp(-z))

    # Derivative of sigmoid prime function
    def sigmoidPrime(self,z):
        return np.multiply((self.sigmoid(z)) , np.subtract(1,self.sigmoid(z)) )

    # Backpropgation Function
    def deltaTrain(self,X,y):
        # Backpropagates error from learned values
        self.yHat = self.forward(X)
        # Backpropagates error to weights from the hidden layer
        # to the output
        delta4 = np.multiply((y-self.yHat), self.sigmoidPrime(self.z4))
        djdW3 = np.dot(self.a3.T,delta4)
        self.W3 += djdW3*self.learningRate
        
        delta3 = np.multiply(np.dot(delta4,self.W3.T),self.sigmoidPrime(self.z3))
        djdW2 = np.dot(self.a2.T,delta3)
        #print(djdW2.shape)
        self.W2 += djdW2*self.learningRate

        delta2 = np.multiply(np.dot(delta3,self.W2.T),self.sigmoidPrime(self.z2))
        djdW1 = np.dot(X.T,delta2)
        self.W1 += djdW1*self.learningRate

    # Calculates the error of the neural network
    def errorFunction(self,X,y):
        self.yHat = self.forward(X)
        E = 0.5*sum(np.square(y-self.yHat))
        return E

    def costFunctionPrime(self,X,y):
        self.yHat = self.forward(X)
        delta = np.multiply((y-self.yHat),self.sigmoidPrime(self.z1))
        djdW1 = np.dot(np.transpose(X),delta)
        return djdW1
    
    def fit(self,X,y):
        costHist = []
        for i in range(self.MAXITER):
            self.deltaTrain(X,y)
            costHist.append(self.errorFunction(X,y))
            pass
        pass


import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
# Create neural network object class
X,y = make_moons(n_samples=600,noise=0.2)
bias = np.ones((X.shape[0],1))
X = np.hstack((X,bias))
y_ = y
y = y.reshape(-1,1)

XTrain = X[:500]
yTrain = y[:500]
XTest = X[500:]
yTest = y[500:]

plt.figure(figsize=(10,8))
plt.title("Training Data")
colors = ['r','b']
for i in range(2):
    idx = np.where(yTrain==i)
    plt.scatter(XTrain[idx,0],XTrain[idx,1],c=colors[i],label=i)
plt.xlabel("x1")
plt.ylabel("y1")
plt.legend()
plt.show()

plt.figure(figsize=(10,8))
plt.title("Testing Data")
colors = ['r','b']
for i in range(2):
    idx = np.where(yTest==i)
    plt.scatter(XTest[idx,0],XTest[idx,1],c=colors[i],label=i)
plt.xlabel("x1")
plt.ylabel("y1")
plt.legend()
plt.show()

NN = Neural_Network()
print("-- error before on testing---- ")
print(NN.errorFunction(XTest,yTest))
NN.fit(X,y)
print("-- error after on testing---- ")
print(NN.errorFunction(XTest,yTest))

from matplotlib.colors import ListedColormap

def graphDecisionBounds(X,Y,NN_):
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    h = 0.02
    xx,yy = np.meshgrid(np.arange(-50,50),np.arange(-50,50))

    X_ = X
    Y_ = Y

    x_min, x_max = X_[:, 0].min() - 1, X_[:, 0].max() + 1
    y_min, y_max = X_[:, 1].min() - 1, X_[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    xx_ = np.c_[xx.ravel(), yy.ravel(),np.ones((xx.ravel().shape[0],1))]
    Z = NN_.forward(xx_)

    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    #plt.scatter(X_[:, 0], X_[:, 1], c=y[:100], cmap=cmap_bold
    
    plt.scatter(X_[:, 0], X_[:, 1], c=Y_.flatten(), cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

plt.figure(figsize=(10,8))
plt.title("Decision Boundary on Training Set (Red = 1, Blue = 0)")
graphDecisionBounds(XTrain,yTrain,NN)

plt.figure(figsize=(10,8))
plt.title("Decision Boundary on Testing Set (Red = 1, Blue = 0)")
graphDecisionBounds(XTest,yTest,NN)

