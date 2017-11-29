import numpy as np

class Neural_Network(object):
	debug = True
	def __init__(self):
		self.inputLayerSize = 2
		self.hiddenLayerSize = 2
		self.outputLayerSize = 1
		self.learningRate = 0.4
		#self.hiddenLayerSize = 1

		# Weights
		self.W1 = np.random.randn(self.inputLayerSize+1,self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

	def forward(self, X):
		# Propogate inputs through network
		self.z2 = np.dot(X,self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2,self.W2)
		self.yHat = self.sigmoid(self.z3)
		return self.yHat

	# Activation functions
	def sigmoid(self,z):
		# Apply sigmoid activation function
		return 1/(1+np.exp(-z))

	# Error functions
	def deltaTrain(self,X,y):
		self.yHat = self.forward(X)

		delta3 = np.multiply((y-self.yHat), self.sigmoidPrime(self.z3))
		djdW2 = np.dot(self.a2.T,delta3)
		NN.W2 += djdW2*self.learningRate

		delta2 = np.multiply(np.dot(delta3,self.W2.T),self.sigmoidPrime(self.z2))
		djdW1 = np.dot(X.T,delta2)
		NN.W1 += djdW1*self.learningRate

	def errorFunction(self,X,y):
		self.yHat = self.forward(X)
		E = 0.5*sum(np.square(y-self.yHat))
		return E

	def costFunctionPrime(self,X,y):
		self.yHat = self.forward(X)
		delta = np.multiply((y-self.yHat),self.sigmoidPrime(self.z1))
		djdW1 = np.dot(np.transpose(X),delta)
		return djdW1

	def sigmoidPrime(self,z):
		return np.multiply((self.sigmoid(z)) , np.subtract(1,self.sigmoid(z)) )


NN = Neural_Network()
X = np.matrix("0 0 1; 0 1 1; 1 0 1; 1 1 1")
y = np.matrix("0; 1; 1; 0")
print("-- error before---- ")
print(NN.errorFunction(X,y))
for i in range(9000):
	NN.deltaTrain(X,y)
print("-- error after")
print(NN.errorFunction(X,y))
print("-- new weights")
print("-- NN W1")
print(NN.W1)
print("-- NN W2")
print(NN.W2)