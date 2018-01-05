import numpy as np

class Neural_Network(object):
	debug = True
	def __init__(self):
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.learningRate = 0.1

		# Weights
		self.W1 = np.random.randn(self.inputLayerSize+1,self.outputLayerSize)
		print(self.W1)

	def forward(self, X):
		# Propogate inputs through network
		self.z1 = np.dot(X,self.W1)
		self.yHat = self.sigmoid(self.z1)
		'''
		print(self.yHat)'''
		return self.yHat


	# Activation functions
	def sigmoid(self,z):
		# Apply sigmoid activation function
		return 1/(1+np.exp(-z))

	# Error functions
	def deltaTrain(self,X,y):
		yHat = self.forward(X)

		for row in range(X.shape[0]):
			delta = self.learningRate*self.sigmoidPrime(yHat[row])*(y[row]-yHat[row])
			result = np.transpose(delta*X[row])
			self.W1 = self.W1+result

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
		return np.matmul((self.sigmoid(z)) , np.subtract(1,self.sigmoid(z)))


NN = Neural_Network()
X = np.matrix("0 0 1; 0 1 1; 1 0 1; 1 1 1")
y = np.matrix("0; 0; 0; 1")
#X = np.matrix("1 0 1; 0 1 1")
#y = np.matrix("1; 1")
print("error before :", NN.errorFunction(X,y) )
for i in range(500):
	NN.deltaTrain(X,y)
print("error after  :", NN.errorFunction(X,y) )
print(NN.W1)
#print(NN.W1)
'''
print("error before training:",NN.errorFunction(X,y))
for i in range(300):
	NN.deltaTrain(X,y)
print("error after training: ",NN.errorFunction(X,y))
print("---- Final weights ----")
print(NN.W1)'''
