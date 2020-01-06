import numpy as np
import matplotlib.pyplot as plt

class SVM:
	def __init__(self,eta, C, niter, batch_size, verbose):
		self.eta = eta; self.C = C; self.niter = niter; self.batch_size = batch_size; self.verbose = verbose

	def make_one_versus_all_labels(self, y, m):
		"""
		y : numpy array of shape (n,)
		m : int (in this homework, m will be 10)
		returns : numpy array of shape (n,m)
		"""
		self.m=m
		indic=-np.ones((len(y),m))
		for i in range(len(y)):
			indic[i,y[i]]=1
		return indic


	def compute_loss(self, x, y):
		"""
		x : numpy array of shape (minibatch size, 401)
		y : numpy array of shape (minibatch size, 10)
		w : numpy array of shape (401,10)
		<w,x> : numpy array of shape (minibatch size, 10)
		returns : float
		"""
		return 0.5 * (self.w ** 2).sum() + self.C/x.shape[0]*(np.maximum(0, 1 - np.dot(x,self.w)*y)**2).sum()


	def compute_gradient(self, x, y):
		"""
		x : numpy array of shape (minibatch size, 401)
		y : numpy array of shape (minibatch size, 10)
		returns : numpy array of shape (401, 10)
		"""
		return self.w-2*self.C/x.shape[0] * np.dot(np.transpose(x),np.maximum(0,1-np.dot(x,self.w)*y)*y)

	# Batcher function
	def minibatch(self, iterable1, iterable2, size=1):
		l = len(iterable1)
		n = size
		for ndx in range(0, l, n):
			index2 = min(ndx + n, l)
			yield iterable1[ndx: index2], iterable2[ndx: index2]

	def infer(self, x):
		"""
		x : numpy array of shape (number of examples to infer, 401)
		returns : numpy array of shape (number of examples to infer, 10)
		"""
		y_infer=-np.ones((x.shape[0],self.w.shape[1]))
		index=np.argmax(np.dot(x,self.w),axis=1)
		for i in range(x.shape[0]):
			y_infer[i,index[i]]=1
		return y_infer

	def compute_accuracy(self, y_inferred, y):
		"""
		y_inferred : numpy array of shape (number of examples, 10)
		y : numpy array of shape (number of examples, 10)
		returns : float
		"""
		return 1 - 0.25 * np.abs(y-y_inferred).sum()/len(y)

	def fit(self, x_train, y_train, x_test, y_test):
		"""
		x_train : numpy array of shape (number of training examples, 401)
		y_train : numpy array of shape (number of training examples, 10)
		x_test : numpy array of shape (number of training examples, 401)
		y_test : numpy array of shape (number of training examples, 10)
		returns : float, float, float, float
		"""
		self.num_features = x_train.shape[1]
		self.m = y_train.max() + 1
		y_train = self.make_one_versus_all_labels(y_train, self.m)
		y_test = self.make_one_versus_all_labels(y_test, self.m)
		self.w = np.zeros([self.num_features, self.m])
		train_loss=np.zeros(self.niter)
		train_accuracy=np.zeros(self.niter)
		test_loss=np.zeros(self.niter)
		test_accuracy=np.zeros(self.niter)

		for iteration in range(self.niter):
			# Train one pass through the training set
			for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
				grad = self.compute_gradient(x,y)
				self.w -= self.eta * grad

			# Measure loss and accuracy on training set
			train_loss[iteration] = self.compute_loss(x_train,y_train)
			y_inferred = self.infer(x_train)
			train_accuracy[iteration] = self.compute_accuracy(y_inferred, y_train)

			# Measure loss and accuracy on test set
			test_loss[iteration] = self.compute_loss(x_test,y_test)
			y_inferred = self.infer(x_test)
			test_accuracy[iteration] = self.compute_accuracy(y_inferred, y_test)

			if self.verbose:
				print("Iteration %d:" % iteration)
				print("Train accuracy: %f" % train_accuracy)
				print("Train loss: %f" % train_loss)
				print("Test accuracy: %f" % test_accuracy)
				print("Test loss: %f" % test_loss)
				print("")

		return train_loss, train_accuracy, test_loss, test_accuracy

if __name__ == "__main__":
	# Load the data files
	print("Loading data...")
	x_train = np.load("train_features.npy")
	x_test = np.load("test_features.npy")
	y_train = np.load("train_labels.npy")
	y_test = np.load("test_labels.npy")

	print("Fitting the model...")
	train_l=[]
	train_a=[]
	test_l=[]
	test_a=[]
	C=[0.1, 1, 30, 50]
	for c in C:
		svm = SVM(eta=0.001, C=c, niter=200, batch_size=5000, verbose=False)
		train_loss, train_accuracy, test_loss, test_accuracy = svm.fit(x_train, y_train, x_test, y_test)
		train_l.append(train_loss)
		train_a.append(train_accuracy)
		test_a.append(test_accuracy)
		test_l.append(test_loss)

	fig, ax = plt.subplots()
	for i in range(len(C)):
		ax.plot(np.arange(1,201), train_accuracy[i], 'b', label='C='+str(C[i]))
	ax.legend(loc='center right', shadow=True)
	plt.ylabel("Accuracy")
	plt.xlabel("Iteration")
	plt.savefig('acc.png')
	# # to infer after training, do the following:
	# y_inferred = svm.infer(x_test)

	## to compute the gradient or loss before training, do the following:
	# y_train_ova = svm.make_one_versus_all_labels(y_train, 10) # one-versus-all labels
	# svm.w = np.zeros([401, 10])
	# grad = svm.compute_gradient(x_train, y_train_ova)
	# loss = svm.compute_loss(x_train, y_train_ova)

