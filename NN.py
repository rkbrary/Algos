import pickle
import numpy as np


class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot"
                 ):
        self.hidden_dims = hidden_dims      # Tuple with the number of neurons in each hidden layer
        self.n_hidden = len(hidden_dims)    # Number of hidden layers
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr                        # Learning Rate
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon
        self.weights = {}
        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        # self.weights is a dictionary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))
            self.weights[f"W{layer_n}"] = np.random.uniform(-1/np.sqrt(all_dims[layer_n-1]),1/np.sqrt(all_dims[layer_n-1]),(all_dims[layer_n-1],all_dims[layer_n]))

    def relu(self, x, grad=False):
        if grad:
            return (x>0)
        return np.maximum(x,0)

    def sigmoid(self, x, grad=False):
        if grad:
            return np.exp(-x)/(1+np.exp(-x))**2
        return 1/(1+np.exp(-x))

    def tanh(self, x, grad=False):
        if grad:
            return 4/(np.exp(x)+np.exp(-x))**2
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            return self.tanh(x, grad)
        else:
            raise Exception("invalid")

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        # WRITE CODE HERE
        c = np.max(x)
        X = np.exp(x - c)
        if len(x.shape) == 1: return X/X.sum()
        return X/(X.sum(axis=1)[:,np.newaxis])

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionary with keys Z0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i

        for layer_n in range(1,self.n_hidden+1):
            cache[f"A{layer_n}"]= self.weights[f"b{layer_n}"]+np.dot(cache[f"Z{layer_n-1}"],self.weights[f"W{layer_n}"])
            cache[f"Z{layer_n}"]=self.activation(cache[f"A{layer_n}"])

        cache[f"A{self.n_hidden+1}"]=self.weights[f"b{self.n_hidden+1}"]+np.dot(cache[f"Z{self.n_hidden}"],self.weights[f"W{self.n_hidden+1}"])
        cache[f"Z{self.n_hidden+1}"] = self.softmax(cache[f"A{self.n_hidden+1}"])
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        if len(labels.shape)==1: labels=labels[np.newaxis,:]
        # grads is a dictionary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        grads = {}
        grads[f"dA{self.n_hidden+1}"]=output-labels
        grads[f"dW{self.n_hidden + 1}"] = np.dot(cache[f"Z{self.n_hidden}"].T, grads[f"dA{self.n_hidden + 1}"]) / len(labels)
        grads[f"db{self.n_hidden+1}"]=np.mean(grads[f"dA{self.n_hidden+1}"],axis=0)[np.newaxis,:]
        for layer_n in range(1,self.n_hidden + 1):
            grads[f"dZ{self.n_hidden + 1 - layer_n}"]=np.dot(grads[f"dA{self.n_hidden+2-layer_n}"],self.weights[f"W{self.n_hidden+2-layer_n}"].T)
            grads[f"dA{self.n_hidden + 1 - layer_n}"]=grads[f"dZ{self.n_hidden+2-layer_n-1}"]*self.activation(cache[f"A{self.n_hidden+2-layer_n-1}"],grad=True)
            grads[f"dW{self.n_hidden + 1 - layer_n}"]=np.dot(cache[f"Z{self.n_hidden-layer_n}"].T,grads[f"dA{self.n_hidden + 1 - layer_n}"])/len(labels)
            grads[f"db{self.n_hidden + 1 - layer_n}"]=np.mean(grads[f"dA{self.n_hidden + 1 - layer_n}"],axis=0)[np.newaxis,:]
        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"]-=self.lr*grads[f"dW{layer}"]
            self.weights[f"b{layer}"]-=self.lr*grads[f"db{layer}"]

    def one_hot(self, y):
        one_y=np.zeros((len(y),self.n_classes),dtype=int)
        for i,value in enumerate(y):
            one_y[i, int(value)]=1
        return one_y

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        return -np.sum(labels*np.log(prediction))/len(labels)

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                self.update(self.backward(self.forward(minibatchX),minibatchY))
            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)
            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        return self.compute_loss_and_accuracy(X_test, y_test)[0:2]