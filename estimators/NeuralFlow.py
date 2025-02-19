import matplotlib.pyplot as plt
import skflow
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

from utils.initializer import *


class NeuralFlowRegressor(BaseEstimator):
    def get_params(self, deep=True):
        return {
            "uniform_init": self.uniform_init,
            "learning_rate": self.learning_rate,
            "activation": self.activation_name,
            "optimize": self.optimize,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "weights_matrix": self.weights_matrix,
            "model_fn": self.model_fn
        }

    def set_params(self, **params):
        for param, value in params.items():
            self.__setattr__(param, value)
        return self

    def __init__(self, uniform_init=True, learning_rate=1E-01, activation="ReLu", optimize="Adam", steps=1000,
                 batch_size=100, weights_matrix=None, model_fn=None,verbose=0,cross_validation=False,hidden_nodes=None):
        print "Initialization"
	self.activation_name = activation
        if (activation == "ReLU"):
            self.activation = tf.nn.relu
        elif (activation == "sigmoid"):
            self.activation = tf.nn.sigmoid
        # Initialize neural network shape
        self.learning_rate = learning_rate
        self.steps = steps
        self.optimize = optimize
        self.batch_size = batch_size
        self.weights_matrix = weights_matrix
        self.uniform_init = uniform_init
        if (model_fn != None):
            self.model_fn = model_fn
        else:
            self.model_fn = self.model_regression
        self.weights_matrix = None
        self.network = None
        self.verbose = verbose
        self.cross_validation = cross_validation
        self.hidden_nodes = hidden_nodes
    def model_regression(self, X, y):
        input_flow = X
        for hidden_node in self.n_hidden:
            input_layer = tf.contrib.layers.fully_connected(input_flow, hidden_node, activation_fn=tf.nn.relu,
                                                        weight_init=self.weight_init, bias_init=self.bias_init)
            input_flow = input_layer
        # hidden_layer = tf.contrib.layers.fully_connected(input_layer,self.n_hidden,activation_fn=tf.nn.relu)
        pred = tf.contrib.layers.fully_connected(input_flow, self.n_output, weight_init=self.weight_init,
                                                 bias_init=self.bias_init)
        return skflow.models.linear_regression(pred, y)

    def score(self, X, y):
        return mean_squared_error(y_true=y, y_pred=self.network.predict(X))

    def predict(self, X):
        return self.network.predict(X)

    def fit(self, X, y, **param):
        self.neural_shape = []
        if(param.has_key('neural_shape')):
            self.neural_shape = param.get("neural_shape")
            self.n_output = self.neural_shape[-1]
            self.n_hidden = self.neural_shape[1:-1]
            self.number_of_layers = len(self.neural_shape)
        else:
            self.n_input = len(X[0])
            self.n_output = len(y[0])
            self.neural_shape = self.hidden_nodes.tolist()
            self.neural_shape.insert(0,self.n_input)
            self.neural_shape.append(self.n_output)
            self.n_hidden = self.hidden_nodes
        self.kFold = KFold(X.shape[0], n_folds=5)
        self.weights_matrix = param.get('weights_matrix')
        self.weight_layers = [(self.neural_shape[t - 1], self.neural_shape[t]) for t in
                              range(1, len(self.neural_shape))]
        self.bias_layers = [self.neural_shape[t] for t in range(1, len(self.neural_shape))]
        self.total_nodes_per_layer = zip(self.weight_layers, self.bias_layers)
        self.total_nodes = 0
        for layer in self.total_nodes_per_layer:
            self.total_nodes += (layer[0][0] + 1) * layer[0][1]
        # If weights are None then initialize randomly
        if (self.weights_matrix == None):
            self.W, self.b = initialize_param(self.weight_layers, self.bias_layers, self.uniform_init)
        else:
            self.W, self.b = self.set_weights(self.weights_matrix)
        # Iterator for weights and bias
        self.W_iter = iter(self.W)
        self.b_iter = iter(self.b)
        # Initialize neural network layers
        self.X = tf.placeholder("float", [None, self.neural_shape[0]], name="input")
        self.y = tf.placeholder("float", [None, self.neural_shape[-1]], name="output")
        #self.config_addon = skflow.addons.ConfigAddon(num_cores=4, gpu_memory_fraction=0.6)
        self.network = skflow.TensorFlowEstimator(model_fn=self.model_fn, n_classes=0,
                                                  steps=self.steps, learning_rate=self.learning_rate,
                                                  batch_size=self.batch_size,
                                                  optimizer=self.optimize,verbose=self.verbose,continue_training=True)
        if(self.cross_validation):
            for train,test in self.kFold:
                self.network.fit(X[train],y[train])
        else:
            self.network.fit(X,y)
        return self

    def weight_init(self, shape, dtype):
        W = self.W_iter.next()
        return tf.convert_to_tensor(W, dtype=dtype)

    def save(self, filename):
        return self.network.save(filename)

    def bias_init(self, shape, dtype):
        b = self.b_iter.next()
        return tf.convert_to_tensor(b, dtype=dtype)

    def plot(self, y_actual, y_pred, label=["predict", "actual"]):
        ax = plt.subplot()
        ax.plot(y_actual, label=label[0])
        ax.plot(y_pred, label=label[1])
        plt.show()

    def set_weights(self, weights_matrix):
        if (len(weights_matrix) != self.total_nodes):
            print "Check again weights shape, must be equal with total nodes"
            return
        self.total_nodes_per_layer = zip(self.weight_layers, self.bias_layers)
        current_pos = 0
        W = []
        b = []
        for layer in self.total_nodes_per_layer:
            total_nodes = (layer[0][0] + 1) * layer[0][1]
            weights = weights_matrix[current_pos:total_nodes + current_pos]
            current_pos = total_nodes
            b.append(weights[-layer[1]:])
            W.append(np.array(weights[:-layer[1]].reshape(layer[0])))
        self.W = W
        self.b = b
        return W, b
