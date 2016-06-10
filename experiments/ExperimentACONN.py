from estimators.ACOEstimator import ACOEstimator
from estimators.NeuralFlow import NeuralFlowRegressor

from estimators.OptimizerNNEstimator import OptimizerNNEstimator
from utils.GraphUtil import *
from utils.initializer import *
from estimators.FuzzyStep import *
from utils.SlidingWindowUtil import SlidingWindow

print 'FACO Model'
data = pd.read_csv('../sample_610_10min.csv')

# define training_set and testing _set
training_set_size = 2000
testing_set_size = 500


######## FUZZY before TRAINING
# length of sliding windows for input
n_sliding_window = 1

#define training_set, testing set
X_train = np.zeros(training_set_size)
for i in range (0,training_set_size):
    X_train[i] = data.cpu_rate[i]

y_train = np.zeros(training_set_size)
for i in range (0,training_set_size):
    y_train[i] = data.cpu_rate[i + 1]

X_test = np.zeros(testing_set_size)
for i in range (0,testing_set_size):
    X_test[i] = data.cpu_rate[i + training_set_size]

y_test = np.zeros(testing_set_size - 1)
for i in range (0,testing_set_size - 1):
    y_test[i] = data.cpu_rate[i + training_set_size + 1]

# Number of hiddens node (one hidden layer)
n_hidden = np.array([55])
# Define neural shape
    # Input layer: [n_sample*n_size]
    # Hidden layer:
    # Output layer: regression
# neural_shape = [fuzzy_set_size,n_hidden,fuzzy_set_size]
# neural_shape = [fuzzy_set_size,n_hidden,fuzzy_set_size]

# Initialize ACO Estimator 130
estimator = ACOEstimator(Q=0.65,epsilon=0.2,number_of_solutions=10,hidden_nodes=n_hidden)

# fit_param = {'neural_shape':neural_shape}


# fuzzy before training
# fuzzy step:
X_train_f = fuzzy(X_train)
y_train_f = fuzzy(y_train)
X_test_f = fuzzy(X_test)

# # # Initialize neural network model for regression
neuralNet = NeuralFlowRegressor(learning_rate=1E-04,hidden_nodes=n_hidden)
# # There are many techniques for combining GA with NN. One of this, the optimizer solution of GA will be weights initialized of NN
optimizer = OptimizerNNEstimator(estimator,neuralNet)
# # optimizer = neuralNet
optimizer.fit(X_train_f,y_train_f)
# print  optimizer.score(X_train_f,y_train_f)
#
y_pred = optimizer.predict(X_test_f)

#defuzzy step:
ft = defuzzy(X_test, y_pred)
#mean_squared_error
print mean_squared_error(y_test, ft)
plot_figure(ft,y_test,color=['blue','red'],title='FGANN')
