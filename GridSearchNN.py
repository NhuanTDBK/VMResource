from GraphUtil import *
from MetricFeeder import *
from GAEstimator import GAEstimator
from initializer import *
from NeuralFlow import NeuralFlowRegressor
from OptimizerNNEstimator import OptimizerNNEstimator
from sklearn.metrics import mean_squared_error

n_windows = 4
n_hidden = 15
range_training = (-1,28919)
range_test = (28919,-1)
metric_types = ["cpu_util","disk_write_rate"]
dataFeeder = MetricFeeder(skip_lists=5)

print "Getting data"
X_train,y_train,X_test,y_test = dataFeeder.split_train_and_test(metric_types,n_windows)
neural_shape = [2*n_windows,n_hidden,2]
fit_param = {'neural_shape':neural_shape}
optimizer = NeuralFlowRegressor()
print optimizer.score(X_test,y_test)