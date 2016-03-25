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
estimator = GAEstimator(cross_rate=0.7,mutation_rate=0.02999999999999999999,pop_size=45)
archive_solution = construct_solution(estimator.number_of_solutions,neural_shape)
fit_param = {'neural_shape':neural_shape,"archive":archive_solution}
neuralNet = NeuralFlowRegressor()
optimizer = OptimizerNNEstimator(estimator,neuralNet)
optimizer.fit(X_train,y_train,**fit_param)
print optimizer.score(X_test,y_test)
y_pred = optimizer.predict(X_test)
disk_write_pred = dataFeeder.convert(y_pred[:,1],type=metric_types[1])
disk_write_test = dataFeeder.convert(y_test[:,1],type=metric_types[1])
plot_figure(y_pred=disk_write_pred,y_true=disk_write_test,title="Disk write Prediction based on GANN - score = %s"%np.sqrt(mean_squared_error(disk_write_test,disk_write_pred)))
plot_figure(y_pred=y_pred[:,0],y_true=y_test[:,0],title="CPU Prediction based on GANN - score = %s"%np.sqrt(mean_squared_error(y_pred[:,0],y_test[:,0])))
