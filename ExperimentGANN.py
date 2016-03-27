from GraphUtil import *
from MetricFeeder import *
from GAEstimator import GAEstimator
from initializer import *
from NeuralFlow import NeuralFlowRegressor
from OptimizerNNEstimator import OptimizerNNEstimator
from sklearn.metrics import mean_squared_error
from numpy import savez

n_windows = 4
n_hidden = 15

metric_types = ["cpu_util","disk_write_rate"]
dataFeeder = MetricFeeder(skip_lists=5)

print "Getting data"
X_train,y_train,X_test,y_test = dataFeeder.split_train_and_test(n_sliding_window=n_windows)
# savez("train_test",X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
neural_shape = [len(dataFeeder.metric_type)*n_windows,n_hidden,len(dataFeeder.metric_type)]
model = None
best_fit = 1.0
for i in range(2):
    estimator = GAEstimator(cross_rate=0.7,mutation_rate=0.01,pop_size=45)
    archive_solution = construct_solution(estimator.number_of_solutions,neural_shape,uniform_init=False)
    fit_param = {'neural_shape':neural_shape,"archive":archive_solution}
    neuralNet = NeuralFlowRegressor()
    optimizer = OptimizerNNEstimator(estimator,neuralNet)
    optimizer.fit(X_train,y_train,**fit_param)
    score = optimizer.score(X_test,y_test)
    if(score<best_fit):
        best_fit=score
        model = estimator
# print model
print best_fit

y_pred = optimizer.predict(X_test)
plot_metric_figure(y_pred=y_pred,y_test=y_test, metric_type=dataFeeder.metric_type,title="GANN")
