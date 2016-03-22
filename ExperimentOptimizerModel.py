from OptimizerNNEstimator import *
from MetricFeeder import *
from ACOEstimator import ACOEstimator
from initializer import *
from sklearn.grid_search import GridSearchCV

n_windows = 5
n_hidden = 27
range_training = (-1,28919)
range_test = (28919,-1)
metric_types = ["cpu_util","disk_write_rate"]
dataFeeder = MetricFeeder(skip_lists=5)
# X_train,y_train = dataFeeder.fetch(metric_types,n_windows)
# X_test,y_test = dataFeeder.fetch(metric_types,n_windows)
print "Getting data"
X_train,y_train,X_test,y_test = dataFeeder.split_train_and_test(metric_types,n_windows)
param_dicts = {
    "Q":np.arange(0.01,0.05,step=0.01),
    "epsilon":np.arange(0.1,0.8,step=0.05),
    "number_of_solutions":np.arange(30,200)
}
neural_shape = [2*n_windows,n_hidden,2]
estimator = ACOEstimator()
archive_solution = construct_solution(estimator.number_of_solutions,neural_shape)
fit_param = {'neural_shape':neural_shape,"archive":archive_solution}
# estimator.fit(X,y,**fit_param)
print "Estimate parameter"
gs = GridSearchCV(estimator,param_grid=param_dicts,n_jobs=-1,fit_params=fit_param,scoring='mean_squared_error')
gs.fit(X_train,y_train)
print gs.best_estimator_