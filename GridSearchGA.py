from sklearn.grid_search import GridSearchCV
from initializer import *
from MetricFeeder import MetricFeeder
from GAEstimator import GAEstimator
param_dicts = {
    "cross_rate":[0.6,0.65,0.7,0.8,0.9],
    "pop_size":[45,50,60],
    "mutation_rate":np.arange(0.01,0.05,step=0.01)
}
n_windows = 4
n_hidden = 10
range_training = (-1,28919)
range_test = (28919,-1)
# metric_types = ["cpu_util","disk_write_rate","disk_read_rate","network_"]
dataFeeder = MetricFeeder()
X_train,y_train,X_test,y_test = dataFeeder.split_train_and_test(n_sliding_window=n_windows)

neural_shape = [n_windows*len(dataFeeder.metric_type),n_hidden,len(dataFeeder.metric_type)]
estimator = GAEstimator()
fit_param = {'neural_shape':neural_shape}
# estimator.fit(X,y,**fit_param)
gs = GridSearchCV(estimator,param_grid=param_dicts,n_jobs=1,fit_params=fit_param,scoring='mean_squared_error')
gs.fit(X_train,y_train)
print gs.best_estimator_
print gs.best_estimator_.score(X_test,y_test)