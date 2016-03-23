from sklearn.grid_search import GridSearchCV
from initializer import *
from MetricFeeder import MetricFeeder
from GAEstimator import GAEstimator
param_dicts = {
    "cross_rate":[0.6,0.65,0.7,0.8,0.9],
    "pop_size":[45,50,60],
    "mutation_rate":np.arange(0.01,0.05,step=0.01)
}
n_sliding_window = 13
n_periodic = 1
n_input = n_sliding_window + n_periodic
neural_shape=[2*n_input,15 ,1]
metric_types = ["cpu_util","disk_write_rate"]
dataFeeder = MetricFeeder(split_size=5)
X_train,y_train,X_test,y_test = dataFeeder.split_train_and_test(metrics=metric_types,n_sliding_window=n_sliding_window)
estimator = GAEstimator()
fit_param = {'neural_shape':neural_shape}
# estimator.fit(X,y,**fit_param)
gs = GridSearchCV(estimator,param_grid=param_dicts,n_jobs=-1,fit_params=fit_param,scoring='mean_squared_error')
gs.fit(X_train,y_train)
print gs.best_estimator_
print gs.best_estimator_.score(X_test,y_test)