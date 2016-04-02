from sklearn.grid_search import GridSearchCV
from estimators.GAEstimator import GAEstimator
from io_utils.GFeeder import GFeeder
from utils.initializer import *

param_dicts = {
    "cross_rate":[0.6,0.65,0.7,0.8,0.9],
    "pop_size":[45,50,60],
    "mutation_rate":np.arange(0.01,0.05,step=0.01)
}
n_windows = 4
n_hidden = 10
# metric_types = ["cpu_util","disk_write_rate","disk_read_rate","network_"]
dataFeeder = GFeeder()
X_train,y_train,X_test,y_test = dataFeeder.split_train_and_test(n_sliding_window=n_windows)

neural_shape = [n_windows*len(dataFeeder.metric_type),n_hidden,len(dataFeeder.metric_type)]
estimator = GAEstimator()
fit_param = {'neural_shape':neural_shape}
# estimator.fit(X,y,**fit_param)
gs = GridSearchCV(estimator,param_grid=param_dicts,n_jobs=1,fit_params=fit_param,scoring='mean_squared_error')
gs.fit(X_train,y_train)
print gs.best_estimator_
print gs.best_estimator_.score(X_test,y_test)