# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from estimators.ACOEstimator import ACOEstimator
from estimators.NeuralFlow import NeuralFlowRegressor
from estimators.OptimizerNNEstimator import OptimizerNNEstimator
from utils.GraphUtil import *
from utils.initializer import *
from estimators.FuzzyStep import *
from utils.SlidingWindowUtil import SlidingWindow
from estimators.GAEstimator import GAEstimator
from preprocessing.FuzzyProcessor import FuzzyProcessor
from sklearn.cross_validation import train_test_split
from sklearn.utils import check_X_y
from sklearn.grid_search import GridSearchCV
from io_utils import GFeeder
from estimators.GAEstimator import GAEstimator
from estimators.NeuralFlow import NeuralFlowRegressor
from io_utils.GFeeder import GFeeder
from utils.GraphUtil import *
from utils.initializer import *

import numpy as np

# <codecell>

print 'FACO Model'
# data = pd.read_csv('../cpu_usage.csv')
# datafeeder = GFeeder()
data = pd.read_csv('../data/gdata/sampling_617685_metric_1min_datetime.csv',index_col=0)
data.cpu_rate[data.cpu_rate<0.1]=0.1
fuzzy_transform = FuzzyProcessor(automf=True)
training_set_size = 29231
testing_set_size = 12527

n_sliding_window = 2
# <codecell>
fuzzy_train = np.load('fuzzy_train.npz')['fuzzy_train']
# fuzzy_train = fuzzy_transform.fit_transform(data.cpu_rate)

# # <codecell>
# n_sliding_window = 2
X_dat = np.array(list(SlidingWindow(fuzzy_train,n_sliding_window,concatenate=True)))
X_train = X_dat[:training_set_size]
X_test = X_dat[training_set_size:training_set_size+testing_set_size]
y_train = fuzzy_train[1:training_set_size+1]
y_test = data.cpu_usage[training_set_size+1:training_set_size+testing_set_size+1]
#
# # Setup paramters of model
# n_hidden = np.array([80])
# # neural_shape = [X_train.shape[1],n_hidden,y_train.shape[1]]
# # Initialize ACO Estimator
# estimator = ACOEstimator(Q=0.65,epsilon=0.1,number_of_solutions=130,hidden_nodes=n_hidden)
# # estimator = GAEstimator()
# # fit_param = {'neural_shape':neural_shape}
neuralNet = NeuralFlowRegressor(hidden_nodes=10)
gridSearch = GridSearchCV(neuralNet,{'hidden_nodes':[28,50]},n_jobs=-1)
gridSearch.fit(X_train,y_train)
# print gridSearch.score(X_test,y_test)
print gridSearch.grid_scores_
print "Best model %s"%gridSearch.best_estimator_
# optimizer = OptimizerNNEstimator(estimator,neuralNet)
# optimizer.fit(X_train,y_train)
#
# # <codecell>
#
# # param_aco = {
# #     'Q':[0.65,0.7,0.75],
# #     'epsilon':[0.1,0.2,0.3],
# #     'hidden_nodes': [n_hidden]
# #     }
# # gridSearch = GridSearchCV(estimator,param_aco,n_jobs=-1)
# # gridSearch.fit(X_train,y_train)
#
# # <codecell>
#
# y_pred_f = optimizer.predict(X_test)
# X_actual_test = data.cpu_usage[training_set_size+1:training_set_size+1+testing_set_size].as_matrix()
# y_pred = fuzzy_transform.defuzzy(X_actual_test,y_pred_f)
# score_nn = mean_squared_error(y_test[1:], y_pred)
# print score_nn
# plot_figure(y_pred,y_test[1:],color=['blue','red'],title='FGANN %s'%score_nn)
#
# # <codecell>

