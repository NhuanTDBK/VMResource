
# coding: utf-8

# In[12]:

from __init__ import *
import matplotlib.pyplot as plt
from NeuralFlow import NeuralFlowRegressor
from sklearn.cross_validation import KFold
import datetime
from MetricFeeder import MetricFeeder
from GraphUtil import *
from sklearn.grid_search import ParameterGrid,Parallel,delayed
import numpy as np
#----------------
range_training = (-1,28919)
range_test = (28919,-1)
metric_types = ["cpu_util","disk_write_rate"]
params_estimate = {
    "n_windows":np.arange(5,30),
    "hidden_node":np.arange(10,60)
}
result = {}

candidate_param = ParameterGrid(param_grid=params_estimate)

def estimator(n_windows,n_hidden_nodes):

    dataFeeder = MetricFeeder(split_size=5)
    X_train,y_train = dataFeeder.fetch(metric_types,n_windows,range_training)
    X_test,y_test = dataFeeder.fetch(metric_types,n_windows,range_test)
    #-----------------
    # hidden_node = 15
    fit_param = {
                'neural_shape':[2*n_windows,n_hidden_nodes,2]
            }
    neuralNet = NeuralFlowRegressor()
    kfold = KFold(X_train.shape[0],5)
    score_lst = np.zeros(len(kfold),dtype=np.float32)
    for k,(train,test) in enumerate(kfold):
        neuralNet.fit(X_train[train],y_train[train],**fit_param)
    nn_shape = "%s-%s"%(2*n_windows,n_hidden_nodes)
    score = neuralNet.score(X_test,y_test)
    neuralNet.save("tmp/score_%s"%score)
    return nn_shape,score

# print neuralNet.score(X_test,y_test)
# y_pred = neuralNet.predict(X_test)
# plot_figure(y_pred[:,0],y_test[:,0])
out = [Parallel(n_jobs=-1)(delayed(estimator)(k["n_windows"],k["hidden_node"]) for k in candidate_param)]
result_sorted = sorted(result,key=lambda x:x[1])
print result_sorted[0:10]
np.savez("result_model_%s"%datetime.datetime.now(),result=result_sorted)