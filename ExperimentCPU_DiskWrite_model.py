
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

#----------------
range_training = (-1,28919)
range_test = (28919,-1)
metric_types = ["cpu_util","disk_write_rate"]
params_estimate = {
    "n_windows":np.arange(5,30),
    "hidden_node":np.arange(10,60)
}
result = {}

candidate_param = list(ParameterGrid(param_grid=params_estimate))

def estimator(n_windows,n_hidden_nodes):

    dataFeeder = MetricFeeder()
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
        result[nn_shape]=score
# print neuralNet.score(X_test,y_test)
# y_pred = neuralNet.predict(X_test)
# plot_figure(y_pred[:,0],y_test[:,0])
out = Parallel(n_jobs=-1)(delayed(estimator)(n_window,n_hidden) for n_window, n_hidden in candidate_param)
pd.DataFrame.from_dict(result,orient='index').to_json("result_exp_%s"%datetime.datetime.today())