# Experiment GABPNN
import pandas as pd
import numpy as np
from estimators.GAEstimator import GAEstimator
from estimators.NeuralFlow import NeuralFlowRegressor
from estimators.OptimizerNNEstimator import OptimizerNNEstimator
from io_utils.GFeeder import GFeeder
# from utils.GraphUtil import *
# from utils.initializer import *

# length of sliding windows for input
n_sliding_window = 2

#Getting Google cluster data
dataFeeder = GFeeder(skip_lists=1)
metrics_types = [dataFeeder.CPU_UTIL]
# metrics_windows = {
#     dataFeeder.CPU_UTIL : 3,
#     dataFeeder.DISK_IO_TIME,
#     dataFeeder.DISK_SPACE,
#
# }
dat = pd.read_csv('sample_610_10min.csv',index_col=0,parse_dates=True)
print "Getting data"
X_train,y_train,X_test,y_test = dataFeeder.split_train_and_test(data=dat,metrics=['cpu_rate'],n_sliding_window=n_sliding_window)
# Number of hiddens node (one hidden layer)

score_list = {}
for n_hidden in np.arange(10,30,step=1):
    # n_hidden = 80
    # Define neural shape
        # Input layer: [n_sample*n_size]
        # Hidden layer:
        # Output layer: regression
    neural_shape = [dataFeeder.input_size,n_hidden,dataFeeder.output_size]
    # Initialize GA Estimator
    # estimator = GAEstimator(cross_rate=0.7,mutation_rate=0.04,pop_size=60,gen_size=100)

    fit_param = {'neural_shape':neural_shape}

    # Initialize neural network model for regression
    neuralNet = NeuralFlowRegressor(learning_rate=1E-03,optimize='SGD',activation='sigmoid')

    # There are many techniques for combining GA with NN. One of this, the optimizer solution of GA will be weights initialized of NN
    # optimizer = OptimizerNNEstimator(estimator,neuralNet)
    optimizer = neuralNet
    optimizer.fit(X_train,y_train,**fit_param)
    score = optimizer.score(X_test,y_test)
    print score
    score_list[n_hidden]=score
    optimizer.save("params/model_full_metric_%s"%score)
# if score < 0.01:
# y_pred = optimizer.predict(X_test)
score_list = pd.Series(score_list)
print "Optimal hidden nodes: %s, with score = %s"%(score_list.argmin(),score_list.min())
