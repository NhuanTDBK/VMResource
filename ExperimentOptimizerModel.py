from OptimizerNNEstimator import *
from MetricFeeder import *

n_windows = 5
range_training = (-1,28919)
range_test = (28919,-1)
metric_types = ["cpu_util","disk_write_rate"]
dataFeeder = MetricFeeder()
X_train,y_train = dataFeeder.fetch(metric_types,n_windows,range_training)
X_test,y_test = dataFeeder.fetch(metric_types,n_windows,range_test)