from __init__ import *
from sklearn.metrics import mean_squared_error
from io_utils.GFeeder import GFeeder
from math import fabs
import skflow
from utils.GraphUtil import *
from io_utils.NumLoad import *
import matplotlib.pyplot as plt
model = skflow.TensorFlowEstimator.restore("params/model_full_metric_0.00409917244599")

n_sliding_window = 2
skip_lists=3
dataFeeder = GFeeder(skip_lists,normalize_space=True)
# dataFeederNormalize = GFeeder()
#
metric_type = [dataFeeder.CPU_UTIL,dataFeeder.DISK_IO_TIME,dataFeeder.DISK_SPACE,dataFeeder.MEM_USAGE]

X_train,y_train,X_test,y_test = dataFeeder.split_train_and_test(metrics=metric_type,n_sliding_window=n_sliding_window)
# X_trainn,y_trainn,X_testn,y_testn = dataFeederNormalize.split_train_and_test(metrics=metrics_types,n_sliding_window=n_sliding_window)

# X_train,y_train,X_test,y_test = load_training_from_npz("data/gdata/data_training_origin.npz")
# ax = plt.subplot()
# X_trainn,y_trainn,X_testn,y_testn = load_training_from_npz("data/gdata/data_training.npz")
y_pred = model.predict(X_test)
# plot_metric_figure(y_pred=y_pred,y_test=y_test, metric_type=metric_type,title="GA Prediction")
for k, metric in enumerate(metric_type):
    print "%s = %s"%(metric_type[k],mean_squared_error(y_pred[:, k], y_test[:, k]))
#ax = plt.subplot()
#ax.plot(y_test,label='actual')
#ax.plot(y_pred,label='predict')
#plt.show()
# for i in [1,2]:
#     io_max = y_test[:,i].max()
#     io_min = y_test[:,i].min()
#     y_pred[i] = np.array(y_pred[i])*fabs(io_max-io_min)+io_min
# plot_figure(y_pred_convert, y_test[:, 1], title="GABP")
