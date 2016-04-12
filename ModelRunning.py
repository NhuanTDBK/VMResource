from __init__ import *
from sklearn.metrics import mean_squared_error
from io_utils.GFeeder import GFeeder
import skflow
from utils.GraphUtil import *
model = skflow.TensorFlowEstimator.restore("experiments/params/model_full_metric0.00371368895876/")
dataFeeder = GFeeder(file_name='data/gdata/gcluster_1268205_1min.json')
metrics_types = [dataFeeder.CPU_UTIL,dataFeeder.DISK_IO_TIME,dataFeeder.DISK_SPACE,dataFeeder.MEM_USAGE]

n_sliding_window = 2
X_train,y_train,X_test,y_test = dataFeeder.split_train_and_test(metrics=metrics_types,n_sliding_window=n_sliding_window)
y_pred = model.predict(X_test)
io_max = y_test[:,1].max()
io_min = y_test[:,1].min()
y_pred_convert = y_pred[:,1]*(io_max-io_min)+io_min
print mean_squared_error(y_pred_convert,y_test)
plot_figure(y_pred_convert, y_test[:, 1], title="GABP")

