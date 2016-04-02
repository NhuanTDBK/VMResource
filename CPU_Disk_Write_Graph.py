import skflow
from sklearn import preprocessing

from __init__ import *
from utils.SlidingWindowUtil import *

result = pd.read_json("result_exp1")
neural_network = skflow.TensorFlowEstimator.restore("score_0.000155509367764/")


vm_name = " 17d13bbf-a55d-4f8e-ad9e-c739e12db98d "
total_cpu_util = pd.read_json("sample_cpu_util.json")["Volume"]
total_disk_write = pd.read_json("sample_disk_write.json")["Volume"]
n_range = total_cpu_util.shape[0]*70/100
n_size = 5

scaler  = preprocessing.MinMaxScaler(feature_range=(0,1))
disk_write_test_scaler = scaler.fit_transform(total_disk_write[n_range:].reshape(-1,1))
cpu_test_scaler = scaler.fit_transform(total_cpu_util[n_range:].reshape(-1,1))

disk_write_X = list(SlidingWindow(disk_write_test_scaler,n_size))
cpu_util_X =  list(SlidingWindow(cpu_test_scaler,n_size))
disk_write_y = disk_write_test_scaler[n_size-1:]
cpu_util_y = cpu_test_scaler[n_size-1:]
X_test = np.asarray([np.array(t,dtype=np.float32).flatten().tolist() for t in zip(cpu_util_X,disk_write_X)])
y_test = np.asarray([np.array(t).flatten().tolist() for t in zip(cpu_util_y,disk_write_y)])

predict_y = neural_network.predict(X_test)
