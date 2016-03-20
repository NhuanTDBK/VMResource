
# coding: utf-8

# In[12]:

from __init__ import *
from FeedFlow import FeedFlow
from SlidingWindowUtil import SlidingWindow
import matplotlib.pyplot as plt
from NeuralFlow import NeuralFlowRegressor
from sklearn import preprocessing
from sklearn.cross_validation import KFold

vm_name = " 17d13bbf-a55d-4f8e-ad9e-c739e12db98d "
total_cpu_util = pd.read_json("sample_cpu_util.json")["Volume"]
total_disk_write = pd.read_json("sample_disk_write.json")["Volume"]
n_range = total_cpu_util.shape[0]*70/100
scaler  = preprocessing.MinMaxScaler(feature_range=(0,1))

disk_write_train_scaler = scaler.fit_transform(total_disk_write[:n_range].reshape(-1,1))
cpu_train_scaler = scaler.fit_transform(total_cpu_util[:n_range].reshape(-1,1))

disk_write_test_scaler = scaler.fit_transform(total_disk_write[n_range:].reshape(-1,1))
cpu_test_scaler = scaler.fit_transform(total_cpu_util[n_range:].reshape(-1,1))
result = {}
for n_size in np.arange(5,6):
    disk_write_X = list(SlidingWindow(disk_write_train_scaler,n_size))
    cpu_util_X =  list(SlidingWindow(cpu_train_scaler,n_size))
    disk_write_y = disk_write_train_scaler[n_size-1:]
    cpu_util_y = cpu_train_scaler[n_size-1:]
    X_train = np.asarray([np.array(t,dtype=np.float32).flatten().tolist() for t in zip(cpu_util_X,disk_write_X)])
    y_train = np.asarray([np.array(t).flatten().tolist() for t in zip(cpu_util_y,disk_write_y)])
    # In[9]:

    disk_write_X = list(SlidingWindow(disk_write_test_scaler,n_size))
    cpu_util_X =  list(SlidingWindow(cpu_test_scaler,n_size))
    disk_write_y = disk_write_test_scaler[n_size-1:]
    cpu_util_y = cpu_test_scaler[n_size-1:]
    X_test = np.asarray([np.array(t,dtype=np.float32).flatten().tolist() for t in zip(cpu_util_X,disk_write_X)])
    y_test = np.asarray([np.array(t).flatten().tolist() for t in zip(cpu_util_y,disk_write_y)])
    for hidden_node in np.arange(10,11):
        fit_param = {
            'neural_shape':[2*n_size,hidden_node,2]
        }
        neuralNet = NeuralFlowRegressor()
        kfold = KFold(X_train.shape[0],10)
        score_lst = np.zeros(len(kfold),dtype=np.float32)
        for k,(train,test) in enumerate(kfold):
            neuralNet.fit(X_train[train],y_train[train],**fit_param)
        # neuralNet.save("pa2")
        nn_shape = "%s-%s"%(2*n_size,hidden_node)
        score = neuralNet.score(X_test,y_test)
        neuralNet.save("score_%s"%score)
        result[nn_shape]=score

pd.DataFrame.from_dict(result,orient='index').to_json("result_exp")