import numpy as np
from sklearn.grid_search import Parallel,delayed
from estimators.NeuralFlow import NeuralFlowRegressor
from io_utils.GFeeder import GFeeder
def get_params():
    dataFeeder = GFeeder()
    out = Parallel(n_jobs=-1)(delayed(put_queue)
                        (n_input,dataFeeder) for n_input in range(4,21))
    return out
def put_queue(n_input,dataFeeder):
    X_train,y_train,X_test,y_test  = dataFeeder.split_train_and_test(n_sliding_window=n_input)
    retrieve = [n_input,(X_train,y_train,X_test,y_test)]
    return retrieve
def model_fit(param):
    print "Training %s"%param[0]
    X_train = param[1][0]
    y_train = param[1][1]
    X_test = param[1][2]
    y_test = param[1][3]

    neural_shape = [y_train.shape[1]*param[0],10,y_train.shape[1]]
    fit_param = {
        "neural_shape":neural_shape
    }
    neuralNet = NeuralFlowRegressor()
    neuralNet.fit(X_train,y_train,**fit_param)
    return param[0],np.sqrt(neuralNet.score(X_test,y_test))
def fit_and_evaluate():
    out = get_params()
    result = Parallel(n_jobs=-1)(delayed(model_fit)(param) for param in out)
    result_sorted = sorted(result,key=lambda x:x[1])
    # print result_sorted[0]
    return result_sorted
print fit_and_evaluate()