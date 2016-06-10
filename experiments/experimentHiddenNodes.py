from estimators.NeuralFlow import NeuralFlowRegressor
from estimators.BruceForceGridSearchBase import BruteForceGridSearchBase
import numpy as np
from utils.GraphUtil import *
from sklearn.preprocessing import MinMaxScaler
from estimators.GAEstimator import GAEstimator
from estimators.OptimizerNNEstimator import OptimizerNNEstimator
from preprocessing.FuzzyProcessor import FuzzyProcessor
print "Loading data"
scaler = MinMaxScaler()
data = scaler.fit_transform(pd.read_csv('../sample_610_10min.csv',index_col=0)['cpu_rate'])
bruteGrid = BruteForceGridSearchBase(n_sliding_ranges=np.arange(2,4),fuzzy_transform=FuzzyProcessor(automf=True,fuzzy_distance=0.01))
bruteGrid.transform(data)
result = []
for train_item in bruteGrid.train_bucket:
    X_train, y_train, X_test, y_test = train_item.getitems()
    hidden_assume = X_train.shape[1]+y_train.shape[1]
    n_hidden_range = np.arange(hidden_assume,hidden_assume+2)
    for hidden_node in n_hidden_range:
        n_hidden = np.array([hidden_node])
        estimator = GAEstimator(cross_rate=0.65, mutation_rate=0.01,pop_size=45,gen_size=100)
        neuralNet = NeuralFlowRegressor(learning_rate=1E-03,steps=6000, hidden_nodes=n_hidden)
        neural_shape = [X_train.shape[1], n_hidden[0], y_train.shape[1]]
        fit_param = {'neural_shape': neural_shape}
        optimizer = OptimizerNNEstimator(estimator, neuralNet)
        optimizer.fit(X_train, y_train, **fit_param)
        X_test_f = bruteGrid.data_source[bruteGrid.train_len + 1:bruteGrid.train_len + bruteGrid.test_len + 1]
        # y_pred_f = neuralNet.predict(X_test)
        y_pred_f = optimizer.predict(X_train)
        # y_pred = bruteGrid.fuzzy_transform.defuzzy(X_test_f, y_pred_f)
        score_nn = np.sqrt(mean_squared_error(y_pred=y_pred_f, y_true=y_train))
        tmp = {
            'score': score_nn,
            'n_sliding': train_item.metadata['sliding_windows'],
            'hidden_node':hidden_node
        }
        result.append(tmp)
        break
result = pd.DataFrame(result)
result.sort(['score']).to_csv('neural_flow_hiddens.csv',index=None)
print result[result.score==result.score.min()]