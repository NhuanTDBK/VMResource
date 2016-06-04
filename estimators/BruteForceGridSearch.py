import numpy as np
import pandas as pd
from estimators.ACOEstimator import ACOEstimator
from estimators.NeuralFlow import NeuralFlowRegressor
from estimators.OptimizerNNEstimator import OptimizerNNEstimator
from sklearn.grid_search import GridSearchCV, delayed
from utils.SlidingWindowUtil import SlidingWindow
from estimators.TrainObject import TrainObject
from preprocessing.FuzzyProcessor import FuzzyProcessor
from sklearn.metrics import mean_squared_error
class BruteForceGridSearch():
    def __init__(self,n_sliding_ranges):
        self.n_sliding_ranges = n_sliding_ranges
        self.fuzzy_transform = FuzzyProcessor(automf=True)
    def transform(self,data_source):
        self.data_source = data_source
        self.data_transform = self.fuzzy_transform.fit_transform(data_source)
        self.train_bucket = []
        self.data_placeholder = []
        for n_sliding_window in self.n_sliding_ranges:
            print "Normalize complete"
            training_set_size = int(self.data_transform.shape[0] * 0.7)
            testing_set_size = int(self.data_transform.shape[0] * 0.3)
            print "Preparing sliding window dataset..."
            X_dat = np.array(list(SlidingWindow(self.data_transform, n_sliding_window, concatenate=True)))
            X_train = X_dat[:training_set_size]
            y_train = self.data_transform[1:training_set_size + 1]
            X_test =  data_source[training_set_size+1:training_set_size+1+testing_set_size].as_matrix()
            y_test = data_source[training_set_size + 1:training_set_size + testing_set_size + 1]
            metadata = {
                "sliding_windows":n_sliding_window
            }
            trainObject = TrainObject(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,metadata=metadata)
            self.train_bucket.append(trainObject)
    def fit(self,data=None):
        n_hidden= np.array([55])
        param_aco = {
            'Q':[0.65,0.7,0.75],
            'epsilon':[0.1,0.2,0.3],
            'hidden_nodes': [n_hidden]
        }
        estimator = ACOEstimator(Q=0.65, epsilon=0.1, number_of_solutions=130, hidden_nodes=n_hidden)
        neuralNet = NeuralFlowRegressor(learning_rate=1E-03, hidden_nodes=np.array([n_hidden]))
        result = []
        for train_item in self.train_bucket:
            X_train,y_train,X_test,y_test = train_item.getitems()
            gridSearch = GridSearchCV(estimator,param_aco,n_jobs=-1)
            gridSearch.fit(X_train,y_train)
            optimizer = OptimizerNNEstimator(gridSearch.best_estimator_, neuralNet)
            optimizer.fit(X_train,y_train)
            y_pred_f = optimizer.predict(X_test)
            y_pred = self.fuzzy_transform.defuzzy(X_test,y_pred_f)
            score_nn = np.sqrt(mean_squared_error(y_test[1:], y_pred))
            tmp = {
                'score':score_nn,
                'n_sliding':train_item.metadata['sliding_windows']
            }
            result.append(tmp)
        pd.DataFrame(result).to_csv('score_grid_exhaust.csv',index=None)

