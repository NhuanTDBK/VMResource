import pickle
import numpy as np
import pandas as pd
from estimators.GAEstimator import GAEstimator
from estimators.NeuralFlow import NeuralFlowRegressor
from estimators.OptimizerNNEstimator import OptimizerNNEstimator
from sklearn.grid_search import GridSearchCV, delayed
from utils.SlidingWindowUtil import SlidingWindow
from estimators.TrainObject import TrainObject
from preprocessing.FuzzyProcessor import FuzzyProcessor
from sklearn.metrics import mean_squared_error
from estimators.BruceForceGridSearchBase import *
class BruteForceGridSearch(BruteForceGridSearchBase):
    def fit(self,data=None):
        result = []
        for train_item in self.train_bucket:
            X_train, y_train, X_test, y_test = train_item.getitems()
            n_hidden = np.array([55])
            param_dict = {
                "cross_rate": [0.65, 0.7],
                "pop_size": [45,50,60],
                "mutation_rate": np.arange(0.01, 0.04, step=0.01)
            }
            estimator = GAEstimator(cross_rate=0.7, mutation_rate=0.04, pop_size=60, gen_size=100)
            neuralNet = NeuralFlowRegressor(learning_rate=1E-03, hidden_nodes=n_hidden)
            neural_shape = [X_train.shape[1], n_hidden[0],y_train.shape[1]]
            fit_param = {'neural_shape': neural_shape}
            print "Preparing for grid search"
            gridSearch = GridSearchCV(estimator,param_dict,n_jobs=-1,fit_params=fit_param,scoring='mean_squared_error',verbose=1)
            gridSearch.fit(X_train,y_train)
            optimizer = OptimizerNNEstimator(gridSearch.best_estimator_, neuralNet)
            optimizer.fit(X_train,y_train)
            X_test_f = self.data_source[self.train_len+1:self.train_len+self.test_len+1]
            y_pred_f = optimizer.predict(X_test)
            y_pred = self.fuzzy_transform.defuzzy(X_test_f,y_pred_f)
            score_nn = np.sqrt(mean_squared_error(y_test[1:], y_pred))
            tmp = {
                'score':score_nn,
                'n_sliding':train_item.metadata['sliding_windows'],
		'best_estimator':'%s'%gridSearch.best_estimator_
            }
            result.append(tmp)
        np.savez('model_saved/%s'%score_nn,y_pred=y_pred,y_test=y_test[1:])
        optimizer.save('model_saved/%s_model'%score_nn)
        pd.DataFrame(result).to_csv('score_grid_exhaust_ga.csv',index=None)

