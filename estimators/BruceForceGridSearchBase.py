import pickle
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
class BruteForceGridSearchBase():
    def __init__(self,n_sliding_ranges,fuzzy_transform=FuzzyProcessor(automf=True,fuzzy_distance=0.02)):
        self.n_sliding_ranges = n_sliding_ranges
        self.fuzzy_transform = fuzzy_transform
    def transform(self,data_source):
        self.data_source = data_source
        self.data_transform = self.fuzzy_transform.fit_transform(data_source)
        self.train_bucket = []
        self.data_placeholder = []
        for n_sliding_window in self.n_sliding_ranges:
            print "Normalize complete"
            #training_set_size = int(self.data_transform.shape[0] * 0.7)
            training_set_size = 3000
            self.train_len = training_set_size
                #testing_set_size = int(self.data_transform.shape[0] * 0.3)-2
            testing_set_size = 600
            self.test_len = testing_set_size
            print "Preparing sliding window dataset..."
            X_dat = np.array(list(SlidingWindow(self.data_transform, n_sliding_window, concatenate=True)))
            X_train = X_dat[:training_set_size]
            y_train = self.data_transform[1:training_set_size + 1]
            X_test =  X_dat[training_set_size+1:training_set_size+1+testing_set_size]
            y_test = data_source[training_set_size + 1:training_set_size + testing_set_size + 1]
            metadata = {
                "sliding_windows":n_sliding_window
            }
            trainObject = TrainObject(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,metadata=metadata)
            self.train_bucket.append(trainObject)