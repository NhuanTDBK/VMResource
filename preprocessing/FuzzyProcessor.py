import pandas as pd
import numpy as np
import math
class FuzzyProcessor:
    #alpha is bonus difference between T(t-1) and T(t)
    def __init__(self,fuzzy_set_size=26,fuzzy_distance=0.02,alpha=0.25,automf=False):
        self.fuzzy_set_size = fuzzy_set_size
        self.fuzzy_distance = fuzzy_distance
        self.alpha = alpha
        self.automf = automf
    def fuzzy(self,training_set):
        length_x_train = len(training_set)
        difference = np.zeros(length_x_train - 1)
        for i in range(0, length_x_train - 1):
            difference[i] = training_set[i + 1] - training_set[i] + self.alpha
        max_val = difference.max()
        min_val = difference.min()
        if(self.automf==True):
            self.fuzzy_set_size = int(math.ceil(max_val/self.fuzzy_distance)+1)
        fuzzy_result = np.zeros([length_x_train - 1, self.fuzzy_set_size])
        self.fuzzy_set = np.zeros(self.fuzzy_set_size)
        for i in range(0, self.fuzzy_set_size):
            self.fuzzy_set[i] = self.fuzzy_distance * i + self.fuzzy_distance / 2
        for i in range(1, length_x_train - 2):
            # try:
            j = int(difference[i] / self.fuzzy_distance)
            fuzzy_result[i][j] = 1
            fuzzy_result[i][j - 1] = (self.fuzzy_set[j + 1] + self.fuzzy_set[j] - 2 * difference[i]) / (2 * self.fuzzy_distance)
            fuzzy_result[i][j + 1] = (- self.fuzzy_set[j - 1] - self.fuzzy_set[j] + 2 * difference[i]) / (2 * self.fuzzy_distance)
            # except Exception as e:
            #     print i,j
            #     print e
        return fuzzy_result
    def defuzzy(self, testing_set, training_result):
        length_y_test = len(testing_set)
        y_pred = np.zeros([length_y_test - 1])
        for i in range (0, length_y_test - 1):
            tu = 0.0
            mau = 0.0
            for j in range (0, self.fuzzy_set_size):
                tu = tu + self.fuzzy_set[j] *  training_result[i][j]
                mau = mau + training_result[i][j]
            difference = tu / mau - self.alpha
            y_pred[i] = testing_set[i] + difference
        return y_pred
    def fit_transform(self,dataset):
        return self.fuzzy(dataset)
