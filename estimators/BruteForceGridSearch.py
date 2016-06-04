import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV, delayed
from utils.SlidingWindowUtil import SlidingWindow

class BruteForceGridSearch():
    def __init__(self,n_sliding_ranges):
        self.n_sliding_ranges = n_sliding_ranges
    def transform(self,data_source):
        self.data_source = data_source
        self.data_placeholder = []
        # for n_sliding_window in self.n_sliding_ranges:





