import numpy as np
import pandas as pd
from estimators.BruteForceGridSearchACO import BruteForceGridSearch
from sklearn.preprocessing import MinMaxScaler

print "Loading data"
scaler = MinMaxScaler()
data = scaler.fit_transform(pd.read_csv('sample_610_10min.csv',index_col=0)['cpu_rate'])

bruteGrid = BruteForceGridSearch(n_sliding_ranges=np.arange(2,11))
bruteGrid.transform(data)
bruteGrid.fit(data=data)
