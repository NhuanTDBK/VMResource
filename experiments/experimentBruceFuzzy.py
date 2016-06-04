import pandas as pd
from estimators.BruteForceGridSearch import BruteForceGridSearch
from sklearn.preprocessing import MinMaxScaler

print "Loading data"
scaler = MinMaxScaler()
data = scaler.fit_transform(pd.read_csv('data/gdata/sampling_617685_metric_1min_datetime.csv',index_col=0)['cpu_rate'])

bruteGrid = BruteForceGridSearch(n_sliding_ranges=[2,10])
bruteGrid.transform(data)
bruteGrid.fit(data=data)
