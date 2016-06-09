import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.SlidingWindowUtil import SlidingWindow
from io_utils.GFeeder import GFeeder
from estimators.NeuralFlow import NeuralFlowRegressor
from utils.GraphUtil import *
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dat = pd.read_csv('../sample_610_10min.csv',index_col=0,parse_dates=True)
training_size = 3000
test_size = 600
gFeeder = GFeeder()
dat.cpu_rate = np.array(scaler.fit_transform(dat.cpu_rate))
ax = dat.cpu_rate.plot()
plt.show()

X_dat = np.array(list(SlidingWindow(dat.cpu_rate,sliding_number=4)))

X_train = X_dat[:training_size]
y_train = np.array(dat.cpu_rate[:training_size].tolist()).reshape(-1,1)
X_test = X_dat[training_size + 1:training_size + 1 + test_size]
y_test = np.array(dat.cpu_rate[training_size + 1:training_size + test_size + 1].tolist()).reshape(-1,1)

nn = NeuralFlowRegressor(learning_rate=1E-03,hidden_nodes=np.array([55]))
nn.fit(X_train,y_train)
y_pred = nn.predict(X_test)
# plot_figure(y_pred=y_pred,y_true=y_test,title="Neural Flow sliding window 4")
