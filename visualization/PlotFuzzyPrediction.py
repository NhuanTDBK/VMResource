import numpy as np
import seaborn as sb
from utils.GraphUtil import *

datr = np.load('model_saved/Experiment_7_6_stage_2/0.00809692881851.npz')
ypred = datr['y_pred']
ytest = datr['y_test']
ypred[ypred<0]=0
# print ypred.shape
plot_figure(y_pred=ypred,y_true=ytest,title="Fuzzy Time Series with sliding window 4 next 10-min RMSE 0.00636")

