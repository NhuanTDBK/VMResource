import numpy as np
import matplotlib.pyplot as plt
from utils.GraphUtil import *

datr = np.load('../model_saved/0.00724502031802.npz')
ypred = datr['y_pred']
ytest = datr['y_test']
plot_figure(y_pred=ypred,y_true=ytest,title="Fuzzy Time Series with sliding window 5")

