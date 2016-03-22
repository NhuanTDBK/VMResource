import matplotlib.pyplot as plt
import pandas as pd

def plot_figure(y_pred=None,y_true=None,color=['blue','red']):
    ax = plt.subplot()
    ax.set_color_cycle(color)
    ax.plot(y_pred,'--',label='Predict')
    ax.plot(y_true,label='Actual')
    ax.legend()
    plt.show()