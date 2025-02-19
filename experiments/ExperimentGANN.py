# Experiment GABPNN
from estimators.GAEstimator import GAEstimator
from estimators.NeuralFlow import NeuralFlowRegressor
from io_utils.GFeeder import GFeeder
from utils.GraphUtil import *
from utils.initializer import *

# length of sliding windows for input
n_sliding_window = 2

#Getting Google cluster data
dataFeeder = GFeeder(skip_lists=1,normalize_space=True)
metric_type = [dataFeeder.CPU_UTIL]
# metric_type = [dataFeeder.MEM_USAGE]
# print metrics_types
# metrics_windows = {
#     dataFeeder.CPU_UTIL : 3,
#     dataFeeder.DISK_IO_TIME,
#     dataFeeder.DISK_SPACE,
#
# }

print "Getting data"
X_train,y_train,X_test,y_test = dataFeeder.split_train_and_test(metrics=metric_type,n_sliding_window=n_sliding_window)
# Number of hiddens node (one hidden layer)
score_print = []
for i in range(10):
    score_list = {}
    for n_hidden in np.arange(120,300,step=10):
        # n_hidden = 200
        # Define neural shape
            # Input layer: [n_sample*n_size]
            # Hidden layer:
            # Output layer: regression
        neural_shape = [dataFeeder.input_size,n_hidden,dataFeeder.output_size]
        # Initialize GA Estimator
        estimator = GAEstimator(cross_rate=0.7,mutation_rate=0.04,pop_size=60,gen_size=100)
        fit_param = {'neural_shape':neural_shape}

        # Initialize neural network model for regression
        neuralNet = NeuralFlowRegressor()

        # There are many techniques for combining GA with NN. One of this, the optimizer solution of GA will be weights initialized of NN
        # optimizer = OptimizerNNEstimator(estimator,neuralNet)
        optimizer = neuralNet
        optimizer.fit(X_train,y_train,**fit_param)
        score = optimizer.score(X_test,y_test)
        print score
        score_list[n_hidden]=score
        # if score < 0.01:
        y_pred = optimizer.predict(X_test)

        # plot_metric_figure(y_pred=y_pred,y_test=y_test, metric_type=dataFeeder.metrics,title="GANN")
        # plot_metric_figure(y_pred=y_pred,y_test=y_test,metric_type=metrics_types,title=" GANN ")
        optimizer.save("params/model_full_metric_%s"%score)
    score_list = pd.Series(score_list)
    print "Optimal hidden nodes: %s, with score = %s"%(score_list.argmin(),score_list.min())
    tmp = []
    for k, metric in enumerate(metric_type):
        tmp.append("%s = %s"%(metric_type[k],mean_squared_error(y_pred[:, k], y_test[:, k])))
    score_print.append(tmp)
print score_print