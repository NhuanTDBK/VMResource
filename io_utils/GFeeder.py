from utils.SlidingWindowUtil import SlidingWindow
from __init__ import *
class GFeeder:
    def __init__(self,skip_lists=1,split_size=None):
        self.skip_lists = skip_lists
        if(split_size!=None):
            self.split_size = split_size
        self.result = {}
        self.metric_type = [
            u'cpu_usage', u'disk_io_time', u'disk_space', u'mem_usage',
        ]
        self.file_name = "data/gdata/gcluster_1268205_1min.json"
    def read(self,metrics=None):
        if (metrics==None):
            metrics = self.metric_type
        return pd.read_json(self.file_name)[metrics][:1152614]
    def fetch_metric_train(self,data,n_sliding_window,range_fetch):
        from_range = range_fetch[0]
        to_range = range_fetch[1]
        range_data=None
        if(from_range==-1):
            range_data = data[:to_range]
        else:
            range_data = data[from_range:to_range]
        result = list(SlidingWindow(range_data,n_sliding_window))
        return result
    def fetch_metric_test(self, data, n_sliding_window, range_fetch):
        from_range = range_fetch[0]
        to_range = range_fetch[1]
        range_data=None
        if(to_range==-1):
            range_data = data[from_range + n_sliding_window:].reshape(-1, 1)
        else:
            range_data = data[from_range + n_sliding_window:to_range].reshape(-1, 1)
        result = list(range_data)
        return result
    def _fetch(self,n_sliding_window,range_fetch=None):
        data_fetch_X = []
        data_fetch_y = []
        for column,data in self.result.iteritems():
            # data = self.average_metric(data,skip_lists=self.skip_lists)
            data_fetch_X.append(self.fetch_metric_train(data, n_sliding_window,range_fetch))
            data_fetch_y.append(self.fetch_metric_test(data, n_sliding_window,range_fetch))
        X_test = np.asarray([np.array(t, dtype=np.float32).flatten().tolist() for t in zip(*data_fetch_X)])
        y_test = np.asarray([np.array(t).flatten().tolist() for t in zip(*data_fetch_y)])
        return X_test, y_test
    def split_train_and_test(self,metrics=None,n_sliding_window=4,train_size = 0.7):
        self.result = self.read(metrics)
        length_data = self.result.shape[0]
        point = int(length_data*train_size)
        range_train = (-1,point)
        range_test = (point,-1)
        X_train, y_train = self._fetch(n_sliding_window,range_train)
        X_test, y_test = self._fetch(n_sliding_window,range_test)
        return X_train, y_train,  X_test, y_test