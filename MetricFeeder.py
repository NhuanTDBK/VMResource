from __init__ import *
from sklearn.preprocessing import MinMaxScaler
from SlidingWindowUtil import SlidingWindow


class MetricFeeder:
    def __init__(self,split_size=None):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        if(split_size!=None):
            self.split_size = split_size
        # self.data = pd.read_hdf(self.metric_type.get(metric_type))["Volume"]
        self.result = {}
        self.metric_type = {
            "cpu_util": "sample_cpu_util.json",
            "disk_write_rate": "sample_disk_write.json",
            "disk_read_rate": "sample_disk_read.json",
            "network_incoming_rate": "sample_network_incoming.json",
            "network_outgoing_rate": "sample_network_outgoing.json"
        }

    def fetch(self, metrics, n_sliding_window, range_fetch):
        data_fetch_X = []
        data_fetch_y = []
        for metric in metrics:
            data = pd.read_json(self.metric_type[metric])["Volume"]
            self.result[metric] = data
            data_fetch_X.append(self.fetch_metric_train(data, n_sliding_window, range_fetch))
            data_fetch_y.append(self.fetch_metric_test(data, n_sliding_window, range_fetch))
        X_test = np.asarray([np.array(t, dtype=np.float32).flatten().tolist() for t in zip(*data_fetch_X)])
        y_test = np.asarray([np.array(t).flatten().tolist() for t in zip(*data_fetch_y)])
        return X_test, y_test

    def fetch_metric_train(self, data, n_sliding_window, range_fetch):
        from_range = range_fetch[0]
        to_range = range_fetch[1]
        range_data = None
        if(from_range==-1):
            range_data = data[:to_range].reshape(-1, 1)
        else:
            range_data = data[from_range:to_range].reshape(-1, 1)
        result = list(SlidingWindow(
            self.scaler.fit_transform(range_data),
            n_sliding_window))
        return result

    def fetch_metric_test(self, data, n_sliding_window, range_fetch):
        from_range = range_fetch[0]
        to_range = range_fetch[1]
        if(to_range==-1):
            range_data = data[from_range + n_sliding_window:].reshape(-1, 1)
        else:
            range_data = data[from_range + n_sliding_window:to_range].reshape(-1, 1)
        result = list(
            self.scaler.fit_transform(range_data))
        return result

    def convert(self, data_scale, type="cpu_util"):
        min = self.result[type].min()
        max = self.result[type].max()
        return (data_scale - min) / (max - min)
