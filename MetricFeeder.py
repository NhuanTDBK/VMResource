from __init__ import *
from sklearn.preprocessing import MinMaxScaler
from SlidingWindowUtil import SlidingWindow
class MetricFeeder:
    def __init__(self,metric_type):
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.data = pd.read_hdf(self.metric_type.get(metric_type))["Volume"]
        self.max = self.data.max()
        self.min = self.data.min()
        self.metric_type = {
            "cpu_util":"sample_cpu_util",
            "disk_write_rate":"sample_disk_write",
            "disk_read_rate":"sample_disk_read"
        }

    def fetch(self,n_sliding_window,range_fetch):
        from_range = range_fetch[0]
        to_range = range_fetch[1]
        result = list(SlidingWindow(
                        self.scaler.fit_transform(self.data[from_range:to_range].reshape(-1,1)),
                        n_sliding_window))
        return result
    def convert(self,data_scale):
        return (data_scale-self.min)/(self.max-self.min)
