import pandas as pd
import numpy as np

range_step = np.arange(0.6*1E12,0.95*1E12,step=1E6)
#range_moment = np.arange(90900*1E6,1955099*1E6,1E6)

file_name = ""
dat = pd.read_csv(file_name)
dat_extracted = dat[(dat.moment>=0.6*1E12)|(dat.moment<=0.95*1E12)].to_json('metric',orient='records')
sample_dat = pd.DataFrame(np.zeros((dat_extracted.shape[0]/60,dat_extracted.shape[1])),columns=dat_extracted.columns)
for i in range(sample_dat.shape[0]):
    sample_dat.iloc[i] = dat_extracted.iloc[i*60]
sample_dat.to_json('metric_77_1minute.json',orient='records')




