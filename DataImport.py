
# coding: utf-8

# In[63]:

import pandas as pd
from pymongo import MongoClient,IndexModel, ASCENDING, DESCENDING
import os
# from mongoengine import *


# In[5]:

#Init collections
#Init collections
client = MongoClient()
db = client.gcluster
collection = db.task_usage
#index_task = IndexModel([("mID", ASCENDING),("stime",ASCENDING),("etime",ASCENDING)],name="helloworld")
#collection.create_indexes([index_task])
schema = pd.read_csv("schema_tu.csv").columns

for data in os.listdir('data'):
    print data
    reader = pd.read_csv("data/%s"%data,names=schema)
    collection.insert_many(reader.to_dict('records'))
#     print len(chunk.columns)



