import pandas as pd
import numpy as np
import json
import datasets
np.random.seed(42)

datapath = '/hdd3/sonia/honeygan/data.csv'
df = pd.read_csv(datapath)
print(df.isna().sum())
df.head()

dfs = df[(~df.single_cpe.isna())]
dfs.drop(['cpe', 'cpe_count', 'os', 'category'], axis=1, inplace=True)
print(dfs.isna().sum(), df.shape)
dfs.head()


sents = dfs.apply(lambda x: f"There is a server visible at IP {x['ip_str']}, port {x['port']}. \
Its operating system is {x['os_generic']} and it offers the {x['module']} service.", axis=1)
sents[:5]         

data = datasets.Dataset.from_dict({'input': [str(x) for x in range(len(sents))], 'output':sents.to_list()})
data[:5]

max([len(x) for x in data['output']]), sum([len(x) for x in data['output']])/len(data)


data.save_to_disk('/hdd3/mohammed/llama_qlora/llama_format_feb27.dat')

dfsent = pd.DataFrame(sents)
dfsent.to_csv('/hdd3/mohammed/llama_qlora/feb27.csv', index=True)

