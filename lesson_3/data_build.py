import pandas as pd

'''
根据聚类结果，进行标号
'''

typelabel = {'肝气郁结证型系数': 'A', '热毒蕴结证型系数': 'B', '冲任失调证型系数': 'C',
             '气血两虚证型系数': 'D', '脾胃虚弱证型系数': 'E', '肝肾阴虚证型系数': 'F'}
vss = list(typelabel.values())
keys = list(typelabel.keys())

data = pd.read_excel('data/data.xls')
rg = pd.read_excel('data/data_processed.xls')


for i in range(len(keys)):
    ks = keys[i] # dict -- key
    label = typelabel[ks] # dict -- value

    ixs = rg.loc[label,:] # Series
    sg = data[ks]

    data[label] = None

    data.loc[sg <= ixs[2], label] = label + '1'
    data.loc[(sg > ixs[2]) & (sg <= ixs[3]), label] = label + '2'
    data.loc[(sg > ixs[3]) & (sg <= ixs[4]), label] = label + '3'
    data.loc[sg > ixs[4], label] = label + '4'

data = data[vss]

data = data.rename(columns=dict(zip(vss,keys)))

data.to_excel('data/data_ml.xls',index=None)
