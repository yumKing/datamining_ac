import pandas as pd
from random import shuffle # 导入随机函数，用来打乱数据

datafile = 'data/model.xls'
data = pd.read_excel(datafile,header=None)
data = data.as_matrix() # 将表格转为矩阵
shuffle(data)

# 设置训练数据比例
p = 0.8
train = data[:int(len(data)*p),:]
test = data[int(len(data)*p):,:]
