import pandas as pd
from random import shuffle

from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.models import load_model

import sys
sys.path.append('../')  # 将路径目录添加到系统环境变量 path 下
import cm_plot

# -----------------------------------------------------------------------
# init
data = pd.read_excel('../data/拓展思考样本数据/拓展思考样本数据_qe.xls', index_col=u'纳税人编号')
# 转换为矩阵
data = data.as_matrix()
# 打乱数据，随机分配
shuffle(data)

# 设置训练测试比例
p = 0.8
train = data[:int(len(data) * p), :]
test = data[int(len(data) * p):, :]
# -----------------------------------------------------------------------

# 搭建LM模型,模型设置：14-10-1
# net = Sequential()
# net.add(Dense(input_dim=14,units=10))
# net.add(Activation('relu'))
# net.add(Dense(input_dim=10,units=1))
# net.add(Activation('sigmoid'))
# # 编译模型，使用adam求解
# net.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# #训练模型
# net.fit(train[:,:14],train[:,14],epochs=100,batch_size=1)
# #保存模型
# net.save('../data/拓展思考样本数据/net.pkl')

net = load_model('../data/拓展思考样本数据/net.pkl')

# 预测
predict_result = net.predict_classes(train[:, :14]).reshape(len(train))

# 混淆矩阵检测
cm_plot.cm_plot(train[:, 14], predict_result).show()
