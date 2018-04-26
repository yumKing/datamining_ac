import pandas as pd
from random import shuffle  # 导入随机函数，用来打乱数据
# 构建LM神经网络模型
from keras.models import Sequential  # 导入神经网络初始化函数
from keras.layers.core import Dense, Activation  # 导入神经网络层函数，激活函数
# 混淆矩阵可视化结果
from lesson_1.cm_plot import *

datafile = 'data/model.xls'
data = pd.read_excel(datafile)
data = data.as_matrix()  # 将表格转为矩阵
shuffle(data)

# 设置训练数据比例
p = 0.8
train = data[:int(len(data) * p), :]
test = data[int(len(data) * p):, :]

# 生成的模型路径
netfile = 'data/net.model'
net = Sequential()  # 建立神经网络
net.add(Dense(input_dim=3, units=10))  # 添加输入层(3节点)到隐藏层(10节点)的链接
net.add(Activation('relu'))  # 隐藏层使用relu激活函数
net.add(Dense(input_dim=10, units=1))  # 添加隐藏层(10节点)到输出层(1节点)的链接
net.add(Activation('sigmoid'))  # 输出层使用sigmoid激活函数
# 编译模型，使用adam方法求解
net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型 循环n次
net.fit(train[:, :3], train[:, 3], epochs=100, batch_size=1)
# 保存模型
net.save_weights(netfile)

# 预测结果变形
# keras 用predict给出预测概率，predict_classes给出预测类别，预测结果是n*1维数组
predict_result = net.predict_classes(train[:, :3]).reshape(len(train))

cm_plot(train[:, 3], predict_result).show()
