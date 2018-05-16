import pandas as pd
from random import shuffle
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.models import load_model

import sys
sys.path.append('../')  # 将路径目录添加到系统环境变量 path 下
import cm_plot

data = pd.read_csv(
    '../data/extract_file/data_extract_fl.csv', encoding='utf-8')
data = data.as_matrix()
shuffle(data)
p = 0.8
train = data[:int(len(data) * p), :]
test = data[int(len(data) * p):, :]


# net = Sequential()
# net.add(Dense(input_dim=7, units=10))
# net.add(Activation('relu'))
# net.add(Dense(input_dim=10, units=1))
# net.add(Activation('sigmoid'))

# net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# net.fit(train[:,:7],train[:,7],epochs=100,batch_size=1)

# net.save('../data/extract_file/net.model')
net = load_model('../data/extract_file/net.model')

# 预测结果变形
# keras 用predict给出预测概率，predict_classes给出预测类别，预测结果是n*1维数组
predict_result = net.predict_classes(train[:, :7]).reshape(len(train))

cm_plot.cm_plot(train[:, 7], predict_result).show()