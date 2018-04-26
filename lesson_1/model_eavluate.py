# 导入ROC曲线函数
from sklearn.metrics import roc_curve

import pandas as pd
from random import shuffle  # 导入随机函数，用来打乱数据
from sklearn.externals import joblib
from keras.models import load_model
import matplotlib.pyplot as plt

datafile = 'data/model.xls'
data = pd.read_excel(datafile)
data = data.as_matrix()  # 将表格转为矩阵
shuffle(data)

# 设置训练数据比例
p = 0.8
train = data[:int(len(data) * p), :]
test = data[int(len(data) * p):, :]
# ================LM模型评估===========================
# 生成的模型路径
netfile = 'data/net.model'
net = load_model(netfile)
# 预测结果变形
predict_result = net.predict(test[:, :3]).reshape(len(test))
fpr, tpr, thresholds = roc_curve(test[:, 3], predict_result, pos_label=1)
# 画ROC曲线
plt.plot(fpr, tpr, linewidth=2, label='ROC of LM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive  Rate')
plt.ylim(0, 1.05)  # 边界范围
plt.xlim(0, 1.05)  # 边界范围
plt.legend(loc=4)  # 图例
plt.show()

# ==============CART决策树评估=========================
# treefile = 'data/tree.pkl'
# tree = joblib.load(treefile)
# fpr, tpr, thresholds = roc_curve(test[:, 3], tree.predict_proba(test[:, :3])[:, 1], pos_label=1)
# plt.plot(fpr, tpr, linewidth=2, label='ROC of CART')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.xlim(0, 1.05)
# plt.ylim(0, 1.05)
# plt.legend(loc=4)
# plt.show()
