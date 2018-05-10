from sklearn.metrics import roc_curve  # ROC

import pandas as pd  # 加载xls文件的库
from random import shuffle  # 对数据打乱的库
from keras.models import load_model  # 预读LM模型的库
from sklearn.externals import joblib  # 预读决策树模型的库
import matplotlib.pyplot as plt  # 画图库

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
#===============LM神经网络评估
net = load_model('../data/拓展思考样本数据/net.pkl')
predict_result = net.predict(test[:, :14]).reshape(len(test))
fpr, tpr, thresholds = roc_curve(test[:, 14], predict_result, pos_label=1)

plt.plot(fpr, tpr, linewidth=2, label='ROC of LM')
plt.xlabel('x False rate')
plt.ylabel('y True rate')
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.legend(loc=4)
plt.savefig("../data/拓展思考样本数据/LM_eval.png")


# ==============CART决策树评估=========================
# tree = joblib.load('../data/拓展思考样本数据/tree.pkl')
# fpr, tpr, thresholds = roc_curve(test[:, 14], tree.predict_proba(test[:, :14])[:, 1], pos_label=1)
# plt.plot(fpr, tpr, linewidth=2, label='ROC of CART')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.xlim(0, 1.05)
# plt.ylim(0, 1.05)
# plt.legend(loc=4)
# plt.savefig("../data/拓展思考样本数据/CART_eval.png")