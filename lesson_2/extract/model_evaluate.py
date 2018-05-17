from sklearn.metrics import roc_curve

from keras.models import load_model
from random import shuffle

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib

data = pd.read_csv('../data/extract_file/data_extract_fl.csv',encoding='utf-8')
data = data.as_matrix()
shuffle(data)

p = 0.8
train = data[:int(len(data)*p),:]
test = data[int(len(data)*p):,:]

# net = load_model('../data/extract_file/net.model')
# predict_result = net.predict(test[:, :7]).reshape(len(test))
# fpr,tpr,thresholds=roc_curve(test[:,7],predict_result,pos_label=1)

# plt.plot(fpr,tpr,linewidth=2,label='LM OF ROC')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive  Rate')
# plt.ylim(0, 1.05)  # 边界范围
# plt.xlim(0, 1.05)  # 边界范围
# plt.legend(loc=4)  # 图例
# plt.show()

tree = joblib.load('../data/extract_file/tree.pkl')
predict_result = tree.predict_proba(test[:,:7])[:,1]
fpr,tpr,thresholds=roc_curve(test[:,7],predict_result,pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label='ROC of CART')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.legend(loc=4)
plt.show()