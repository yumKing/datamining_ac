import pandas as pd
from random import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

import sys
sys.path.append('../')
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

tree_path = '../data/拓展思考样本数据/tree.pkl'
# tree = DecisionTreeClassifier()
# tree.fit(train[:,:14],train[:,14])
# joblib.dump(tree,tree_path)
tree = joblib.load(tree_path)

cm_plot.cm_plot(test[:,14],tree.predict(test[:,:14])).show()