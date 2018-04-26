import pandas as pd
from random import shuffle  # 导入随机函数，用来打乱数据
# 构建CART决策树模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
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

treefile = 'data/tree.pkl'
# 建立决策树模型
tree = DecisionTreeClassifier()
tree.fit(train[:, :3], train[:, 3])
# 保存模型
joblib.dump(tree, treefile)

# 混淆矩阵可视化
cm_plot(train[:, 3], tree.predict(train[:, :3])).show()
