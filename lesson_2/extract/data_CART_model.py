import pandas as pd

from random import shuffle

from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

import sys
sys.path.append('../')
import cm_plot

data = pd.read_csv('../data/extract_file/data_extract_fl.csv',encoding='utf-8')

data = data.as_matrix()

shuffle(data)

p = 0.8
train = data[:int(len(data)*p),:]
test = data[int(len(data)*p):,:]

tree = DecisionTreeClassifier()
tree.fit(train[:,:7],train[:,7])
joblib.dump(tree,'../data/extract_file/tree.pkl')

cm_plot.cm_plot(test[:,7],tree.predict(test[:,:7])).show()