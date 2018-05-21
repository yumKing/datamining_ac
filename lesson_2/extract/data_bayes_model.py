import pandas as pd
from random import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

import sys
sys.path.append('../')
import cm_plot

data = pd.read_csv('../data/extract_file/data_extract_mnb.csv', encoding='utf-8')

data = data.as_matrix()

shuffle(data)

# ==============================
p = 0.8
train = data[:int(len(data) * p), :]
test = data[int(len(data) * p):, :]
# ================================

mnb = MultinomialNB()
mnb.fit(train[:,:7],train[:,7])
joblib.dump(mnb,'../data/extract_file/mnb.pkl')

predict_result = mnb.predict(train[:,:7])
cm_plot.cm_plot(train[:,7],predict_result).show()