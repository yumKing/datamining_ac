from sklearn.metrics import roc_curve

from keras.models import load_model
from random import shuffle

import pandas asp pd

data = pd.read_csv('../data/extract_file/data_extract_fl.csv',encoding='utf-8')
data = data.as_matrix()
shuffle(data)

p = 0.8