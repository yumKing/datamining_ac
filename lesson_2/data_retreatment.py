'''
对数据进基本探索
返回缺失值个数及最小最大值
'''

import pandas as pd

file_path = 'data/air_data.xls'
result_path = 'data/retreatment.xls'

data = pd.read_excel(file_path)

# percentiles:指定计算多少的分位数表(如 1/4分位数，中位数)
retreatment = data.describe(percentiles=[], include='all').T
# count表示的是非空的数量
retreatment['null'] = len(data) - retreatment['count']

retreatment = retreatment[['null', 'min', 'max']]
retreatment.columns = ['空值', '最小值', '最大值']
retreatment.to_excel(result_path)
