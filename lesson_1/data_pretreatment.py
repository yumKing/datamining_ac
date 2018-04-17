import pandas as pd  # 导入数据分析库，读取excel数据
from scipy.interpolate import lagrange  # 导入拉格朗日方法

inputfile = 'data/missing_data.xls'
outfile = 'data/repaired_data.xls'

# 读取xls
data = pd.read_excel(inputfile, header=None)
# 自定义列向量插值函数
# s为列向量，n为被插值的位置，k为取前后数据的个数，默认k为5


def ployinterp_column(s, n, k=5):
    # 取出要插值位置的前后k个数据
    y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]
    # 剔除空值
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)


# 逐个元素判断是否需要插值
for i in data.columns:
    for j in range(len(data)):
        # 如果该值为空，则需要插值
        if(data[i].isnull())[j]:
            data.loc[j, i] = ployinterp_column(data[i], j)

data.to_excel(outfile, header=None, index=False)
