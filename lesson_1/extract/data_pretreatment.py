import pandas as pd
import matplotlib.pyplot as plt

'''
数据预处理，针对特征进行数值化处理，是使用map()方法
'''
data = pd.read_excel('../data/拓展思考样本数据.xls', index_col=u'纳税人编号')

# 将输出特征转换为0=正常，1=异常
data[u'输出'] = data[u'输出'].map({u'正常': 0, '异常': 1})
# 将输入特征转换
data[u'销售类型'] = data[u'销售类型'].map(
    {u'国产轿车': 1, u'进口轿车': 2, u'大客车': 3, u'卡车及轻卡': 4, u'微型面包车': 5, u'商用货车': 6, u'工程车': 7, u'其它': 8})
data[u'销售模式'] = data[u'销售模式'].map(
    {u'4S店': 1, u'一级代理商': 2, u'二级及二级以下代理商': 3, u'多品牌经营店': 4, u'其它': 5})

data.to_excel('../data/拓展思考样本数据/拓展思考样本数据_qe.xls')
