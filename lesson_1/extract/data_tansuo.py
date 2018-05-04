import pandas as pd
import matplotlib.pyplot as plt

'''
数据探索
1、使用画图工具找出各个列的数据分布情况
2、对整体数据的探查，是否有缺失值，异常值

探索结果:
销售类型集中在国产轿车，进口轿车
销售模式集中在4s店，一级代理商
不存在缺失值，但是发现几个属性的最小值有为0的
'''

# 制定索引为纳税人编号
data = pd.read_excel(u'../data/拓展思考样本数据.xls', index_col=u'纳税人编号')
# 查看数据,默认返回前5条数据
# print(data.head())

# 创建画布,1行2列,第一列显示销售类型分布情况，第二列显示销售模式分布情况
fig, axes = plt.subplots(1, 2)
# 设置画布大小,宽=12 ，高=4
fig.set_size_inches(12, 4)
# 获取2列的轴对象
ax0, ax1 = axes.flat

# 统计该列元素重复个数，返回Series对象，索引为元素，值为重复个数
sale_type = data[u'销售类型'].value_counts()
sale_model = data[u'销售模式'].value_counts()

# 分类汇总后绘制水平柱状图,barh表示画水品柱状图
sale_type.plot(kind='barh', ax=ax0, figsize=(12, 4), title=u'销售类型分布情况')
sale_model.plot(kind='barh', ax=ax1, figsize=(12, 4), title=u'销售模式分布情况')

# 解决中文乱码及负号显示方框
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.show()
plt.savefig("../data/拓展思考样本数据/fenbu.png")

# 对数值变量进行统计描述
result = data.describe().T

result.to_excel('../data/拓展思考样本数据/describe.xls')

# print(data.describe().T)