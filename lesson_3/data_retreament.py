import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_excel('data/data.xls')

processed_file = 'data/data_processed.xls'

typelabel = {'肝气郁结证型系数': 'A', '热毒蕴结证型系数': 'B', '冲任失调证型系数': 'C',
             '气血两虚证型系数': 'D', '脾胃虚弱证型系数': 'E', '肝肾阴虚证型系数': 'F'}

k = 4 # 聚类的类别数
keys = list(typelabel.keys())

result = pd.DataFrame()

if __name__ == '__main__':
	for i in range(len(keys)):
		print('正在进行"%s"的聚类...' % keys[i])
		kmodel = KMeans(n_clusters=k,n_jobs=1)
		kmodel.fit(data[[keys[i]]].as_matrix())

		# 聚类中心
		r1 = pd.DataFrame(kmodel.cluster_centers_,columns=[typelabel[keys[i]]])

		# 分类统计
		r2 = pd.Series(kmodel.labels_).value_counts()
		# 转换为DataFrame对象，记录各个类别的数目
		r2 = pd.DataFrame(r2,columns=[typelabel[keys[i]]+'n'])

		# 匹配聚类中心和类别数目
		r = pd.concat([r1,r2],axis=1).sort_values(typelabel[keys[i]])
		r.index = [1,2,3,4]

		# 计算相邻2列的均值作为边界值,并将原来的聚类中心改为边界值
		r[typelabel[keys[i]]]  = r[typelabel[keys[i]]].rolling(window=2,center=False).mean()
		r.loc[1,typelabel[keys[i]]] = 0.0
		r = r.round(3)

		result = result.append(r.T)
	# 以index排序
	result.sort_index()
	result.to_excel(processed_file)