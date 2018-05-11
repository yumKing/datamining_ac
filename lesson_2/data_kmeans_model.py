import pandas as pd
from sklearn.cluster import KMeans
from sklearn.externals import joblib

data = pd.read_csv('data/model.csv', encoding='utf-8')

# 聚类类别数
k = 5

# k-means 算法,n_clusters表示需要聚类的类别数，n_jobs表示并行数(等于CPU数较好)
kmodel = KMeans(n_clusters=k, n_jobs=1, verbose=1)
# 训练模型
kmodel.fit(data)


# joblib.dump(kmodel, 'data/km.pkl')

# ==========简单输出================
# 统计各类别聚类数
r1 = pd.Series(kmodel.labels_).value_counts()
# 获取聚类中心
r2 = pd.DataFrame(kmodel.cluster_centers_)
# 横向链接，得到聚类中心对应的类别下的数目
r = pd.concat([r1,r2],axis=1)
# 重命名表头
r.columns =['类别数目'] + list(data.columns)
r.to_csv('data/kmeans_info.csv',index=False)
# ==================================

# ========详细输出===================
# km = pd.concat([data,pd.Series(kmodel.labels_,index=data.index)],axis=1)
# km.columns = list(data.columns) +['聚类类别']
# km.to_csv('data/kmeans_reslut.csv',index=False)