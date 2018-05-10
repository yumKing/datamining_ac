import pandas as pd

# # ==================剔除无效数据 =================
# # data = pd.read_excel('data/air_data.xls',index_col='MEMBER_NO',dtype={'FFP_DATE':str,'FIRST_FLIGHT_DATE':str,'LOAD_TIME':str,'LAST_FLIGHT_DATE':str})
# data = pd.read_csv('data/air_data.csv',encoding='utf-8',index_col='MEMBER_NO')
# # 票价非空的数据保留

# data = data[(data['SUM_YR_1'].notnull()) & (data['SUM_YR_2'].notnull())]
# # data.fillna(0)
# # 只保留票价非零的，或者平均折扣率与总飞行里程数同时记录为0的记录
# index1 = data['SUM_YR_1'] != 0
# index2 = data['SUM_YR_2'] != 0
# index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0)  # '与'规则
# data = data[index1 | index2 | index3]  # '或'规则

# # a['z'] = pd.to_datetime(a['z'],format='%Y%m%d')
# data.to_csv('data/cleaned_data.csv')
# # ========  END ============

# ============提取相关特征 ,属性变换============
data = pd.read_csv('data/cleaned_data.csv',encoding='utf-8')

# data.drop(['MEMBER_NO'], axis=1, inplace=True)

data['L'] = pd.to_datetime(data['LOAD_TIME'],format='%Y/%m/%d') - pd.to_datetime(data['FFP_DATE'],format='%Y/%m/%d')
# map函数
# data['L'] = data['L'].map(lambda x:format(x.days/30,'.2f')) # 不能四舍五入
data['L'] = data['L'].map(lambda x:round(x.days/30,2)) # 四舍五入
data['avg_discount'] = data['avg_discount'].round(2)

data.rename(columns={'LAST_TO_END':'R','FLIGHT_COUNT':'F','SEG_KM_SUM':'M','avg_discount':'C'}, inplace = True)
data = data[['L','R','F','M','C']]
# =============== END ==================

# =================数据标准化： 使用标准差标准化===================
# mean函数计算每一列的平均值，std函数计算每一列的标准差 ,
# axis表示以列或行为轴心 ，0：行，1：列
data = (data - data.mean(axis=0))/data.std(axis=0)
data = data.round(2)
# data.columns = ['Z'+i for i in data.columns]
data.columns = list(map(lambda x:'Z'+x,data.columns))
# ================ END =================

data.to_csv('data/model.csv',index=False)
