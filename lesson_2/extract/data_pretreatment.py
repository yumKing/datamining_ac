import pandas as pd

data = pd.read_csv('../data/extract_file/data_extract.csv', encoding='utf-8')

# =================
# 缺失值处理
# print(data.describe())
data = data[(data['SUM_YR_1'].notnull()) & (data['SUM_YR_2'].notnull()) & (
    data['P1Y_BP_SUM'].notnull()) & (data['L1Y_BP_SUM'].notnull()) & (data.loc[:,'P1Y_BP_SUM'] != 0) & (data.loc[:,'L1Y_BP_SUM'] != 0)]

# ================================================================
# 统计类别

data['FL'] = None
# 已流失客户数据
data.loc[data['P1Y_Flight_Count'] / data['L1Y_Flight_Count'] < 0.5, 'FL'] = 1

# 准流失客户
data.loc[(data['P1Y_Flight_Count'] / data['L1Y_Flight_Count'] >= 0.5)
         & (data['P1Y_Flight_Count'] / data['L1Y_Flight_Count'] < 0.9), 'FL'] = 2

# 未流失客户
data.loc[data['P1Y_Flight_Count'] / data['L1Y_Flight_Count'] >= 0.9, 'FL'] = 3
# ================================================================

# ====================
# 单位里程票价 (unit_price) = 总票价/总里程数，总里程数用里程积分表示
data['unit_price'] = (data['SUM_YR_1'] / data['P1Y_BP_SUM'] +
                      data['SUM_YR_2'] / data['L1Y_BP_SUM']) / 2

# 单位里程积分 (unit_integral) = 观测窗口总积分/观测窗口总里程数
data['unit_integral'] = data['BP_SUM'] / data['SEG_KM_SUM']
# ====================

# 只要指定特征属性
data_res = data[['FFP_TIER', 'AVG_INTERVAL', 'avg_discount', 'EXCHANGE_COUNT',
             'Eli_Add_Point_Sum', 'unit_price', 'unit_integral']]
data_res.rename(columns={'FFP_TIER': 'FR', 'AVG_INTERVAL': 'AL', 'avg_discount': 'AT', 'EXCHANGE_COUNT': 'ET',
                     'Eli_Add_Point_Sum': 'ES', 'unit_price': 'UE', 'unit_integral': 'UL'}, inplace=True)
# ===================
# 数据变换，非乘机积分总和属性值很大，需要标准化处理
data_res = (data_res - data_res.mean(axis=0)) / data_res.std(axis=0)
data_res = data_res.round(2)  # 保留2位小数
# ===================

data_res = pd.concat([data_res,data.loc[:,'FL']],axis=1)

data_res.to_csv('../data/extract_file/data_extract_fl.csv', index=None)
