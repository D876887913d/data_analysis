import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as datasets

def load_data():
    boston_data = datasets.load_boston()
    # data: 特征 target：标签 feature_name：特征列名 DESCR: 数据集介绍  filename: 文件名  data_module：调用的基类
    # dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename', 'data_module'])

    feature = boston_data['data']
    label = boston_data['target']
    columns = boston_data['feature_names']

    # 用处不大的几个键值对
    # descr = boston_data['DESCR']
    # filename = boston_data['filename'] # boston_house_prices.csv
    # data_module = boston_data['data_module']

    boston_data_df = pd.DataFrame(feature, None, columns, )
    boston_label_df = pd.DataFrame(label, None, ['price'])

    return boston_data_df, boston_label_df

boston_data_df, boston_label_df = load_data()

# boston_data_df.plot()
# plt.show()

boston_data_df = boston_data_df[['DIS', 'TAX']]
boston_all = boston_data_df.join(boston_label_df)
print(boston_all.head())

# ==========================移动窗口均值平滑============================
# ax = boston_all.plot()
# boston_all.rolling(40).mean().plot(ax = ax)
# boston_all.rolling(40).sum().plot()
# boston_all.rolling(40).std().plot(ax = ax)
# boston_all.expanding().mean().plot(ax = ax)

# EWM算法进行普通移动平滑
# boston_all = boston_all.ewm(span=30).mean().plot(ax = ax)

# boston_corr = boston_all.rolling(125, min_periods=100).corr(boston_all['price']).plot()
# plt.show()

# print(boston_corr)
# =======================移动窗口均值平滑结束============================

