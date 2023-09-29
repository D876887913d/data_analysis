import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

# 读取数据，并将header设定为 None 
iris_df = pd.read_csv("iris/iris.data", header = None)
# print(iris_df.head())

# # 判断空缺值是否存在
# missing_values = iris_df.isnull().any()
# columns_with_missing_values = missing_values[missing_values == True]
# print(columns_with_missing_values)

# 拆分数据集, 分成标签及特征
X = iris_df.drop(4, axis=1)
y = iris_df[4]

# 初始化LabelEncoder
label_encoder = LabelEncoder()
y = pd.DataFrame(label_encoder.fit_transform(y))

# print(X.head())
# print(y.head())

# print(y[0].unique())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 设置LightGBM的参数
params = {
    'objective': 'multiclass',
    'num_class': 3,  # 三分类问题
    'boosting_type': 'gbdt',
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, train_data, num_boost_round=100)

# 预测
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_class = np.argmax(y_pred, axis=1)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred_class)
print(f'Accuracy: {accuracy}')
