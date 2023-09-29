import pandas as pd

# 创建一个包含分类特征的DataFrame
data = pd.DataFrame({'color': ['red', 'green', 'blue', 'red', 'blue']})

# 使用独热编码
data_encoded = pd.get_dummies(data, columns=['color'])

# 输出独热编码后的结果
print(data_encoded)
