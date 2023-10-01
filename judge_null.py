import pandas as pd

# 从CSV文件读取数据
data = pd.read_csv('your_csv_file.csv')

# 检查每一列是否有空缺值
missing_values = data.isnull().any()

# 显示有空缺值的列
columns_with_missing_values = missing_values[missing_values == True]
print(columns_with_missing_values)
