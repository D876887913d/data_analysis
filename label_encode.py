from sklearn.preprocessing import LabelEncoder

# 创建一个包含类别特征的列表
categories = ['low', 'medium', 'high', 'low', 'medium']

# 初始化LabelEncoder
label_encoder = LabelEncoder()

# 使用LabelEncoder对类别进行编码
encoded_categories = label_encoder.fit_transform(categories)

# 输出编码后的结果
print(encoded_categories)
