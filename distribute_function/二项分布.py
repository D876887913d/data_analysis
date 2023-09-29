import numpy as np
import matplotlib.pyplot as plt

# 设置试验次数和成功概率
n = 10  # 试验次数
p = 0.3  # 成功概率

# 生成二项分布随机样本
data = np.random.binomial(n, p, 1000)

# 配置Matplotlib支持中文字符
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 绘制直方图
plt.hist(data, bins=range(0, n+2), density=True, alpha=0.6, color='b')
plt.title('二项分布')
plt.xlabel('成功次数')
plt.ylabel('概率密度')
plt.show()
