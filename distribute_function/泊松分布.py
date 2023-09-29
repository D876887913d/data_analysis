import numpy as np
import matplotlib.pyplot as plt

# 设置平均发生率
lambda_ = 3

# 生成泊松分布随机样本
data = np.random.poisson(lambda_, 1000)


# 配置Matplotlib支持中文字符
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 绘制直方图
plt.hist(data, bins=range(0, 12), density=True, alpha=0.6, color='r')
plt.title('泊松分布')
plt.xlabel('发生次数')
plt.ylabel('概率密度')
plt.show()
