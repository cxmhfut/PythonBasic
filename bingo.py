# coding:utf-8

from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']


def bingo(x):
    return 1 - 0.998 ** x


x = np.linspace(0, 2000, 500)
y = bingo(x)
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
# 设置显示边界
plt.xlim(0, 2000)
plt.ylim(0, 1)
# 将上方和下方的边界隐藏
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
# 将下边界设置为x轴并移动到数据0的位置
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])  # 设置坐标轴刻度
# 将左边界设置为y轴并移动到数据0的位置
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.plot(x, y, color='blue')
plt.title("Bingo")
plt.xlabel("Bingo宝石个数")
plt.ylabel("最后一个数出现概率")
plt.legend()
plt.show()
